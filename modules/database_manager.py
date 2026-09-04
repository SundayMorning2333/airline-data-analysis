"""
数据库管理模块 - MySQL数据库操作功能

性能优化要点（相比原版）：
1. 批量插入：用 itertuples / df.values 替代 iterrows()，速度提升 10-20 倍
2. 显式事务：每 chunk_size 行提交一次，避免 autocommit 每行 fsync
3. 分块写入：DataFrame 切片，控制单次内存峰值
4. 分批删除：避免长事务持有大量行锁
5. 服务端游标 + fetchmany：避免一次性将百万行结果集载入内存
6. 超时配置：批量写操作使用更长的 write_timeout，杜绝超时断连
7. 批量导入模式：临时关闭 unique_checks / foreign_key_checks 进一步加速
"""

import datetime
import numpy as np
import pandas as pd
import pymysql
from pymysql.cursors import DictCursor, SSCursor
from dbutils.pooled_db import PooledDB
from config.settings import DB_POOL_CONFIG


# 批量操作默认参数
DEFAULT_CHUNK_SIZE = 10000          # 单次 executemany 行数
DEFAULT_DELETE_BATCH = 10000        # 单次 DELETE 行数
DEFAULT_FETCH_BATCH = 5000         # 流式查询每次 fetch 行数


class DatabaseManager:
    """MySQL数据库管理器，负责连接管理、表结构创建和数据操作。"""

    def __init__(self, host='localhost', port=3306, user='root', password='', database='airline_analysis'):
        """
        初始化数据库管理器。

        Parameters
        ----------
        host : str
            数据库主机地址
        port : int
            数据库端口
        user : str
            数据库用户名
        password : str
            数据库密码
        database : str
            数据库名称
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        # 连接池（线程安全），由 PooledDB 管理
        self._pool = None
        # 共享连接（向后兼容：现有代码可能仍直接使用 db_manager.connection.cursor()）
        self.connection = None

    def connect(self):
        """建立数据库连接（初始化连接池，并创建一个共享连接用于向后兼容）。"""
        try:
            # 注意：必须传 pymysql 模块本身，不能传 pymysql.connect（后者是 Connection 类，
            # 拥有 .connect 方法，会导致 PooledDB 的 threadsafety 检测误判为 0 并抛出
            # NotSupportedError("Database module is not thread-safe.")）
            self._pool = PooledDB(
                creator=pymysql,
                mincached=DB_POOL_CONFIG['mincached'],
                maxcached=DB_POOL_CONFIG['maxcached'],
                maxconnections=DB_POOL_CONFIG['maxconnections'],
                blocking=DB_POOL_CONFIG['blocking'],
                maxusage=DB_POOL_CONFIG['maxusage'],
                ping=DB_POOL_CONFIG['ping'],
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=True,
                connect_timeout=DB_POOL_CONFIG['connect_timeout'],
                read_timeout=DB_POOL_CONFIG['read_timeout'],
                write_timeout=DB_POOL_CONFIG['write_timeout'],
            )
            # 共享连接：供过渡期仍直接访问 db_manager.connection 的旧代码使用
            self.connection = self._pool.connection()
            return True
        except pymysql.Error as e:
            raise Exception(f"数据库连接失败: {e}")

    def disconnect(self):
        """关闭数据库连接（归还共享连接并关闭整个连接池）。"""
        if self.connection is not None:
            try:
                # SteadyDBConnection 不暴露 .open 属性，直接尝试 close()
                # 若连接已关闭，close() 不会重复关闭
                self.connection.close()
            except Exception:
                pass
            self.connection = None
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None

    def reconnect(self):
        """重新连接数据库（先断开再连接）。"""
        self.disconnect()
        return self.connect()

    def is_connected(self):
        """
        检查是否已连接。

        主动通过 ping 探活并触发 pymysql 的自动重连；任何异常均返回 False，不抛出。
        """
        if self._pool is None or self.connection is None:
            return False
        try:
            self.connection.ping(reconnect=True)
            return True
        except Exception:
            return False

    def ensure_connection(self):
        """
        确保数据库已连接；若未连接则尝试自动重连。

        供 UI 进入数据库管理页面或 MCP 工具调用前主动调用。

        Returns
        -------
        bool
            True 表示当前已连接可用；False 表示重连后仍不可用。
        """
        if self.is_connected():
            return True
        try:
            self.reconnect()
            return self.is_connected()
        except Exception:
            return False

    def execute_query_safe(self, sql, params=None):
        """
        安全执行查询 SQL（线程安全，带自动重连重试）。

        供 MCP 工具并发调用：每次从连接池获取独立连接，遇到
        OperationalError / InterfaceError 自动重试一次。

        Parameters
        ----------
        sql : str
            SQL 语句（使用 %s 占位符）
        params : tuple or list, optional
            SQL 参数

        Returns
        -------
        tuple
            (pandas.DataFrame, "") 成功时返回结果 DataFrame 与空错误信息；
            (None, str) 失败时返回 None 与错误信息。
        """
        if not self.ensure_connection():
            return (None, "数据库未连接")

        last_error = None
        for _ in range(2):  # 最多重试一次
            conn = None
            cursor = None
            try:
                conn = self._pool.connection()
                cursor = conn.cursor()
                if params is not None:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                results = cursor.fetchall()
                return (pd.DataFrame(results), "")
            except (pymysql.OperationalError, pymysql.InterfaceError) as e:
                # 连接级错误：关闭当前连接，下一轮从池中取新连接重试
                last_error = e
                continue
            except pymysql.Error as e:
                return (None, f"SQL 执行失败: {e}")
            except Exception as e:
                return (None, f"SQL 执行异常: {e}")
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass
                if conn is not None:
                    try:
                        conn.close()  # 归还到连接池
                    except Exception:
                        pass

        return (None, f"数据库异常（重试后仍失败）: {last_error}")

    def create_tables(self):
        """
        创建数据库表结构。

        创建以下表：
        - rfm_analysis: RFM分析结果表
        - customer_clusters: 客户分群表
        - member_data: 原始客户数据表
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        # 创建RFM分析结果表
        rfm_analysis_sql = """
        CREATE TABLE IF NOT EXISTS rfm_analysis (
            rfm_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            member_no VARCHAR(32) NOT NULL COMMENT '会员编号',
            batch_no VARCHAR(32) NOT NULL COMMENT '分析批次编号',
            r_value DECIMAL(12,2) COMMENT 'R值-最近一次消费距今天数',
            f_value DECIMAL(12,2) COMMENT 'F值-消费频率',
            m_value DECIMAL(12,2) COMMENT 'M值-消费总金额',
            analysis_date DATE NOT NULL COMMENT '分析执行日期',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
            UNIQUE KEY uk_member_batch (member_no, batch_no),
            INDEX idx_batch_no (batch_no),
            INDEX idx_analysis_date (analysis_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RFM分析结果表'
        """

        # 创建客户分群表
        customer_clusters_sql = """
        CREATE TABLE IF NOT EXISTS customer_clusters (
            cluster_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            member_no VARCHAR(32) NOT NULL COMMENT '会员编号',
            batch_no VARCHAR(32) NOT NULL COMMENT '聚类批次编号',
            cluster_label INT NOT NULL COMMENT '聚类标签编号',
            customer_type VARCHAR(32) COMMENT '客户类型',
            cluster_date DATE NOT NULL COMMENT '聚类执行日期',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
            UNIQUE KEY uk_member_cluster_batch (member_no, batch_no),
            INDEX idx_batch_no (batch_no),
            INDEX idx_cluster_label (cluster_label),
            INDEX idx_customer_type (customer_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户分群表'
        """

        # 创建原始客户数据表
        member_data_sql = """
        CREATE TABLE IF NOT EXISTS member_data (
            data_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '数据记录唯一标识',
            member_no VARCHAR(32) NOT NULL COMMENT '会员编号',
            batch_no VARCHAR(32) NOT NULL COMMENT '数据批次编号',
            ffp_date VARCHAR(32) COMMENT '常旅客计划入会日期',
            first_flight_date VARCHAR(32) COMMENT '第一次乘机日期',
            gender VARCHAR(8) COMMENT '性别',
            ffp_tier INT COMMENT '常旅客计划等级',
            work_city VARCHAR(64) COMMENT '工作城市',
            work_province VARCHAR(64) COMMENT '工作省份',
            work_country VARCHAR(32) COMMENT '工作国家',
            age INT COMMENT '年龄',
            load_time VARCHAR(32) COMMENT '数据加载时间',
            flight_count INT COMMENT '总乘机次数',
            seg_km_sum DECIMAL(14,2) COMMENT '总飞行公里数',
            weighted_seg_km DECIMAL(14,2) COMMENT '加权飞行公里数',
            avg_flight_count DECIMAL(10,2) COMMENT '平均乘机次数',
            last_flight_date VARCHAR(32) COMMENT '最后一次乘机日期',
            begin_to_first DECIMAL(10,2) COMMENT '入会到首次乘机间隔',
            last_to_end DECIMAL(10,2) COMMENT '最后乘机距数据截止日间隔',
            avg_interval DECIMAL(10,2) COMMENT '平均乘机间隔',
            max_interval INT COMMENT '最大乘机间隔',
            p1y_flight_count INT COMMENT '前一年乘机次数',
            l1y_flight_count INT COMMENT '最近一年乘机次数',
            ration_l1y_flight_count DECIMAL(10,6) COMMENT '最近一年乘机次数占比',
            ration_p1y_flight_count DECIMAL(10,6) COMMENT '前一年乘机次数占比',
            bp_sum DECIMAL(14,2) COMMENT '基本积分总和',
            avg_bp_sum DECIMAL(14,2) COMMENT '平均基本积分总和',
            ep_sum DECIMAL(14,2) COMMENT '精英积分总和',
            ep_sum_yr_1 DECIMAL(14,2) COMMENT '第一年精英积分总和',
            ep_sum_yr_2 DECIMAL(14,2) COMMENT '第二年精英积分总和',
            add_point_sum DECIMAL(14,2) COMMENT '额外积分总和',
            eli_add_point_sum DECIMAL(14,2) COMMENT '有效额外积分总和',
            l1y_eli_add_points DECIMAL(14,2) COMMENT '最近一年有效额外积分',
            add_points_sum_yr_1 DECIMAL(14,2) COMMENT '第一年额外积分总和',
            add_points_sum_yr_2 DECIMAL(14,2) COMMENT '第二年额外积分总和',
            points_sum DECIMAL(14,2) COMMENT '总积分',
            l1y_points_sum DECIMAL(14,2) COMMENT '最近一年总积分',
            l1y_bp_sum DECIMAL(14,2) COMMENT '最近一年基本积分总和',
            p1y_bp_sum DECIMAL(14,2) COMMENT '前一年基本积分总和',
            point_not_flight DECIMAL(14,2) COMMENT '非乘机积分',
            ration_p1y_bps DECIMAL(10,6) COMMENT '前一年基本积分占比',
            ration_l1y_bps DECIMAL(10,6) COMMENT '最近一年基本积分占比',
            sum_yr_1 DECIMAL(14,2) COMMENT '第一年票价总额',
            sum_yr_2 DECIMAL(14,2) COMMENT '第二年票价总额',
            exchange_count INT COMMENT '积分兑换次数',
            avg_discount DECIMAL(10,4) COMMENT '平均折扣率',
            import_date DATE NOT NULL COMMENT '数据导入日期',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
            UNIQUE KEY uk_member_data_batch (member_no, batch_no),
            INDEX idx_batch_no (batch_no),
            INDEX idx_import_date (import_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='原始客户数据表'
        """

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(rfm_analysis_sql)
            cursor.execute(customer_clusters_sql)
            cursor.execute(member_data_sql)
            return True
        except pymysql.Error as e:
            raise Exception(f"创建表失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    # ============================================================
    # 内部辅助：批量写入通用方法
    # ============================================================
    def _batch_insert_chunked(self, sql, rows, chunk_size=DEFAULT_CHUNK_SIZE,
                              progress_callback=None):
        """
        分块 + 显式事务执行批量插入。

        - 每chunk_size行调用一次 executemany
        - 每个chunk在显式事务中执行，结束后 COMMIT
        - 失败时 ROLLBACK 并抛出异常

        Parameters
        ----------
        sql : str
            带 %s 占位符的 INSERT ... ON DUPLICATE KEY UPDATE 语句
        rows : list[tuple]
            待插入的参数化数据行
        chunk_size : int
            单次事务的行数
        progress_callback : callable, optional
            进度回调 fn(done, total)，供 UI 显示进度

        Returns
        -------
        int
            成功插入的记录数
        """
        total = len(rows)
        if total == 0:
            return 0

        conn = self._pool.connection()
        cursor = conn.cursor()
        # 使用 begin()/commit()/rollback() 管理事务
        # （DBUtils SteadyDBConnection 不暴露 autocommit() 方法）
        # 连接池配置了 autocommit=True，begin() 挂起自动提交开启事务，
        # commit() 后连接自动恢复为 autocommit 模式。
        try:
            conn.begin()

            # 批量导入优化：临时关闭唯一性检查与外键检查
            # （可显著降低 InnoDB 二级索引维护成本）
            cursor.execute("SET unique_checks=0")
            cursor.execute("SET foreign_key_checks=0")

            done = 0
            for start in range(0, total, chunk_size):
                chunk = rows[start:start + chunk_size]
                cursor.executemany(sql, chunk)
                conn.commit()
                # commit() 结束了当前事务，需重新 begin() 进入下一个 chunk 的事务
                if start + chunk_size < total:
                    conn.begin()
                done += len(chunk)
                if progress_callback is not None:
                    try:
                        progress_callback(done, total)
                    except Exception:
                        pass

            # 恢复会话级检查
            cursor.execute("SET unique_checks=1")
            cursor.execute("SET foreign_key_checks=1")
            return done
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            raise Exception(f"批量插入失败（已回滚）: {e}")
        finally:
            try:
                # 恢复会话级检查（无论成功或失败都执行）
                cursor.execute("SET unique_checks=1")
                cursor.execute("SET foreign_key_checks=1")
            except Exception:
                pass
            cursor.close()
            conn.close()  # 归还到连接池

    def insert_rfm_analysis(self, member_no, batch_no, r_value, f_value, m_value, analysis_date=None):
        """
        插入RFM分析结果。

        Parameters
        ----------
        member_no : str
            会员编号
        batch_no : str
            分析批次编号
        r_value : float
            R值
        f_value : float
            F值
        m_value : float
            M值
        analysis_date : str or date, optional
            分析日期，默认为今天

        Returns
        -------
        int
            插入记录的rfm_id
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        if analysis_date is None:
            analysis_date = datetime.date.today()

        sql = """
        INSERT INTO rfm_analysis (member_no, batch_no, r_value, f_value, m_value, analysis_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            r_value = VALUES(r_value),
            f_value = VALUES(f_value),
            m_value = VALUES(m_value),
            analysis_date = VALUES(analysis_date)
        """

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (member_no, batch_no, r_value, f_value, m_value, analysis_date))
            return cursor.lastrowid
        except pymysql.Error as e:
            raise Exception(f"插入RFM分析数据失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    def insert_customer_cluster(self, member_no, batch_no, cluster_label, customer_type,
                                cluster_date=None):
        """
        插入客户分群结果。

        Parameters
        ----------
        member_no : str
            会员编号
        batch_no : str
            聚类批次编号
        cluster_label : int
            聚类标签编号
        customer_type : str
            客户类型
        cluster_date : str or date, optional
            聚类日期，默认为今天

        Returns
        -------
        int
            插入记录的cluster_id
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        if cluster_date is None:
            cluster_date = datetime.date.today()

        sql = """
        INSERT INTO customer_clusters 
        (member_no, batch_no, cluster_label, customer_type, cluster_date)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            cluster_label = VALUES(cluster_label),
            customer_type = VALUES(customer_type),
            cluster_date = VALUES(cluster_date)
        """

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (member_no, batch_no, cluster_label, customer_type,
                                 cluster_date))
            return cursor.lastrowid
        except pymysql.Error as e:
            raise Exception(f"插入客户分群数据失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    def batch_insert_rfm_analysis(self, df, batch_no, analysis_date=None,
                                  chunk_size=DEFAULT_CHUNK_SIZE,
                                  progress_callback=None):
        """
        批量插入RFM分析结果。

        优化点：
        - 使用 itertuples 替代 iterrows()（速度提升 10-20 倍）
        - 分块事务提交，避免单一大事务
        - 临时关闭 unique_checks/foreign_key_checks

        Parameters
        ----------
        df : pandas.DataFrame
            包含 MEMBER_NO, R, F, M 列的数据
        batch_no : str
            分析批次编号
        analysis_date : str or date, optional
            分析日期，默认为今天
        chunk_size : int
            单次事务行数，默认 5000
        progress_callback : callable, optional
            进度回调 fn(done, total)

        Returns
        -------
        int
            成功插入的记录数
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        if analysis_date is None:
            analysis_date = datetime.date.today()

        sql = """
        INSERT INTO rfm_analysis (member_no, batch_no, r_value, f_value, m_value, analysis_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            r_value = VALUES(r_value),
            f_value = VALUES(f_value),
            m_value = VALUES(m_value),
            analysis_date = VALUES(analysis_date)
        """

        # 用 itertuples 高效遍历（不创建 Series 对象）
        # namedtuple 访问比 row['col'] 快 5-10 倍
        rows = []
        for t in df.itertuples(index=False):
            member_no_val = t.MEMBER_NO
            # 安全转换：处理 float 类型（如 54993.0 -> "54993"）
            if isinstance(member_no_val, float):
                member_no = str(int(member_no_val))
            else:
                member_no = str(member_no_val)
            rows.append((
                member_no,
                batch_no,
                float(t.R),
                float(t.F),
                float(t.M),
                analysis_date,
            ))

        return self._batch_insert_chunked(sql, rows, chunk_size=chunk_size,
                                          progress_callback=progress_callback)

    def batch_insert_customer_clusters(self, df, batch_no, cluster_date=None,
                                       chunk_size=DEFAULT_CHUNK_SIZE,
                                       progress_callback=None):
        """
        批量插入客户分群结果。

        Parameters
        ----------
        df : pandas.DataFrame
            包含 MEMBER_NO, Cluster, 客户分群 列的数据
        batch_no : str
            聚类批次编号
        cluster_date : str or date, optional
            聚类日期，默认为今天
        chunk_size : int
            单次事务行数
        progress_callback : callable, optional
            进度回调 fn(done, total)

        Returns
        -------
        int
            成功插入的记录数
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        if cluster_date is None:
            cluster_date = datetime.date.today()

        sql = """
        INSERT INTO customer_clusters (member_no, batch_no, cluster_label, customer_type, cluster_date)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            cluster_label = VALUES(cluster_label),
            customer_type = VALUES(customer_type),
            cluster_date = VALUES(cluster_date)
        """

        # 注意：itertuples 会把非法标识符列名（如中文 "客户分群"）重命名为 _2/_3 等。
        # 因此这里用 df.values 直接按列索引取值，更稳定高效。
        # 列顺序: MEMBER_NO(0), Cluster(1), 客户分群(2) — 调用方传入的 df 即 cluster_result
        member_col = df['MEMBER_NO'].values
        cluster_col = df['Cluster'].values
        type_col = df['客户分群'].values if '客户分群' in df.columns else [''] * len(df)

        rows = []
        for i in range(len(df)):
            member_no_val = member_col[i]
            if isinstance(member_no_val, float):
                if pd.isna(member_no_val):
                    continue
                member_no = str(int(member_no_val))
            elif member_no_val is None:
                continue
            else:
                member_no = str(member_no_val)
            cluster_val = cluster_col[i]
            if hasattr(cluster_val, 'item'):
                cluster_val = cluster_val.item()
            type_val = type_col[i]
            if type_val is None or (isinstance(type_val, float) and pd.isna(type_val)):
                type_val = ''
            else:
                type_val = str(type_val)
            rows.append((
                member_no,
                batch_no,
                int(cluster_val),
                type_val,
                cluster_date,
            ))

        return self._batch_insert_chunked(sql, rows, chunk_size=chunk_size,
                                          progress_callback=progress_callback)

    def batch_insert_member_data(self, df, batch_no, import_date=None,
                                 chunk_size=DEFAULT_CHUNK_SIZE,
                                 progress_callback=None):
        """
        批量插入原始客户数据。

        优化点：
        - 使用 numpy 向量化转换替代逐行 Python 循环（速度提升 50-100 倍）
        - 分块事务提交（每 chunk_size 行 commit 一次）
        - 临时关闭 unique_checks/foreign_key_checks
        - NaN/None 统一处理为 NULL
        - 进度回调覆盖"行构建 + 数据库写入"全过程

        Parameters
        ----------
        df : pandas.DataFrame
            包含完整清洗后客户数据的 DataFrame
        batch_no : str
            数据批次编号
        import_date : str or date, optional
            导入日期，默认为今天
        chunk_size : int
            单次事务行数，默认 5000
        progress_callback : callable, optional
            进度回调 fn(done, total)

        Returns
        -------
        int
            成功插入的记录数
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        if import_date is None:
            import_date = datetime.date.today()

        # DataFrame 列名到数据库列名的映射
        column_mapping = {
            'MEMBER_NO': 'member_no',
            'FFP_DATE': 'ffp_date',
            'FIRST_FLIGHT_DATE': 'first_flight_date',
            'GENDER': 'gender',
            'FFP_TIER': 'ffp_tier',
            'WORK_CITY': 'work_city',
            'WORK_PROVINCE': 'work_province',
            'WORK_COUNTRY': 'work_country',
            'AGE': 'age',
            'LOAD_TIME': 'load_time',
            'FLIGHT_COUNT': 'flight_count',
            'BP_SUM': 'bp_sum',
            'EP_SUM_YR_1': 'ep_sum_yr_1',
            'EP_SUM_YR_2': 'ep_sum_yr_2',
            'SUM_YR_1': 'sum_yr_1',
            'SUM_YR_2': 'sum_yr_2',
            'SEG_KM_SUM': 'seg_km_sum',
            'WEIGHTED_SEG_KM': 'weighted_seg_km',
            'LAST_FLIGHT_DATE': 'last_flight_date',
            'AVG_FLIGHT_COUNT': 'avg_flight_count',
            'AVG_BP_SUM': 'avg_bp_sum',
            'BEGIN_TO_FIRST': 'begin_to_first',
            'LAST_TO_END': 'last_to_end',
            'AVG_INTERVAL': 'avg_interval',
            'MAX_INTERVAL': 'max_interval',
            'ADD_POINTS_SUM_YR_1': 'add_points_sum_yr_1',
            'ADD_POINTS_SUM_YR_2': 'add_points_sum_yr_2',
            'EXCHANGE_COUNT': 'exchange_count',
            'avg_discount': 'avg_discount',
            'P1Y_Flight_Count': 'p1y_flight_count',
            'L1Y_Flight_Count': 'l1y_flight_count',
            'P1Y_BP_SUM': 'p1y_bp_sum',
            'L1Y_BP_SUM': 'l1y_bp_sum',
            'EP_SUM': 'ep_sum',
            'ADD_Point_SUM': 'add_point_sum',
            'Eli_Add_Point_Sum': 'eli_add_point_sum',
            'L1Y_ELi_Add_Points': 'l1y_eli_add_points',
            'Points_Sum': 'points_sum',
            'L1Y_Points_Sum': 'l1y_points_sum',
            'Ration_L1Y_Flight_Count': 'ration_l1y_flight_count',
            'Ration_P1Y_Flight_Count': 'ration_p1y_flight_count',
            'Ration_P1Y_BPS': 'ration_p1y_bps',
            'Ration_L1Y_BPS': 'ration_l1y_bps',
            'Point_NotFlight': 'point_not_flight',
        }

        db_columns = list(column_mapping.values())
        # 列顺序：member_no, batch_no, 其余列..., import_date
        column_list = 'member_no, batch_no, ' + ', '.join(db_columns[1:]) + ', import_date'
        placeholders = ', '.join(['%s'] * (len(db_columns) + 2))  # +2 for batch_no and import_date
        update_list = ', '.join([f"{col} = VALUES({col})" for col in db_columns[1:]])

        sql = f"""
        INSERT INTO member_data ({column_list})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE
            {update_list},
            import_date = VALUES(import_date)
        """

        # 预先确定 df 中实际存在的列
        present_csv_cols = [c for c in column_mapping.keys() if c in df.columns]
        total = len(df)

        # 过滤掉 MEMBER_NO 为 NaN 的行（向量化操作，极快）
        member_series = df['MEMBER_NO'] if 'MEMBER_NO' in df.columns else None
        if member_series is not None:
            valid_mask = member_series.notna()
            if not valid_mask.all():
                df = df[valid_mask].reset_index(drop=True)
                total = len(df)
                member_series = df['MEMBER_NO']

        # 向量化构建 member_no 字符串列
        # 原始 MEMBER_NO 可能是 float (54993.0)、int、或字符串
        member_arr = member_series.values
        if member_arr.dtype.kind == 'f':  # float
            # 转为 int 再转 str，避免 "54993.0"
            member_str_arr = member_arr.astype('int64').astype(object)
            # numpy.str_ 会被 pymysql 正确处理
            member_str_arr = np.array([str(x) for x in member_str_arr], dtype=object)
        elif member_arr.dtype.kind in ('i', 'u'):  # int/uint
            member_str_arr = member_arr.astype(object)
            member_str_arr = np.array([str(x) for x in member_str_arr], dtype=object)
        else:
            # object dtype：逐元素 str()，但用列表推导比 Python 循环快
            member_str_arr = np.array([str(x) for x in member_arr], dtype=object)

        # 为每列预先向量化处理 NaN -> None
        # 构建 (total, n_cols) 的 object 数组，再逐行打包成 tuple
        batch_no_arr = np.array([batch_no] * total, dtype=object)
        import_date_arr = np.array([import_date] * total, dtype=object)

        # 收集除 member_no/batch_no/import_date 外的列数据
        other_db_cols = [c for c in column_mapping.values() if c != 'member_no']
        col_data_arrays = []  # 每个 element 是长度为 total 的 object array
        for csv_col, db_col in column_mapping.items():
            if db_col == 'member_no':
                continue
            if csv_col in df.columns:
                col_vals = df[csv_col].values
                if col_vals.dtype.kind in ('f', 'i', 'u'):
                    # 数值列：用 where 将 NaN 替换为 None
                    nan_mask = pd.isna(col_vals)
                    if nan_mask.any():
                        obj_arr = col_vals.astype(object)
                        obj_arr[nan_mask] = None
                    else:
                        # 保留 numpy 数值，pymysql 会自动处理
                        obj_arr = col_vals.astype(object)
                    col_data_arrays.append(obj_arr)
                elif col_vals.dtype.kind == 'O':
                    # object 列（字符串、Timestamp 混合）
                    nan_mask = pd.isna(col_vals)
                    if nan_mask.any():
                        obj_arr = col_vals.copy()
                        obj_arr[nan_mask] = None
                    else:
                        obj_arr = col_vals
                    # Timestamp 转 str
                    # 检查第一个非 None 元素是否是 Timestamp
                    has_ts = False
                    for v in obj_arr:
                        if v is not None and hasattr(v, 'isoformat'):
                            has_ts = True
                            break
                    if has_ts:
                        obj_arr = np.array(
                            [v.strftime('%Y-%m-%d') if v is not None and hasattr(v, 'isoformat')
                             else (v if v is not None else None)
                             for v in obj_arr],
                            dtype=object
                        )
                    col_data_arrays.append(obj_arr)
                else:
                    # datetime64 等其他类型
                    col_data_arrays.append(df[csv_col].astype(object).values)
            else:
                # 列不存在，填充 None
                col_data_arrays.append(np.array([None] * total, dtype=object))

        # 拼接所有列：member_no, batch_no, [other_cols...], import_date
        all_col_arrays = [member_str_arr, batch_no_arr] + col_data_arrays + [import_date_arr]

        # 分块构建 rows 并写入数据库，让进度回调覆盖构建阶段
        done = 0
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            # 用 zip 打包当前 chunk 的所有列，形成 list[tuple]
            chunk_cols = [arr[start:end] for arr in all_col_arrays]
            # zip(*chunk_cols) 转置为按行迭代
            # tuple(r) 将每行转为 tuple
            chunk_rows = [tuple(r) for r in zip(*chunk_cols)]
            # 写入数据库
            self._batch_insert_chunked(sql, chunk_rows, chunk_size=len(chunk_rows),
                                       progress_callback=None)
            done += (end - start)
            if progress_callback is not None:
                try:
                    progress_callback(done, total)
                except Exception:
                    pass

        return done

    # ============================================================
    # 流式查询：避免一次性载入百万级数据
    # ============================================================
    def stream_query(self, sql, params=None, batch_size=DEFAULT_FETCH_BATCH):
        """
        流式查询生成器：使用服务端游标，每次只取 batch_size 行，避免内存峰值。

        适用于导出 / 大批量分析场景。调用方应当用 for row in stream_query(...): 消费。

        Parameters
        ----------
        sql : str
            SELECT SQL
        params : tuple, optional
            参数
        batch_size : int
            每次 fetch 的行数

        Yields
        ------
        list[dict]
            每批 batch_size 条记录（dict 形式）
        """
        if not self.ensure_connection():
            raise Exception("数据库未连接")

        conn = self._pool.connection()
        # 使用 SSCursor（服务端游标），结果集保留在 MySQL 端
        # 必须新建一个使用 SSCursor 的连接，因为连接池的默认 cursorclass 是 DictCursor
        cursor = None
        try:
            # 直接用 conn.cursor(SSCursor) 切换游标类型
            cursor = conn.cursor(SSCursor)
            if params is not None:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                yield batch
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_member_data(self, batch_no=None, member_no=None, limit=100):
        """
        查询原始客户数据。

        Parameters
        ----------
        batch_no : str, optional
            批次编号，为None时查询所有
        member_no : str, optional
            会员编号，为None时查询所有
        limit : int
            返回记录数限制

        Returns
        -------
        pandas.DataFrame
            查询结果
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        conditions = []
        params = []

        if batch_no:
            conditions.append("batch_no = %s")
            params.append(batch_no)
        if member_no:
            conditions.append("member_no = %s")
            params.append(member_no.strip())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM member_data WHERE {where_clause} ORDER BY data_id DESC LIMIT %s"
        params.append(limit)

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(params))
            results = cursor.fetchall()
            return pd.DataFrame(results)
        except pymysql.Error as e:
            raise Exception(f"查询原始客户数据失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    def get_rfm_analysis(self, batch_no=None, member_no=None, limit=100):
        """
        查询RFM分析结果。

        Parameters
        ----------
        batch_no : str, optional
            批次编号，为None时查询所有
        member_no : str, optional
            会员编号，为None时查询所有
        limit : int
            返回记录数限制

        Returns
        -------
        pandas.DataFrame
            查询结果
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        conditions = []
        params = []

        if batch_no:
            conditions.append("batch_no = %s")
            params.append(batch_no)
        if member_no:
            conditions.append("member_no = %s")
            params.append(member_no.strip())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM rfm_analysis WHERE {where_clause} ORDER BY rfm_id DESC LIMIT %s"
        params.append(limit)

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(params))
            results = cursor.fetchall()
            return pd.DataFrame(results)
        except pymysql.Error as e:
            raise Exception(f"查询RFM分析数据失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    def get_customer_clusters(self, batch_no=None, customer_type=None, member_no=None, limit=100):
        """
        查询客户分群结果。

        Parameters
        ----------
        batch_no : str, optional
            批次编号，为None时查询所有
        customer_type : str, optional
            客户类型，为None时查询所有
        member_no : str, optional
            会员编号，为None时查询所有
        limit : int
            返回记录数限制

        Returns
        -------
        pandas.DataFrame
            查询结果
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        conditions = []
        params = []

        if batch_no:
            conditions.append("batch_no = %s")
            params.append(batch_no)
        if customer_type:
            conditions.append("customer_type = %s")
            params.append(customer_type)
        if member_no:
            conditions.append("member_no = %s")
            params.append(member_no.strip())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM customer_clusters WHERE {where_clause} ORDER BY cluster_id DESC LIMIT %s"
        params.append(limit)

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(params))
            results = cursor.fetchall()
            return pd.DataFrame(results)
        except pymysql.Error as e:
            raise Exception(f"查询客户分群数据失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    def get_batch_list(self, table_name):
        """
        获取指定表的批次列表。

        Parameters
        ----------
        table_name : str
            表名：rfm_analysis 或 customer_clusters

        Returns
        -------
        list
            批次编号列表
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        if table_name not in ['rfm_analysis', 'customer_clusters', 'member_data']:
            raise ValueError("表名必须是 rfm_analysis、customer_clusters 或 member_data")

        sql = f"SELECT DISTINCT batch_no FROM {table_name} ORDER BY batch_no DESC"

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            return [row['batch_no'] for row in results]
        except pymysql.Error as e:
            raise Exception(f"获取批次列表失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    def get_table_stats(self):
        """
        获取各表的记录数统计。

        Returns
        -------
        dict
            各表的记录数
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        stats = {}
        tables = ['rfm_analysis', 'customer_clusters', 'member_data']

        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = cursor.fetchone()
                stats[table] = result['count'] if result else 0

            # 统计批次数量（三表去重）
            cursor.execute(
                "SELECT COUNT(DISTINCT batch_no) as count FROM ("
                "  SELECT batch_no FROM rfm_analysis"
                "  UNION"
                "  SELECT batch_no FROM customer_clusters"
                "  UNION"
                "  SELECT batch_no FROM member_data"
                ") AS all_batches"
            )
            result = cursor.fetchone()
            stats['batch_count'] = result['count'] if result else 0
            return stats
        except pymysql.Error as e:
            raise Exception(f"获取表统计信息失败: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    # ============================================================
    # 高性能：聚合查询接口（避免全量加载到内存）
    # ============================================================
    def get_customer_type_counts(self, batch_no=None, customer_type=None, member_no=None):
        """
        直接在数据库层做 GROUP BY 聚合，返回各 customer_type 的计数。

        专门用于替代 app.py 中 get_customer_clusters(limit=1000000).value_counts() 的反模式。

        Returns
        -------
        pandas.Series
            以 customer_type 为 index，count 为 value
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        conditions = []
        params = []
        if batch_no:
            conditions.append("batch_no = %s")
            params.append(batch_no)
        if customer_type:
            conditions.append("customer_type = %s")
            params.append(customer_type)
        if member_no:
            conditions.append("member_no = %s")
            params.append(member_no.strip())
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
        SELECT customer_type, COUNT(*) AS cnt
        FROM customer_clusters
        WHERE {where_clause}
        GROUP BY customer_type
        ORDER BY cnt DESC
        """
        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(params))
            results = cursor.fetchall()
            if not results:
                return pd.Series([], dtype='int64', name='count')
            return pd.Series(
                {r['customer_type']: r['cnt'] for r in results},
                name='count',
            )
        except pymysql.Error as e:
            raise Exception(f"查询分群计数失败: {e}")
        finally:
            cursor.close()
            conn.close()

    def count_batch_rows(self, table_name, batch_no):
        """
        统计指定表指定批次的行数（避免 SELECT *）。

        Returns
        -------
        int
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")
        if table_name not in ['rfm_analysis', 'customer_clusters', 'member_data']:
            raise ValueError("表名必须是 rfm_analysis、customer_clusters 或 member_data")
        sql = f"SELECT COUNT(*) AS cnt FROM {table_name} WHERE batch_no = %s"
        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (batch_no,))
            r = cursor.fetchone()
            return int(r['cnt']) if r else 0
        except pymysql.Error as e:
            raise Exception(f"统计行数失败: {e}")
        finally:
            cursor.close()
            conn.close()

    # ============================================================
    # 流式导出：分批 fetch 写入 CSV，避免内存峰值
    # ============================================================
    def stream_export_to_csv(self, table_name, batch_no, output_path,
                             batch_size=DEFAULT_FETCH_BATCH,
                             progress_callback=None):
        """
        流式导出指定批次的全部数据到 CSV 文件，避免一次性载入内存。

        Parameters
        ----------
        table_name : str
            rfm_analysis / customer_clusters / member_data
        batch_no : str
            批次编号
        output_path : str
            输出 CSV 路径
        batch_size : int
            每次 fetch 行数
        progress_callback : callable, optional
            进度回调 fn(done, total)

        Returns
        -------
        int
            导出总行数
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")
        if table_name not in ['rfm_analysis', 'customer_clusters', 'member_data']:
            raise ValueError("表名必须是 rfm_analysis、customer_clusters 或 member_data")

        # 先获取总行数
        total = self.count_batch_rows(table_name, batch_no)
        if total == 0:
            # 写入空 CSV
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write('')
            return 0

        sql = f"SELECT * FROM {table_name} WHERE batch_no = %s"
        done = 0

        # 使用服务端游标流式拉取
        conn = self._pool.connection()
        cursor = None
        try:
            cursor = conn.cursor(SSCursor)
            cursor.execute(sql, (batch_no,))
            # 取第一批以获取列名
            first_batch = cursor.fetchmany(batch_size)
            if not first_batch:
                with open(output_path, 'w', encoding='utf-8-sig') as f:
                    f.write('')
                return 0
            # 从 description 获取列名
            col_names = [d[0] for d in cursor.description]
            # 写入第一批
            df = pd.DataFrame(first_batch, columns=col_names)
            df.to_csv(output_path, index=False, encoding='utf-8-sig', mode='w')
            done += len(first_batch)
            if progress_callback is not None:
                try:
                    progress_callback(done, total)
                except Exception:
                    pass

            # 后续批次追加
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                df = pd.DataFrame(batch, columns=col_names)
                df.to_csv(output_path, index=False, encoding='utf-8-sig',
                          mode='a', header=False)
                done += len(batch)
                if progress_callback is not None:
                    try:
                        progress_callback(done, total)
                    except Exception:
                        pass
                # 显式释放
                del df
                del batch
            return done
        except pymysql.Error as e:
            raise Exception(f"流式导出失败: {e}")
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass

    # ============================================================
    # 分批删除：避免长事务
    # ============================================================
    def delete_batch(self, table_name, batch_no, batch_size=DEFAULT_DELETE_BATCH,
                     progress_callback=None):
        """
        分批删除指定批次的数据。

        优化点：
        - 每次只删除 batch_size 行，避免长事务持有大量行锁
        - 每批立即 COMMIT，释放锁与 undo log
        - 通过循环直至该批次完全清空

        Parameters
        ----------
        table_name : str
            表名：rfm_analysis / customer_clusters / member_data
        batch_no : str
            批次编号
        batch_size : int
            单次 DELETE 行数，默认 5000
        progress_callback : callable, optional
            进度回调 fn(done, total)

        Returns
        -------
        int
            删除的记录数
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        if table_name not in ['rfm_analysis', 'customer_clusters', 'member_data']:
            raise ValueError("表名必须是 rfm_analysis、customer_clusters 或 member_data")

        # 先获取总数（用于进度回调）
        total = self.count_batch_rows(table_name, batch_no) if progress_callback else 0

        conn = self._pool.connection()
        cursor = conn.cursor()
        # 使用 begin()/commit() 管理事务（SteadyDBConnection 不支持 autocommit() 方法）
        total_deleted = 0
        try:
            # 使用主键范围删除：先找到该批次的 PK 区间，逐区间删除
            # 比直接 DELETE FROM ... WHERE batch_no=? LIMIT n 更快（避免全表扫描）
            pk_col = {
                'rfm_analysis': 'rfm_id',
                'customer_clusters': 'cluster_id',
                'member_data': 'data_id',
            }[table_name]

            while True:
                # 找到下一批要删除的 PK 上界
                cursor.execute(
                    f"SELECT {pk_col} FROM {table_name} "
                    f"WHERE batch_no = %s ORDER BY {pk_col} LIMIT %s",
                    (batch_no, batch_size)
                )
                pk_rows = cursor.fetchall()
                if not pk_rows:
                    break
                pk_list = [r[pk_col] for r in pk_rows]
                # 用 IN 列表删除（命中主键索引，速度极快）
                placeholders = ','.join(['%s'] * len(pk_list))
                delete_sql = (
                    f"DELETE FROM {table_name} "
                    f"WHERE {pk_col} IN ({placeholders})"
                )
                # 每批删除在独立事务中执行
                conn.begin()
                cursor.execute(delete_sql, pk_list)
                deleted = cursor.rowcount
                conn.commit()
                total_deleted += deleted
                if progress_callback is not None:
                    try:
                        progress_callback(total_deleted, total)
                    except Exception:
                        pass
                if deleted == 0:
                    break
            return total_deleted
        except pymysql.Error as e:
            try:
                conn.rollback()
            except Exception:
                pass
            raise Exception(f"删除批次数据失败（已回滚）: {e}")
        finally:
            cursor.close()
            conn.close()  # 归还到连接池

    def truncate_all_tables(self):
        """
        使用 TRUNCATE 清空所有业务表数据。

        性能对比（5M 行）：
        - DELETE FROM: 需逐行记录 undo log，耗时数十秒到数分钟
        - TRUNCATE: 直接释放数据页，几乎瞬间完成（<1秒）

        注意事项：
        - TRUNCATE 是 DDL 操作，无法回滚
        - TRUNCATE 会重置 AUTO_INCREMENT 计数器
        - TRUNCATE 不触发触发器

        Returns
        -------
        dict
            各表是否成功清空 {'rfm_analysis': True, 'customer_clusters': True, 'member_data': True}
        """
        if not self.is_connected():
            raise Exception("请先连接数据库")

        tables = ['rfm_analysis', 'customer_clusters', 'member_data']
        result = {}
        conn = self._pool.connection()
        cursor = conn.cursor()
        try:
            # 临时关闭外键检查，避免外键约束阻止 TRUNCATE
            cursor.execute("SET foreign_key_checks=0")
            for table in tables:
                try:
                    cursor.execute(f"TRUNCATE TABLE {table}")
                    result[table] = True
                except pymysql.Error as e:
                    result[table] = False
            cursor.execute("SET foreign_key_checks=1")
            return result
        except pymysql.Error as e:
            raise Exception(f"清空表数据失败: {e}")
        finally:
            try:
                cursor.execute("SET foreign_key_checks=1")
            except Exception:
                pass
            cursor.close()
            conn.close()  # 归还到连接池
