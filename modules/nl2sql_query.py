"""
大模型辅助查询模块
实现自然语言转SQL查询功能，基于MySQL数据库进行查询。
支持 LLM 模式和规则降级模式。
"""

import re
import pandas as pd
from openai import OpenAI

from config.settings import MYSQL_VERSION, DEFAULT_LLM_CONFIG


class QueryHistory:
    """查询历史管理类"""

    def __init__(self):
        """初始化查询历史"""
        self.history = []

    def add(self, question, sql, result_preview, status, error_message=''):
        """
        添加历史记录

        Args:
            question: 用户问题
            sql: 生成的SQL语句
            result_preview: 查询结果预览
            status: 查询状态（'success' 或 'error'）
            error_message: 错误信息（默认为空）
        """
        self.history.append({
            'question': question,
            'sql': sql,
            'result_preview': result_preview,
            'status': status,
            'error_message': error_message
        })

    def get_all(self):
        """获取全部历史"""
        return self.history

    def get_recent(self, n=10):
        """获取最近N条历史记录"""
        return self.history[-n:]

    def clear(self):
        """清空历史"""
        self.history = []


class NL2SQLQueryEngine:
    """自然语言转SQL查询引擎，基于MySQL数据库进行查询"""

    # ================================================================
    # MySQL 数据库表结构说明
    # ================================================================
    DB_TABLE_SCHEMA = f"""数据库: {MYSQL_VERSION} (airline_analysis)，InnoDB，utf8mb4。

【数据模型】
三表按 batch_no 批次存储，同一批次下三表的 member_no 一一对应：
  • member_data        — 原始客户数据（人口统计/乘机/积分/消费）
  • rfm_analysis       — RFM 分析结果（R/F/M 三值）
  • customer_clusters  — K-Means 聚类分群结果（标签 + 客户类型）

三表 batch_no 共享同一格式（CLUSTER_ 前缀，如 'CLUSTER_20260701_120000'）。
多表 JOIN 必须同时关联 member_no 和 batch_no：
  JOIN ... ON a.member_no = b.member_no AND a.batch_no = b.batch_no

【表1: rfm_analysis】RFM 分析结果表
列名            类型           说明
rfm_id         BIGINT         主键
member_no      VARCHAR(32)    会员编号（联合唯一：member_no+batch_no）
batch_no       VARCHAR(32)    批次号（INDEX）
r_value        DECIMAL(12,2)  R值：最近一次消费距今天数
f_value        DECIMAL(12,2)  F值：消费频率（乘机次数）
m_value        DECIMAL(12,2)  M值：消费金额
analysis_date  DATE           分析执行日期（DATE 类型，INDEX）
created_at     DATETIME       创建时间

【表2: customer_clusters】客户分群表
列名            类型           说明
cluster_id     BIGINT         主键
member_no      VARCHAR(32)    会员编号（联合唯一：member_no+batch_no）
batch_no       VARCHAR(32)    批次号（INDEX）
cluster_label  INT            聚类标签（整数 0,1,2,3,4...，INDEX）
customer_type  VARCHAR(32)    客户类型名称（INDEX），取值固定为以下 11 种之一：
    '休眠客户','流失客户','低价值客户','新客户','一般客户','一般保持客户',
    '一般发展客户','潜力客户','重要发展客户','重要保持客户','高价值客户'
cluster_date   DATE           聚类执行日期
created_at     DATETIME       创建时间

【表3: member_data】原始客户数据表（联合唯一：member_no+batch_no）
列名                      类型           说明
data_id                  BIGINT         主键
member_no                VARCHAR(32)    会员编号
batch_no                 VARCHAR(32)    批次号（INDEX）
ffp_date                 VARCHAR(32)    入会日期（字符串，非 DATE）
first_flight_date        VARCHAR(32)    首次乘机日期（字符串）
gender                   VARCHAR(8)     性别（'男'/'女'）
ffp_tier                 INT            常旅客等级
work_city                VARCHAR(64)    工作城市
work_province            VARCHAR(64)    工作省份
work_country             VARCHAR(32)    工作国家
age                      INT            年龄（可 NULL）
load_time                VARCHAR(32)    数据加载时间
flight_count             INT            总乘机次数
seg_km_sum               DECIMAL(14,2)  总飞行公里数
weighted_seg_km          DECIMAL(14,2)  加权飞行公里数
avg_flight_count         DECIMAL(10,2)  年均乘机次数
last_flight_date         VARCHAR(32)    末次乘机日期（字符串）
begin_to_first           DECIMAL(10,2)  入会到首乘间隔（天）
last_to_end              DECIMAL(10,2)  末次乘机距截止日（天）
avg_interval             DECIMAL(10,2)  平均乘机间隔（天）
max_interval             INT            最大乘机间隔（天）
p1y_flight_count         INT            前一年乘机次数（P1Y）
l1y_flight_count         INT            最近一年乘机次数（L1Y）
ration_l1y_flight_count  DECIMAL(10,6)  L1Y 乘机次数占比（0~1）
ration_p1y_flight_count  DECIMAL(10,6)  P1Y 乘机次数占比（0~1）
bp_sum                   DECIMAL(14,2)  基本积分总和
avg_bp_sum               DECIMAL(14,2)  年均基本积分
ep_sum                   DECIMAL(14,2)  精英积分总和
ep_sum_yr_1              DECIMAL(14,2)  第一年精英积分
ep_sum_yr_2              DECIMAL(14,2)  第二年精英积分
add_point_sum            DECIMAL(14,2)  额外积分总和
eli_add_point_sum        DECIMAL(14,2)  有效额外积分
l1y_eli_add_points       DECIMAL(14,2)  L1Y 有效额外积分
add_points_sum_yr_1      DECIMAL(14,2)  第一年额外积分
add_points_sum_yr_2      DECIMAL(14,2)  第二年额外积分
points_sum               DECIMAL(14,2)  总积分
l1y_points_sum           DECIMAL(14,2)  L1Y 总积分
l1y_bp_sum               DECIMAL(14,2)  L1Y 基本积分
p1y_bp_sum               DECIMAL(14,2)  P1Y 基本积分
point_not_flight         DECIMAL(14,2)  非乘机积分
ration_p1y_bps           DECIMAL(10,6)  P1Y 基本积分占比（0~1）
ration_l1y_bps           DECIMAL(10,6)  L1Y 基本积分占比（0~1）
sum_yr_1                 DECIMAL(14,2)  第一年票价总额
sum_yr_2                 DECIMAL(14,2)  第二年票价总额
exchange_count           INT            积分兑换次数
avg_discount             DECIMAL(10,4)  平均折扣率（0~1）
import_date              DATE           数据导入日期（NOT NULL，INDEX）
created_at               DATETIME       创建时间

【关键规则】
1. 仅允许 SELECT；禁止 DELETE/DROP/UPDATE/INSERT/ALTER/CREATE/TRUNCATE
2. VARCHAR 日期字段（ffp_date 等）用字符串比较：WHERE ffp_date >= '2015-01-01'
3. DATE 类型字段（analysis_date/cluster_date/import_date）用日期函数
4. DECIMAL 可直接聚合比较，无需 CAST；可空字段聚合用 COALESCE(col, 0)
5. customer_type 必须用单引号精确匹配：WHERE customer_type = '高价值客户'
6. MySQL 开启 ONLY_FULL_GROUP_BY：SELECT 含聚合函数时，所有非聚合列必须在 GROUP BY 中
7. 不要查询 INFORMATION_SCHEMA，不要使用存储过程
"""


    DANGEROUS_KEYWORDS = [
        'DELETE', 'DROP', 'UPDATE', 'INSERT', 'ALTER',
        'CREATE', 'TRUNCATE', 'REPLACE', 'GRANT', 'REVOKE',
        'EXEC', 'EXECUTE', 'ATTACH', 'DETACH'
    ]

    # ================================================================
    # MySQL 数据库模式规则匹配模板
    # ================================================================
    DB_RULE_PATTERNS = [
        # --- 客户分群相关查询 ---
        (r'各.*客户类型.*数量|各.*客户类型.*多少|各类客户.*多少|客户分群.*统计',
         "SELECT customer_type, COUNT(*) AS count FROM customer_clusters GROUP BY customer_type ORDER BY count DESC"),
        (r'高价值客户.*多少|高价值客户.*数量',
         "SELECT COUNT(*) AS count FROM customer_clusters WHERE customer_type = '高价值客户'"),
        (r'低价值客户.*多少|低价值客户.*数量',
         "SELECT COUNT(*) AS count FROM customer_clusters WHERE customer_type = '低价值客户'"),
        (r'流失客户.*多少|流失客户.*数量',
         "SELECT COUNT(*) AS count FROM customer_clusters WHERE customer_type = '流失客户'"),
        (r'新客户.*多少|新客户.*数量',
         "SELECT COUNT(*) AS count FROM customer_clusters WHERE customer_type = '新客户'"),
        (r'重要保持客户.*多少|重要保持客户.*数量',
         "SELECT COUNT(*) AS count FROM customer_clusters WHERE customer_type = '重要保持客户'"),
        (r'潜力客户.*多少|潜力客户.*数量',
         "SELECT COUNT(*) AS count FROM customer_clusters WHERE customer_type = '潜力客户'"),
        (r'重要发展客户.*多少|重要发展客户.*数量',
         "SELECT COUNT(*) AS count FROM customer_clusters WHERE customer_type = '重要发展客户'"),

        # --- RFM分析相关查询 ---
        (r'R值最小.*前\s*(\d+)|最近消费.*前\s*(\d+)',
         "SELECT * FROM rfm_analysis ORDER BY r_value ASC LIMIT {0}"),
        (r'R值最大.*前\s*(\d+)',
         "SELECT * FROM rfm_analysis ORDER BY r_value DESC LIMIT {0}"),
        (r'F值最大.*前\s*(\d+)|消费频率.*最高.*前\s*(\d+)|乘机次数.*最多.*前\s*(\d+)',
         "SELECT * FROM rfm_analysis ORDER BY f_value DESC LIMIT {0}"),
        (r'F值最小.*前\s*(\d+)|消费频率.*最低.*前\s*(\d+)',
         "SELECT * FROM rfm_analysis ORDER BY f_value ASC LIMIT {0}"),
        (r'M值最大.*前\s*(\d+)|消费金额.*最高.*前\s*(\d+)|飞行里程.*最多.*前\s*(\d+)',
         "SELECT * FROM rfm_analysis ORDER BY m_value DESC LIMIT {0}"),
        (r'M值最小.*前\s*(\d+)|消费金额.*最低.*前\s*(\d+)',
         "SELECT * FROM rfm_analysis ORDER BY m_value ASC LIMIT {0}"),
        (r'平均.*R值|平均.*最近消费',
         "SELECT AVG(r_value) AS avg_r FROM rfm_analysis"),
        (r'平均.*F值|平均.*消费频率|平均.*乘机次数',
         "SELECT AVG(f_value) AS avg_f FROM rfm_analysis"),
        (r'平均.*M值|平均.*消费金额|平均.*飞行里程',
         "SELECT AVG(m_value) AS avg_m FROM rfm_analysis"),
        (r'R.*F.*M.*统计|RFM.*统计|RFM.*摘要',
         "SELECT COUNT(*) AS total, AVG(r_value) AS avg_r, AVG(f_value) AS avg_f, AVG(m_value) AS avg_m, MIN(r_value) AS min_r, MAX(r_value) AS max_r, MIN(f_value) AS min_f, MAX(f_value) AS max_f, MIN(m_value) AS min_m, MAX(m_value) AS max_m FROM rfm_analysis"),

        # --- 跨表关联查询 ---
        (r'各客户类型.*平均.*R|各分群.*平均.*R',
         "SELECT c.customer_type, AVG(r.r_value) AS avg_r FROM customer_clusters c JOIN rfm_analysis r ON c.member_no = r.member_no GROUP BY c.customer_type ORDER BY avg_r"),
        (r'各客户类型.*平均.*F|各分群.*平均.*F',
         "SELECT c.customer_type, AVG(r.f_value) AS avg_f FROM customer_clusters c JOIN rfm_analysis r ON c.member_no = r.member_no GROUP BY c.customer_type ORDER BY avg_f DESC"),
        (r'各客户类型.*平均.*M|各分群.*平均.*M',
         "SELECT c.customer_type, AVG(r.m_value) AS avg_m FROM customer_clusters c JOIN rfm_analysis r ON c.member_no = r.member_no GROUP BY c.customer_type ORDER BY avg_m DESC"),
        (r'各客户类型.*RFM|各分群.*RFM',
         "SELECT c.customer_type, COUNT(*) AS count, AVG(r.r_value) AS avg_r, AVG(r.f_value) AS avg_f, AVG(r.m_value) AS avg_m FROM customer_clusters c JOIN rfm_analysis r ON c.member_no = r.member_no GROUP BY c.customer_type ORDER BY avg_m DESC"),
        (r'高价值客户.*详细|高价值客户.*RFM',
         "SELECT c.member_no, c.customer_type, r.r_value, r.f_value, r.m_value FROM customer_clusters c JOIN rfm_analysis r ON c.member_no = r.member_no WHERE c.customer_type = '高价值客户' ORDER BY r.m_value DESC"),
        (r'高价值客户.*平均.*F|高价值客户.*平均.*f',
         "SELECT AVG(r.f_value) AS avg_f FROM customer_clusters c JOIN rfm_analysis r ON c.member_no = r.member_no WHERE c.customer_type = '高价值客户'"),
        (r'流失客户.*平均.*R|流失客户.*平均.*r',
         "SELECT AVG(r.r_value) AS avg_r FROM customer_clusters c JOIN rfm_analysis r ON c.member_no = r.member_no WHERE c.customer_type = '流失客户'"),
        (r'各.*聚类标签.*数量|各.*聚类标签.*多少|各.*标签.*客户数|各.*标签.*多少',
         "SELECT cluster_label, COUNT(*) AS count FROM customer_clusters GROUP BY cluster_label ORDER BY cluster_label"),
        (r'查询.*聚类标签.*?(\d+)|标签\s*(\d+).*客户',
         "SELECT c.member_no, c.customer_type, r.r_value, r.f_value, r.m_value FROM customer_clusters c LEFT JOIN rfm_analysis r ON c.member_no = r.member_no WHERE c.cluster_label = {0} LIMIT 50"),

        # --- 客户分群 × member_data 跨表关联 ---
        (r'高价值.*性别.*分布|高价值.*男女.*统计',
         "SELECT m.gender, COUNT(*) AS count FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.customer_type = '高价值客户' GROUP BY m.gender ORDER BY count DESC"),
        (r'高价值.*平均.*年龄',
         "SELECT AVG(m.age) AS avg_age, MIN(m.age) AS min_age, MAX(m.age) AS max_age FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.customer_type = '高价值客户'"),
        (r'高价值.*平均.*乘机|高价值.*乘机.*平均',
         "SELECT AVG(m.flight_count) AS avg_flight_count, AVG(m.avg_flight_count) AS avg_annual_fc FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.customer_type = '高价值客户'"),
        (r'高价值.*平均.*(?:总)?积分|高价值.*积分.*平均',
         "SELECT AVG(m.points_sum) AS avg_points, AVG(m.bp_sum) AS avg_bp, AVG(m.ep_sum) AS avg_ep FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.customer_type = '高价值客户'"),
        (r'流失.*平均.*飞行.*公里|流失.*平均.*里程',
         "SELECT AVG(m.seg_km_sum) AS avg_seg_km, AVG(m.flight_count) AS avg_fc FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.customer_type = '流失客户'"),
        (r'各.*客户类型.*平均.*飞行.*公里|各.*分群.*平均.*里程',
         "SELECT c.customer_type, AVG(m.seg_km_sum) AS avg_seg_km, AVG(m.flight_count) AS avg_fc FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no GROUP BY c.customer_type ORDER BY avg_seg_km DESC"),
        (r'各.*客户类型.*乘机.*分布|各.*分群.*乘机.*统计',
         "SELECT c.customer_type, COUNT(*) AS count, AVG(m.flight_count) AS avg_fc, MAX(m.flight_count) AS max_fc, MIN(m.flight_count) AS min_fc FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no GROUP BY c.customer_type ORDER BY avg_fc DESC"),
        (r'重要保持.*平均.*折扣|重要保持.*折扣率',
         "SELECT AVG(m.avg_discount) AS avg_discount, AVG(m.exchange_count) AS avg_exchange FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.customer_type = '重要保持客户'"),
        (r'潜力客户.*平均.*精英.*积分|潜力.*精英积分',
         "SELECT AVG(m.ep_sum) AS avg_ep_sum, AVG(m.bp_sum) AS avg_bp FROM customer_clusters c JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.customer_type = '潜力客户'"),

        # --- member_data 原始客户数据查询 ---
        # -- 人口统计 --
        (r'(?:原始.*)?性别.*(?:分布|比例|多少|统计)|男.*女.*(?:多少|统计)',
         "SELECT gender, COUNT(*) AS count FROM member_data GROUP BY gender ORDER BY count DESC"),
        (r'(?:原始.*)?(?:各.*)?等级.*(?:分布|统计|多少|数量)|FFP.*等级.*(?:分布|统计)',
         "SELECT ffp_tier, COUNT(*) AS count FROM member_data GROUP BY ffp_tier ORDER BY ffp_tier"),
        (r'(?:原始.*)?平均.*年龄',
         "SELECT AVG(age) AS avg_age, MIN(age) AS min_age, MAX(age) AS max_age FROM member_data"),
        (r'(?:原始.*)?年龄.*(?:最大|最长).*前\s*(\d+)',
         "SELECT member_no, age, gender, ffp_tier FROM member_data ORDER BY age DESC LIMIT {0}"),
        (r'(?:原始.*)?年龄.*(?:最小|最年轻).*前\s*(\d+)',
         "SELECT member_no, age, gender, ffp_tier FROM member_data ORDER BY age ASC LIMIT {0}"),
        (r'(?:原始.*)?(?:工作)?(?:城市|省份|国家).*(?:分布|统计|数量)',
         "SELECT work_city, work_province, work_country, COUNT(*) AS count FROM member_data WHERE work_city IS NOT NULL GROUP BY work_city, work_province, work_country ORDER BY count DESC LIMIT 20"),
        (r'(?:原始.*)?各.*工作.*城市.*(?:分布|统计|数量)',
         "SELECT work_city, COUNT(*) AS count FROM member_data WHERE work_city IS NOT NULL GROUP BY work_city ORDER BY count DESC LIMIT 20"),
        (r'(?:原始.*)?各.*工作.*省份.*(?:分布|统计|数量)',
         "SELECT work_province, COUNT(*) AS count FROM member_data WHERE work_province IS NOT NULL GROUP BY work_province ORDER BY count DESC"),
        (r'(?:原始.*)?各.*工作.*国家.*(?:分布|统计|数量)',
         "SELECT work_country, COUNT(*) AS count FROM member_data WHERE work_country IS NOT NULL GROUP BY work_country ORDER BY count DESC"),
        (r'(?:原始.*)?工作.*城市.*(?:为|是|在)(\S+)',
         "SELECT * FROM member_data WHERE work_city = '{0}' LIMIT 50"),

        # -- 乘机行为 --
        (r'(?:原始.*)?乘机次数.*(?:最多|最高|最大).*前\s*(\d+)|飞行次数.*(?:最多|最高).*前\s*(\d+)|乘机.*(?:最多|TOP).*\s*(\d+).*客户',
         "SELECT member_no, flight_count, seg_km_sum, points_sum, avg_flight_count FROM member_data ORDER BY flight_count DESC LIMIT {0}"),
        (r'(?:原始.*)?乘机次数.*(?:最少|最低|最小).*前\s*(\d+)|飞行次数.*(?:最少|最低).*前\s*(\d+)',
         "SELECT member_no, flight_count, seg_km_sum, points_sum, avg_flight_count FROM member_data ORDER BY flight_count ASC LIMIT {0}"),
        (r'(?:原始.*)?乘机次数.*(?:最少|最低|最小).*(?:客户|会员|有哪些|是谁|是哪些)',
         "SELECT member_no, flight_count, seg_km_sum, points_sum, avg_flight_count FROM member_data ORDER BY flight_count ASC LIMIT {0}"),
        (r'(?:原始.*)?总.*飞行.*公里.*(?:最多|前\s*(\d+))|飞行里程.*(?:最多|最高).*前\s*(\d+)|飞行公里.*前\s*(\d+)',
         "SELECT member_no, seg_km_sum, flight_count, weighted_seg_km, points_sum FROM member_data ORDER BY seg_km_sum DESC LIMIT {0}"),
        (r'(?:原始.*)?加权.*飞行.*公里.*(?:最多|前\s*(\d+))',
         "SELECT member_no, weighted_seg_km, seg_km_sum, flight_count FROM member_data ORDER BY weighted_seg_km DESC LIMIT {0}"),
        (r'(?:原始.*)?平均.*乘机次数.*(?:最多|最.*前\s*(\d+))|平均.*飞行次数.*前\s*(\d+)',
         "SELECT member_no, avg_flight_count, flight_count, seg_km_sum FROM member_data ORDER BY avg_flight_count DESC LIMIT {0}"),
        (r'(?:原始.*)?平均.*乘机.*(?:间隔|间距)',
         "SELECT AVG(avg_interval) AS avg_interval, MIN(avg_interval) AS min_interval, MAX(avg_interval) AS max_interval FROM member_data"),
        (r'(?:原始.*)?最大.*乘机.*间隔',
         "SELECT AVG(max_interval) AS avg_max_interval, MAX(max_interval) AS overall_max_interval FROM member_data"),
        (r'(?:原始.*)?(?:最近一|近一|L1Y).*乘机.*(?:前\s*(\d+)|最多)',
         "SELECT member_no, l1y_flight_count, p1y_flight_count, flight_count FROM member_data ORDER BY l1y_flight_count DESC LIMIT {0}"),
        (r'(?:原始.*)?(?:前一|P1Y).*乘机.*(?:前\s*(\d+)|最多)',
         "SELECT member_no, p1y_flight_count, l1y_flight_count, flight_count FROM member_data ORDER BY p1y_flight_count DESC LIMIT {0}"),
        (r'(?:原始.*)?(?:最近一|近一).*乘机.*占比.*(?:最高|最多|前\s*(\d+))',
         "SELECT member_no, ration_l1y_flight_count, l1y_flight_count, p1y_flight_count FROM member_data ORDER BY ration_l1y_flight_count DESC LIMIT {0}"),
        (r'(?:原始.*)?(?:前一|P1Y).*乘机.*占比.*(?:最高|最多|前\s*(\d+))',
         "SELECT member_no, ration_p1y_flight_count, p1y_flight_count, l1y_flight_count FROM member_data ORDER BY ration_p1y_flight_count DESC LIMIT {0}"),
        (r'(?:原始.*)?(?:平均.*)?乘机次数.*(?:统计|摘要|汇总)',
         "SELECT COUNT(*) AS total, AVG(flight_count) AS avg_fc, MIN(flight_count) AS min_fc, MAX(flight_count) AS max_fc, AVG(avg_flight_count) AS avg_annual, AVG(avg_interval) AS avg_ival FROM member_data"),

        # -- 积分分析 --
        (r'(?:原始.*)?总.*积分.*(?:最多|最高|最.*前\s*(\d+))|积分.*TOP.*\s*(\d+)|积分.*排名.*前\s*(\d+)',
         "SELECT member_no, points_sum, bp_sum, ep_sum, add_point_sum, flight_count FROM member_data ORDER BY points_sum DESC LIMIT {0}"),
        (r'(?:原始.*)?基本.*积分.*(?:最多|最高|前\s*(\d+))|BP.*积分.*(?:最多|前\s*(\d+))',
         "SELECT member_no, bp_sum, points_sum, ep_sum, avg_bp_sum FROM member_data ORDER BY bp_sum DESC LIMIT {0}"),
        (r'(?:原始.*)?精英.*积分.*(?:最多|最高|前\s*(\d+))|EP.*积分.*(?:最多|前\s*(\d+))',
         "SELECT member_no, ep_sum, bp_sum, points_sum FROM member_data ORDER BY ep_sum DESC LIMIT {0}"),
        (r'(?:原始.*)?额外.*积分.*(?:最多|最高|前\s*(\d+))',
         "SELECT member_no, add_point_sum, eli_add_point_sum, points_sum FROM member_data ORDER BY add_point_sum DESC LIMIT {0}"),
        (r'(?:原始.*)?(?:最近一|近一|L1Y).*(?:总)?积分.*(?:最多|最高|前\s*(\d+))',
         "SELECT member_no, l1y_points_sum, points_sum, l1y_bp_sum FROM member_data ORDER BY l1y_points_sum DESC LIMIT {0}"),
        (r'(?:原始.*)?(?:最近一|近一).*BP.*占比.*(?:最高|最多|前\s*(\d+))',
         "SELECT member_no, ration_l1y_bps, l1y_bp_sum, bp_sum FROM member_data ORDER BY ration_l1y_bps DESC LIMIT {0}"),
        (r'(?:原始.*)?积分.*兑换.*(?:最多|前\s*(\d+)|统计|分布)',
         "SELECT member_no, exchange_count, points_sum, flight_count FROM member_data ORDER BY exchange_count DESC LIMIT {0}"),
        (r'(?:原始.*)?平均.*折扣.*(?:最低|最小).*前\s*(\d+)|折扣率.*(?:最低|最小).*前\s*(\d+)',
         "SELECT member_no, avg_discount, sum_yr_1, sum_yr_2, flight_count FROM member_data ORDER BY avg_discount ASC LIMIT {0}"),
        (r'(?:原始.*)?平均.*折扣.*(?:最高|最大).*前\s*(\d+)|折扣率.*(?:最高|最大).*前\s*(\d+)',
         "SELECT member_no, avg_discount, sum_yr_1, sum_yr_2, flight_count FROM member_data ORDER BY avg_discount DESC LIMIT {0}"),
        (r'(?:原始.*)?第一年.*票价.*(?:最多|最高|前\s*(\d+))',
         "SELECT member_no, sum_yr_1, sum_yr_2, flight_count, avg_discount FROM member_data ORDER BY sum_yr_1 DESC LIMIT {0}"),
        (r'(?:原始.*)?第二年.*票价.*(?:最多|最高|前\s*(\d+))',
         "SELECT member_no, sum_yr_2, sum_yr_1, flight_count, avg_discount FROM member_data ORDER BY sum_yr_2 DESC LIMIT {0}"),
        (r'(?:原始.*)?非乘机.*积分.*(?:最多|最高|前\s*(\d+))',
         "SELECT member_no, point_not_flight, points_sum, flight_count FROM member_data ORDER BY point_not_flight DESC LIMIT {0}"),
        (r'(?:原始.*)?(?:总)?积分.*(?:统计|摘要|汇总)',
         "SELECT COUNT(*) AS total, AVG(points_sum) AS avg_points, MIN(points_sum) AS min_points, MAX(points_sum) AS max_points, AVG(bp_sum) AS avg_bp, AVG(ep_sum) AS avg_ep FROM member_data"),

        # -- 会员总数与数据量 --
         (r'原始.*(?:总|客户|会员).*(?:多少|数量)|原始数据.*(?:记录|条数)',  # 仅匹配明确提及"原始"的计数问题
          "SELECT COUNT(*) AS total_count FROM member_data"),
        (r'(?:原始.*)?乘机次数.*0.*(?:客户|会员)|未.*乘机.*(?:客户|会员)',
         "SELECT COUNT(*) AS count FROM member_data WHERE flight_count = 0"),
        (r'(?:原始.*)?总.*积分.*超过\s*(\d+).*万',
         "SELECT COUNT(*) AS count FROM member_data WHERE points_sum > {0}0000"),

        # --- 通用查询 ---
        (r'总(?:共)?(?:有)?(?:多少|几|多少个)(?:客户|会员|记录|条)|总(?:客户|会员)(?:数|人数|数量)',
         "SELECT COUNT(DISTINCT member_no) AS total_members FROM customer_clusters"),
        (r'总(?:记录|数据)(?:数|条数)',
         "SELECT 'RFM分析' AS table_name, COUNT(*) AS count FROM rfm_analysis UNION ALL SELECT '客户分群', COUNT(*) FROM customer_clusters UNION ALL SELECT '原始数据', COUNT(*) FROM member_data"),
        (r'批次.*列表|有哪些批次',
         "SELECT DISTINCT batch_no FROM customer_clusters UNION SELECT DISTINCT batch_no FROM rfm_analysis UNION SELECT DISTINCT batch_no FROM member_data ORDER BY batch_no DESC"),
        (r'最新.*批次|最近.*批次',
         "SELECT 'RFM最新' AS type, MAX(batch_no) AS latest_batch FROM rfm_analysis UNION ALL SELECT '分群最新', MAX(batch_no) FROM customer_clusters UNION ALL SELECT '原始最新', MAX(batch_no) FROM member_data"),
        (r'查询.*会员.*?(\d+)',
         "SELECT c.member_no, c.customer_type, c.cluster_label, r.r_value, r.f_value, r.m_value, m.flight_count, m.seg_km_sum, m.points_sum, m.age, m.gender FROM customer_clusters c LEFT JOIN rfm_analysis r ON c.member_no = r.member_no AND c.batch_no = r.batch_no LEFT JOIN member_data m ON c.member_no = m.member_no AND c.batch_no = m.batch_no WHERE c.member_no = '{0}'"),
        (r'R值.*0.*客户|R.*零|最近未消费',
         "SELECT COUNT(*) AS count FROM rfm_analysis WHERE r_value = 0"),
    ]

    def __init__(self, db_manager=None, api_key=None, endpoint=None, model=None):
        """
        初始化查询引擎

        Args:
            db_manager: DatabaseManager 实例（已连接），用于数据库查询模式
            api_key: API密钥（可选，为空时降级为规则方案）
            endpoint: API端点URL（可选）
            model: 模型名称（可选）
        """
        self.db_manager = db_manager
        self.api_key = api_key
        self.endpoint = endpoint or DEFAULT_LLM_CONFIG['endpoint']
        self.model = model or DEFAULT_LLM_CONFIG['model']
        self.client = None
        self.llm_available = False

        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
                self.llm_available = True
            except Exception:
                self.llm_available = False

        self.history = QueryHistory()

    # ================================================================
    # Prompt 构建
    # ================================================================
    def _build_prompt(self, question, schema):
        """
        通用Prompt构建

        Args:
            question: 用户的自然语言问题
            schema: 表结构说明

        Returns:
            构建好的prompt字符串
        """
        prompt = f"""你是一个SQL专家。请根据以下数据库表结构信息和用户问题，生成对应的SQL查询语句。

{self._get_db_version_info()}

{schema}

重要规则：
1. 只生成SELECT查询语句
2. 不要生成任何DELETE、DROP、UPDATE等修改数据的语句
3. 只返回SQL语句，不要返回其他解释内容
4. 使用标准SQL语法（兼容{MYSQL_VERSION}）
5. 字符串值使用单引号
6. 如果不确定字段值，使用LIKE进行模糊匹配

用户问题：{question}

请生成对应的SQL查询语句："""
        return prompt

    @staticmethod
    def _get_db_version_info():
        """获取数据库版本提示信息，用于 LLM prompt 中防止生成不符合语法规则的 SQL"""
        return f"当前数据库版本: {MYSQL_VERSION}，请确保生成的 SQL 符合该版本的语法规则。"

    def build_prompt(self, question, batch_no=None):
        """构建查询Prompt（批次过滤由_inject_batch_filter统一后处理）"""
        return self._build_prompt(question, self.DB_TABLE_SCHEMA)

    def call_llm_api(self, prompt, system_msg=None):
        """
        调用大模型API（使用openai库兼容接口）

        Args:
            prompt: 发送给大模型的prompt
            system_msg: 系统提示消息（可选）

        Returns:
            大模型的响应内容
        """
        if system_msg is None:
            system_msg = f"你是一个MySQL查询生成助手，只输出SQL语句。当前数据库版本是 {MYSQL_VERSION}。"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def match_rule(self, question, rule_patterns):
        """
        基于规则模板匹配自然语言问题，生成SQL

        Args:
            question: 用户的自然语言问题
            rule_patterns: 规则模板列表

        Returns:
            匹配到的SQL语句，未匹配返回None
        """
        for pattern, sql_template in rule_patterns:
            match = re.search(pattern, question)
            if match:
                # 过滤掉None值，确保format只接收有效捕获组
                groups = tuple(g for g in match.groups() if g is not None)
                try:
                    return sql_template.format(*groups)
                except (IndexError, KeyError):
                    # 未捕获到数字参数时（如"最多"而非"前10"），用默认值10替换剩余占位符
                    sql = sql_template
                    for i in range(10):
                        placeholder = '{' + str(i) + '}'
                        if placeholder in sql:
                            sql = sql.replace(placeholder, '10')
                    return sql
        return None

    def extract_sql(self, response):
        """
        从LLM响应中提取SQL语句

        Args:
            response: 大模型的响应文本

        Returns:
            提取出的SQL语句
        """
        # 尝试提取markdown代码块中的SQL
        code_block_pattern = r'```(?:sql)?\s*\n?(.*?)```'
        match = re.search(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 尝试提取以SELECT开头的语句
        select_pattern = r'((?:WITH\s+.*?\s+AS\s*\(.*?\)\s*,?\s*)*SELECT\s+.+?)(?:;|$)'
        match = re.search(select_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return response.strip().rstrip(';').strip()

    def validate_sql(self, sql):
        """
        SQL安全校验

        Args:
            sql: 待校验的SQL语句

        Returns:
            (is_valid, error_message) 元组
        """
        if not sql or not sql.strip():
            return False, "SQL语句为空"

        sql_upper = sql.upper().strip()

        for keyword in self.DANGEROUS_KEYWORDS:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                return False, f"SQL包含危险操作: {keyword}，只允许SELECT查询"

        # 去除行级注释（-- ...）和块级注释（/* ... */）后再检查前缀
        stripped = re.sub(r'--[^\n]*', '', sql_upper)
        stripped = re.sub(r'/\*.*?\*/', '', stripped, flags=re.DOTALL)
        stripped = stripped.strip()

        if not (stripped.startswith('SELECT') or stripped.startswith('WITH')):
            return False, "只允许SELECT查询语句"

        return True, ""

    # ================================================================
    # 查询执行
    # ================================================================
    def execute_query(self, sql):
        """
        执行SQL查询（通过DatabaseManager的MySQL连接）

        Args:
            sql: SQL查询语句

        Returns:
            (result_df, error_message) 元组
        """
        if self.db_manager is None or not self.db_manager.is_connected():
            return None, "数据库未连接，请先在数据库管理页面连接MySQL数据库"
        result_df, error_msg = self.db_manager.execute_query_safe(sql)
        if error_msg:
            return None, error_msg
        return result_df, ""

    # ================================================================
    # 完整查询流程
    # ================================================================
    def _inject_batch_filter(self, sql, batch_no):
        """
        在生成的SQL中注入批次过滤条件

        Args:
            sql: 原始SQL
            batch_no: 批次编号

        Returns:
            注入过滤条件后的SQL
        """
        if not batch_no:
            return sql

        # 如果SQL中已经包含相同的batch_no过滤条件，跳过重复注入
        escaped_batch = batch_no.replace("'", "\\'")
        if f"batch_no = '{escaped_batch}'" in sql or f"batch_no='{escaped_batch}'" in sql:
            return sql

        # 用正则检测SQL子句（兼容换行、Tab等任意空白，而非仅空格）
        has_where = re.search(r'\bWHERE\b', sql, re.IGNORECASE)
        has_group_by = re.search(r'\bGROUP\s+BY\b', sql, re.IGNORECASE)
        has_order_by = re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE)
        has_limit = re.search(r'\bLIMIT\b', sql, re.IGNORECASE)
        has_join = re.search(r'\bJOIN\b', sql, re.IGNORECASE)

        # 批次号统一为 CLUSTER_ 前缀格式，三表共享同一批次号
        # 检测 SQL 中引用的所有表，为每个表注入 batch_no 过滤
        target_tables = []
        for table in ['rfm_analysis', 'customer_clusters', 'member_data']:
            if re.search(rf'\b{table}\b', sql, re.IGNORECASE):
                target_tables.append(table)
        if not target_tables:
            # 向后兼容：未检测到表名时默认注入到 rfm_analysis 和 customer_clusters
            target_tables = ['rfm_analysis', 'customer_clusters', 'member_data']

        # 为每个目标表构建 batch_no 过滤条件
        batch_conditions = []
        for table in target_tables:
            prefix = ''
            if has_join:
                alias_pattern = rf'\b(?:FROM|JOIN)\s+{table}(?:\s+AS)?\s+(\w+)'
                m = re.search(alias_pattern, sql, re.IGNORECASE)
                if m:
                    prefix = m.group(1) + '.'
            batch_conditions.append(f"{prefix}batch_no = '{escaped_batch}'")

        combined_condition = ' AND '.join(batch_conditions)

        # 情况1：SQL中有WHERE子句 → 追加AND条件
        if has_where:
            sql = re.sub(
                r'\bWHERE\b',
                f"WHERE {combined_condition} AND ",
                sql,
                count=1,
                flags=re.IGNORECASE
            )
        # 情况2：SQL中有GROUP BY（没有WHERE）
        elif has_group_by:
            sql = re.sub(
                r'(\bGROUP\s+BY\b)',
                f"WHERE {combined_condition} \\1",
                sql,
                count=1,
                flags=re.IGNORECASE
            )
        # 情况3：SQL中有ORDER BY（没有WHERE和GROUP BY）
        elif has_order_by:
            sql = re.sub(
                r'(\bORDER\s+BY\b)',
                f"WHERE {combined_condition} \\1",
                sql,
                count=1,
                flags=re.IGNORECASE
            )
        # 情况4：SQL中有LIMIT（没有WHERE、GROUP BY、ORDER BY）
        elif has_limit:
            sql = re.sub(
                r'(\bLIMIT\b)',
                f"WHERE {combined_condition} \\1",
                sql,
                count=1,
                flags=re.IGNORECASE
            )
        # 情况5：简单SELECT（结尾无其他子句，可能有分号）
        else:
            sql = sql.rstrip().rstrip(';')
            sql += f" WHERE {combined_condition}"

        return sql

    def query(self, question, batch_no=None, force_rule=False):
        """
        数据库查询流程（查询MySQL数据库）

        Args:
            question: 用户的自然语言问题
            batch_no: 可选的批次编号过滤条件
            force_rule: 是否强制使用规则模式（忽略LLM）

        Returns:
            (sql, result_df, status, error_message) 元组
        """
        sql = None
        result_df = None

        use_llm = self.llm_available and not force_rule

        try:
            if use_llm:
                prompt = self.build_prompt(question, batch_no=batch_no)
                response = self.call_llm_api(prompt)
                sql = self.extract_sql(response)
            else:
                sql = self.match_rule(question, self.DB_RULE_PATTERNS)
                if sql is None:
                    error_msg = "规则方案无法匹配该问题，请在项目根目录 .env 文件中配置 LLM_API_KEY 以启用大模型。"
                    self.history.add(question, None, None, 'error', error_msg)
                    return None, None, 'error', error_msg

            # 注入批次过滤条件
            if batch_no:
                sql = self._inject_batch_filter(sql, batch_no)

            is_valid, error_msg = self.validate_sql(sql)
            if not is_valid:
                self.history.add(question, sql, None, 'error', error_msg)
                return sql, None, 'error', error_msg

            result_df, error_msg = self.execute_query(sql)
            if error_msg:
                self.history.add(question, sql, None, 'error', error_msg)
                return sql, None, 'error', error_msg

            result_preview = result_df.head(10) if len(result_df) > 0 else result_df
            self.history.add(question, sql, result_preview, 'success', '')
            return sql, result_df, 'success', ''

        except Exception as e:
            error_msg = f"查询流程异常: {str(e)}"
            self.history.add(question, sql, None, 'error', error_msg)
            return sql, None, 'error', error_msg

    def _fix_sql_with_llm(self, question, sql, error, batch_no=None):
        """
        让 LLM 修复执行失败的 SQL 语句（自修复）。

        Args:
            question: 原始自然语言问题
            sql: 执行失败的 SQL 语句
            error: 数据库返回的错误信息
            batch_no: 批次编号（可选）

        Returns:
            修复后的 SQL 语句字符串；若 LLM 不可用或修复失败则返回 None
        """
        if not self.llm_available:
            return None

        batch_hint = f"\n批次过滤要求：所有查询必须限定 batch_no = '{batch_no}'。" if batch_no else ""

        prompt = f"""以下 SQL 查询执行失败，请修复它。

{self._get_db_version_info()}

{self.DB_TABLE_SCHEMA}

用户问题: {question}
{batch_hint}

原始 SQL:
```sql
{sql}
```

错误信息:
{error}

请分析错误原因并返回修复后的 SQL。只返回修复后的 SQL 语句，用 ```sql ... ``` 包裹。如果无法修复，请返回空内容。"""

        system_msg = f"你是一个MySQL查询生成助手，只输出SQL语句。当前数据库版本是 {MYSQL_VERSION}。"

        try:
            response = self.call_llm_api(prompt, system_msg=system_msg)
            fixed_sql = self.extract_sql(response)
            return fixed_sql if fixed_sql and fixed_sql.strip() else None
        except Exception:
            return None

    def query_single(self, question, batch_no=None, force_rule=False, max_retries=2, progress_callback=None):
        """
        处理单条自然语言查询，返回结构化结果（供 MCP 工具调用）。

        与 query() 不同，本方法返回结构化 dict，便于 MCP 工具统一序列化。
        当 SQL 执行失败且 LLM 可用时，会自动进行自修复重试（最多 max_retries 次）。

        Args:
            question: 用户的自然语言问题
            batch_no: 可选的批次编号过滤条件
            force_rule: 是否强制使用规则模式（忽略LLM）
            max_retries: SQL 执行失败时的最大自修复重试次数（默认 2）
            progress_callback: 可选的进度回调函数，接收 dict 参数

        Returns:
            dict: {
                'question': str,
                'sql': str | None,           # 最终执行的 SQL（可能是修复后的）
                'status': 'success' | 'error',
                'row_count': int,
                'data': list | None,         # DataFrame 转为 list of dict（前 100 行）
                'error': str,                # 空字符串表示无错误
                'original_sql': str | None,  # 首次失败的 SQL（若发生过重试）
                'original_error': str,       # 首次失败的错误信息（若发生过重试）
                'was_fixed': bool,           # 是否经过自修复后成功
                'retry_count': int,          # 实际重试次数
            }
        """

        def _notify(phase, **extra):
            if progress_callback:
                try:
                    progress_callback({'question': question, 'phase': phase, **extra})
                except Exception:
                    pass

        result = {
            'question': question,
            'sql': None,
            'status': 'error',
            'row_count': 0,
            'data': None,
            'error': '',
            'original_sql': None,
            'original_error': '',
            'was_fixed': False,
            'retry_count': 0,
        }

        sql = None
        use_llm = self.llm_available and not force_rule

        try:
            _notify('generating_sql')
            if use_llm:
                prompt = self.build_prompt(question, batch_no=batch_no)
                response = self.call_llm_api(prompt)
                sql = self.extract_sql(response)
            else:
                sql = self.match_rule(question, self.DB_RULE_PATTERNS)
                if sql is None:
                    result['error'] = "规则方案无法匹配该问题"
                    _notify('done', result=result)
                    return result

            # 注入批次过滤条件
            if batch_no:
                sql = self._inject_batch_filter(sql, batch_no)

            result['sql'] = sql

            # 安全校验
            is_valid, error_msg = self.validate_sql(sql)
            if not is_valid:
                result['error'] = error_msg
                _notify('done', result=result)
                return result

            # 执行查询
            _notify('executing', sql=sql)
            result_df, exec_error = self.execute_query(sql)

            # 失败时尝试自修复重试（仅在 LLM 可用时）
            if exec_error and use_llm and max_retries > 0:
                original_sql = sql
                original_error = exec_error
                current_sql = sql
                current_error = exec_error

                for attempt in range(max_retries):
                    result['retry_count'] = attempt + 1
                    _notify('retrying', attempt=attempt + 1, max_retries=max_retries, error=exec_error)

                    # 让 LLM 修复 SQL
                    fixed_sql = self._fix_sql_with_llm(question, current_sql, current_error, batch_no)
                    if not fixed_sql or fixed_sql.strip() == current_sql.strip():
                        # LLM 未能生成新 SQL，停止重试
                        break

                    # 注入批次过滤条件到修复后的 SQL
                    if batch_no:
                        fixed_sql = self._inject_batch_filter(fixed_sql, batch_no)

                    # 校验修复后的 SQL
                    is_valid, fix_valid_error = self.validate_sql(fixed_sql)
                    if not is_valid:
                        current_sql = fixed_sql
                        current_error = fix_valid_error
                        continue

                    # 执行修复后的 SQL
                    _notify('executing', sql=fixed_sql, retry=attempt + 1)
                    result_df, exec_error = self.execute_query(fixed_sql)
                    if not exec_error:
                        # 修复成功
                        sql = fixed_sql
                        result['sql'] = sql
                        result['original_sql'] = original_sql
                        result['original_error'] = original_error
                        result['was_fixed'] = True
                        break
                    else:
                        # 仍然失败，继续下一次重试
                        current_sql = fixed_sql
                        current_error = exec_error

                # 重试后仍然失败
                if exec_error:
                    result['error'] = exec_error
                    result['original_sql'] = original_sql
                    result['original_error'] = original_error
                    result['was_fixed'] = False
                    _notify('done', result=result)
                    return result

            elif exec_error:
                # 无 LLM 或不重试，直接返回错误
                result['error'] = exec_error
                _notify('done', result=result)
                return result

            # 成功
            result['status'] = 'success'
            result['row_count'] = len(result_df)
            # 转为 list of dict，限制最多 100 行避免过大
            result['data'] = result_df.head(100).to_dict(orient='records')
            _notify('done', result=result)
            return result

        except Exception as e:
            result['error'] = f"查询流程异常: {str(e)}"
            result['sql'] = sql
            _notify('done', result=result)
            return result

    def query_batch(self, questions, batch_no=None, max_workers=5, force_rule=False, max_retries=2, progress_callback=None):
        """
        批量并发处理多条自然语言问题（供 MCP 工具调用）。

        使用线程池并发执行 query_single，每条问题独立调用 LLM 与数据库查询。
        数据库查询通过 DatabaseManager.execute_query_safe（连接池）保证线程安全。
        每条问题独立进行自修复重试（最多 max_retries 次）。

        Args:
            questions: 自然语言问题列表（1-5 条）
            batch_no: 可选的批次编号过滤条件
            max_workers: 最大并发数（默认 5）
            force_rule: 是否强制使用规则模式
            max_retries: SQL 执行失败时的最大自修复重试次数（默认 2）
            progress_callback: 可选的进度回调函数，接收 dict 参数

        Returns:
            list[dict]: 每条问题对应一个 query_single 返回的结构化 dict，顺序与输入一致
        """
        if not questions:
            return []

        # 限制并发数
        max_workers = min(max_workers, len(questions), 5)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(questions)

        def _process(idx, question):
            return idx, self.query_single(
                question, batch_no=batch_no, force_rule=force_rule,
                max_retries=max_retries, progress_callback=progress_callback
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process, i, q): i
                for i, q in enumerate(questions)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def get_history(self):
        """获取查询历史"""
        return self.history.get_all()

    def clear_history(self):
        """清空查询历史"""
        self.history.clear()
