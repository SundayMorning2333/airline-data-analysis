"""
数据清洗与预处理模块

提供 CSV 数据的读取、缺失值处理、异常值检测与处理、
数据类型转换、去重以及数据质量报告生成等功能。
"""

import os
import chardet
import pandas as pd
import numpy as np


class DataCleaner:
    """数据清洗与预处理类，用于对航空客户数据进行系统化清洗。"""

    # 日期类型字段
    DATE_COLUMNS = [
        'FFP_DATE', 'FIRST_FLIGHT_DATE', 'LOAD_TIME', 'LAST_FLIGHT_DATE'
    ]

    # 数值类型字段
    NUMERIC_COLUMNS = [
        'AGE', 'FLIGHT_COUNT', 'BP_SUM', 'EP_SUM_YR_1', 'EP_SUM_YR_2',
        'SUM_YR_1', 'SUM_YR_2', 'SEG_KM_SUM', 'WEIGHTED_SEG_KM',
        'AVG_FLIGHT_COUNT', 'AVG_BP_SUM', 'BEGIN_TO_FIRST', 'LAST_TO_END',
        'AVG_INTERVAL', 'MAX_INTERVAL', 'ADD_POINTS_SUM_YR_1',
        'ADD_POINTS_SUM_YR_2', 'EXCHANGE_COUNT', 'avg_discount',
        'P1Y_Flight_Count', 'L1Y_Flight_Count', 'P1Y_BP_SUM', 'L1Y_BP_SUM',
        'EP_SUM', 'ADD_Point_SUM', 'Eli_Add_Point_Sum', 'L1Y_ELi_Add_Points',
        'Points_Sum', 'L1Y_Points_Sum', 'Ration_L1Y_Flight_Count',
        'Ration_P1Y_Flight_Count', 'Ration_P1Y_BPS', 'Ration_L1Y_BPS',
        'Point_NotFlight', 'FFP_TIER'
    ]

    # 分类类型字段
    CATEGORICAL_COLUMNS = ['GENDER', 'WORK_CITY', 'WORK_PROVINCE', 'WORK_COUNTRY']

    def __init__(self, file_path=None, encoding=None):
        """
        初始化 DataCleaner。

        Parameters
        ----------
        file_path : str, optional
            CSV 文件路径。若提供则在初始化时自动加载。
        encoding : str, optional
            CSV 文件编码。为 None 时自动检测。
        """
        self.file_path = file_path
        self.encoding = encoding
        self.raw_data = None
        self.data = None
        self.outlier_mask = None

        if file_path is not None:
            self.load_data(file_path)

    # ------------------------------------------------------------------
    # 编码检测
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_encoding(file_path):
        """
        检测文件编码。

        Parameters
        ----------
        file_path : str
            文件路径。

        Returns
        -------
        str
            检测到的编码名称。
        """
        with open(file_path, 'rb') as f:
            raw = f.read(min(os.path.getsize(file_path), 100000))
        result = chardet.detect(raw)
        return result['encoding']

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------
    def load_data(self, file_path=None):
        """
        读取 CSV 文件并存储为 DataFrame。

        Parameters
        ----------
        file_path : str, optional
            CSV 文件路径，为 None 时使用初始化时的路径。

        Returns
        -------
        pandas.DataFrame
            读取到的数据。
        """
        if file_path is not None:
            self.file_path = file_path
        if self.file_path is None:
            raise ValueError("未指定文件路径，请在初始化时或调用 load_data 时提供 file_path。")

        encoding = self.encoding or self._detect_encoding(self.file_path)
        # gb2312 是 gb18030 的子集，遇到罕见字符可能解码失败，因此向上兼容
        if encoding and encoding.lower() in ('gb2312', 'gbk'):
            encoding = 'gb18030'
        try:
            self.raw_data = pd.read_csv(self.file_path, encoding=encoding)
        except UnicodeDecodeError:
            self.raw_data = pd.read_csv(self.file_path, encoding='gb18030')
        self.data = self.raw_data.copy()
        self.outlier_mask = None
        return self.data

    # ------------------------------------------------------------------
    # 数据概览
    # ------------------------------------------------------------------
    def overview(self):
        """
        返回数据概览信息。

        Returns
        -------
        dict
            包含以下键值:
            - shape:          (行数, 列数)
            - dtypes:         各列数据类型
            - missing_counts: 各列缺失值数量
            - missing_ratio:  各列缺失值比例
            - duplicates:     按 MEMBER_NO 的重复行数
            - describe:       数值列的统计描述
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        missing = self.data.isnull().sum()
        result = {
            'shape': self.data.shape,
            'dtypes': self.data.dtypes,
            'missing_counts': missing,
            'missing_ratio': (missing / len(self.data) * 100).round(2),
            'duplicates': self.data.duplicated(subset=['MEMBER_NO']).sum()
                          if 'MEMBER_NO' in self.data.columns else 0,
            'describe': self.data.describe(include='all'),
        }
        return result

    # ------------------------------------------------------------------
    # 缺失值处理
    # ------------------------------------------------------------------
    def handle_missing_values(self, strategy='drop'):
        """
        处理缺失值。

        Parameters
        ----------
        strategy : str
            处理策略:
            - 'do nothing' : 不处理缺失值
            - 'drop'       : 删除含缺失值的行
            - 'mean'       : 数值列用均值填充，分类列用众数填充
            - 'median'     : 数值列用中位数填充，分类列用众数填充
            - 'mode'       : 所有列用众数填充

        Returns
        -------
        pandas.DataFrame
            处理后的数据。
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        if strategy == 'do nothing':
            # 不处理缺失值
            return self.data

        if strategy == 'drop':
            self.data = self.data.dropna()
        elif strategy in ('mean', 'median', 'mode'):
            for col in self.data.columns:
                if self.data[col].isnull().sum() == 0:
                    continue
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    if strategy == 'mean':
                        fill_value = self.data[col].mean()
                    elif strategy == 'median':
                        fill_value = self.data[col].median()
                    else:  # mode
                        mode_val = self.data[col].mode()
                        fill_value = mode_val.iloc[0] if len(mode_val) > 0 else np.nan
                else:
                    mode_val = self.data[col].mode()
                    fill_value = mode_val.iloc[0] if len(mode_val) > 0 else np.nan
                self.data[col] = self.data[col].fillna(fill_value)
        else:
            raise ValueError(f"不支持的缺失值处理策略: {strategy}，可选: do nothing/drop/mean/median/mode")

        self.data = self.data.reset_index(drop=True)
        return self.data

    # ------------------------------------------------------------------
    # 数据类型转换
    # ------------------------------------------------------------------
    def convert_dtypes(self):
        """
        将各列转换为合适的数据类型。

        - 日期列转为 datetime
        - 数值列转为 float（无法转换的设为 NaN）
        - 分类列转为 category

        Returns
        -------
        pandas.DataFrame
            转换类型后的数据。
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        # 日期列
        for col in self.DATE_COLUMNS:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')

        # 数值列：先转换已知的数值列，再自动检测其他可转换的列
        for col in self.NUMERIC_COLUMNS:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # 自动检测并转换其他 object 列为数值（如果可以转换）
        for col in self.data.select_dtypes(include=['object']).columns:
            if col in self.DATE_COLUMNS or col in self.CATEGORICAL_COLUMNS:
                continue
            converted = pd.to_numeric(self.data[col], errors='coerce')
            # 如果至少有 50% 的值能成功转换，就认为是数值列
            if converted.notna().sum() > len(converted) * 0.5:
                self.data[col] = converted

        # MEMBER_NO 确保为整数字符串（避免 Streamlit Arrow 序列化失败，同时防止浮点格式 xxx.0）
        if 'MEMBER_NO' in self.data.columns:
            self.data['MEMBER_NO'] = pd.to_numeric(self.data['MEMBER_NO'], errors='coerce').astype('Int64').astype(str)
            self.data['MEMBER_NO'] = self.data['MEMBER_NO'].replace('<NA>', None)

        # 分类列
        for col in self.CATEGORICAL_COLUMNS:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype('category')

        return self.data

    # ------------------------------------------------------------------
    # 异常值检测
    # ------------------------------------------------------------------
    def detect_outliers(self, method='iqr'):
        """
        检测数值列中的异常值。

        Parameters
        ----------
        method : str
            检测方法:
            - 'do nothing' : 不检测异常值
            - 'iqr'        : 四分位距法 (IQR * 1.5)
            - 'zscore'     : Z-score 法 (|z| > 3)

        Returns
        -------
        dict
            各数值列的异常值布尔掩码（True = 异常值）。
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        if method == 'do nothing':
            # 不检测异常值
            self.outlier_mask = {}
            return {}

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        outlier_dict = {}

        if method == 'iqr':
            for col in numeric_cols:
                series = self.data[col].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_dict[col] = (self.data[col] < lower) | (self.data[col] > upper)

        elif method == 'zscore':
            for col in numeric_cols:
                series = self.data[col].dropna()
                if series.empty or series.std() == 0:
                    continue
                z_scores = np.abs((self.data[col] - series.mean()) / series.std())
                outlier_dict[col] = z_scores > 3

        else:
            raise ValueError(f"不支持的异常值检测方法: {method}，可选: do nothing/iqr/zscore")

        self.outlier_mask = outlier_dict
        return outlier_dict

    # ------------------------------------------------------------------
    # 异常值处理
    # ------------------------------------------------------------------
    def handle_outliers(self, strategy='remove', method='iqr'):
        """
        根据检测结果处理异常值。

        Parameters
        ----------
        strategy : str
            处理策略:
            - 'do nothing' : 不处理异常值
            - 'remove'     : 删除含有异常值的行
            - 'clip'       : 将异常值裁剪到边界值
        method : str
            边界计算方法（仅在 strategy='clip' 时使用）:
            - 'do nothing' : 不计算边界
            - 'iqr'        : 使用 IQR 边界 (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            - 'zscore'     : 使用 Z-score 边界 (mean - 3*std, mean + 3*std)

        Returns
        -------
        pandas.DataFrame
            处理后的数据。
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        if strategy == 'do nothing':
            # 不处理异常值
            return self.data

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if strategy == 'remove':
            # 如果已经调用过 detect_outliers 则复用其结果，否则重新检测
            if self.outlier_mask is None:
                self.detect_outliers(method=method)

            combined_mask = pd.Series(False, index=self.data.index)
            for col, mask in self.outlier_mask.items():
                combined_mask = combined_mask | mask.fillna(False)
            self.data = self.data[~combined_mask].reset_index(drop=True)

        elif strategy == 'clip':
            for col in numeric_cols:
                series = self.data[col].dropna()
                if series.empty:
                    continue

                if method == 'zscore':
                    # Z-score 边界: mean ± 3*std
                    mean_val = series.mean()
                    std_val = series.std()
                    if std_val == 0:
                        continue
                    lower = mean_val - 3 * std_val
                    upper = mean_val + 3 * std_val
                else:
                    # IQR 边界: Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr

                self.data[col] = self.data[col].clip(lower=lower, upper=upper)

        else:
            raise ValueError(f"不支持的异常值处理策略: {strategy}，可选: do nothing/remove/clip")

        self.outlier_mask = None
        return self.data

    # ------------------------------------------------------------------
    # 数据去重
    # ------------------------------------------------------------------
    def remove_duplicates(self, subset=None):
        """
        去除重复行。

        Parameters
        ----------
        subset : list[str], optional
            用于判断重复的列名列表。默认使用 ['MEMBER_NO']。

        Returns
        -------
        pandas.DataFrame
            去重后的数据。
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        if subset is None:
            subset = ['MEMBER_NO'] if 'MEMBER_NO' in self.data.columns else None

        before = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep='first')
        self.data = self.data.reset_index(drop=True)
        after = len(self.data)
        print(f"去重完成: 删除了 {before - after} 条重复记录，剩余 {after} 条。")
        return self.data

    # ------------------------------------------------------------------
    # 数据质量报告
    # ------------------------------------------------------------------
    def generate_quality_report(self):
        """
        生成数据质量报告。

        Returns
        -------
        dict
            包含以下键值:
            - total_rows:       总行数
            - total_columns:    总列数
            - missing_info:     各列缺失值统计 (列名, 缺失数, 缺失率%)
            - duplicate_rows:   重复行数
            - dtype_summary:    各列类型统计
            - numeric_stats:    数值列关键统计量
            - outlier_summary:  异常值摘要 (基于 IQR 方法)
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        total_rows = len(self.data)
        total_cols = len(self.data.columns)

        # 缺失值信息
        missing = self.data.isnull().sum()
        missing_info = pd.DataFrame({
            'column': missing.index,
            'missing_count': missing.values,
            'missing_ratio_%': (missing / total_rows * 100).round(2).values
        }).sort_values('missing_count', ascending=False).reset_index(drop=True)

        # 重复行
        dup_rows = 0
        if 'MEMBER_NO' in self.data.columns:
            dup_rows = self.data.duplicated(subset=['MEMBER_NO']).sum()

        # 类型统计
        dtype_summary = self.data.dtypes.value_counts().to_dict()

        # 数值列统计
        numeric_data = self.data.select_dtypes(include=[np.number])
        numeric_stats = numeric_data.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        # 异常值摘要
        outlier_summary = {}
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = ((series < lower) | (series > upper)).sum()
            if count > 0:
                outlier_summary[col] = {
                    'outlier_count': int(count),
                    'outlier_ratio_%': round(count / len(series) * 100, 2),
                    'lower_bound': round(lower, 4),
                    'upper_bound': round(upper, 4)
                }

        report = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'missing_info': missing_info,
            'duplicate_rows': dup_rows,
            'dtype_summary': dtype_summary,
            'numeric_stats': numeric_stats,
            'outlier_summary': outlier_summary,
        }
        return report

    # ------------------------------------------------------------------
    # 获取清洗后数据
    # ------------------------------------------------------------------
    def get_clean_data(self):
        """
        返回清洗后的 DataFrame 副本。

        Returns
        -------
        pandas.DataFrame
            当前清洗状态下的数据。
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")
        return self.data.copy()

    # ------------------------------------------------------------------
    # 一键清洗流水线
    # ------------------------------------------------------------------
    def run_pipeline(self, missing_strategy='drop', outlier_method='iqr',
                     outlier_strategy='remove'):
        """
        一键执行完整清洗流程。

        步骤: 去重 → 类型转换 → 缺失值处理 → 异常值检测 → 异常值处理

        Parameters
        ----------
        missing_strategy : str
            缺失值处理策略，默认 'drop'。
        outlier_method : str
            异常值检测方法，默认 'iqr'。
        outlier_strategy : str
            异常值处理策略，默认 'remove'。

        Returns
        -------
        pandas.DataFrame
            完整清洗后的数据。
        """
        if self.data is None:
            raise RuntimeError("请先调用 load_data 加载数据。")

        print("=" * 50)
        print("开始执行数据清洗流水线")
        print("=" * 50)

        # Step 1: 去重
        print(f"\n[1/5] 数据去重 (按 MEMBER_NO) ...")
        self.remove_duplicates()

        # Step 2: 数据类型转换
        print(f"\n[2/5] 数据类型转换 ...")
        self.convert_dtypes()
        print(f"  完成。")

        # Step 3: 缺失值处理
        print(f"\n[3/5] 缺失值处理 (策略: {missing_strategy}) ...")
        if missing_strategy == 'do nothing':
            print(f"  跳过。")
        else:
            before = len(self.data)
            self.handle_missing_values(strategy=missing_strategy)
            after = len(self.data)
            print(f"  完成。处理前 {before} 行, 处理后 {after} 行。")

        # Step 4: 异常值检测
        print(f"\n[4/5] 异常值检测 (方法: {outlier_method}) ...")
        if outlier_method == 'do nothing':
            outlier_dict = {}
            print(f"  跳过。")
        else:
            outlier_dict = self.detect_outliers(method=outlier_method)
            total_outliers = sum(mask.sum() for mask in outlier_dict.values())
            print(f"  完成。共检测到 {total_outliers} 个异常值点。")

        # Step 5: 异常值处理
        print(f"\n[5/5] 异常值处理 (策略: {outlier_strategy}, 方法: {outlier_method}) ...")
        if outlier_strategy == 'do nothing':
            print(f"  跳过。")
        elif outlier_strategy == 'clip':
            # clip策略：统计被裁剪的异常值数量
            clipped_count = 0
            for col, mask in outlier_dict.items():
                if mask.any():
                    clipped_count += int(mask.sum())
            self.handle_outliers(strategy=outlier_strategy, method=outlier_method)
            print(f"  完成。共裁剪了 {clipped_count} 个异常值到边界值。")
        else:
            # remove策略：统计删除的行数
            before = len(self.data)
            self.handle_outliers(strategy=outlier_strategy, method=outlier_method)
            after = len(self.data)
            print(f"  完成。处理前 {before} 行, 处理后 {after} 行。")

        print("\n" + "=" * 50)
        print(f"清洗流水线完成! 最终数据: {self.data.shape[0]} 行 × {self.data.shape[1]} 列")
        print("=" * 50)

        return self.data
