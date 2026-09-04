"""
数据分析模块 - RFM 模型计算与 K-Means 聚类分析
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config.settings import CLUSTERING_DEFAULTS
# 抑制 sklearn 矩阵运算中的数值警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')


class RFMAnalyzer:
    """RFM 模型分析器，基于客户乘机数据计算 Recency、Frequency、Mileage 指标。"""

    def __init__(self, data, reference_date=None):
        """
        初始化 RFM 分析器。

        Parameters
        ----------
        data : pandas.DataFrame
            清洗后的客户数据，需包含 MEMBER_NO, LAST_FLIGHT_DATE, FLIGHT_COUNT, SEG_KM_SUM 列。
        reference_date : str or datetime-like, optional
            参考日期，用于计算 Recency。默认为 None，此时取 LAST_FLIGHT_DATE 的最大值。
        """
        # 优化：只在需要写入时才 copy，避免对只读视图的整表复制
        # 调用方传入的 clean_data 已经在 session_state 中，本对象仅做只读访问
        self.data = data
        self.reference_date = reference_date
        self.rfm_data = None

    def calculate_rfm(self):
        """
        计算 RFM 指标。

        Returns
        -------
        pandas.DataFrame
            包含 MEMBER_NO, R, F, M 列的 DataFrame。
        """
        # 优化：直接用列引用构建新 DataFrame，避免 .copy() 整表
        # 只取需要的 4 列，且通过显式构造 DataFrame 一次成型，避免 SettingWithCopyWarning
        df = pd.DataFrame({
            'MEMBER_NO': self.data['MEMBER_NO'].values,
            'LAST_FLIGHT_DATE': pd.to_datetime(
                self.data['LAST_FLIGHT_DATE'], format='mixed', errors='coerce'
            ).values,
            'FLIGHT_COUNT': self.data['FLIGHT_COUNT'].values,
            'SEG_KM_SUM': self.data['SEG_KM_SUM'].values,
        })

        # 确定参考日期
        if self.reference_date is not None:
            ref = pd.to_datetime(self.reference_date)
        else:
            ref = df['LAST_FLIGHT_DATE'].max()

        # 计算 R：最近一次乘机距参考日期的天数
        df['R'] = (ref - df['LAST_FLIGHT_DATE']).dt.days

        # F：乘机次数
        df['F'] = df['FLIGHT_COUNT']

        # M：总飞行里程
        df['M'] = df['SEG_KM_SUM']

        # 过滤无效日期导致的 NaN 行
        # 优化：直接赋值切片，避免再次 .copy()
        self.rfm_data = df.loc[df['R'].notna(), ['MEMBER_NO', 'R', 'F', 'M']]
        return self.rfm_data

    def get_rfm_summary(self):
        """
        返回 RFM 各指标的统计摘要。

        Returns
        -------
        pandas.DataFrame
            包含 count, mean, std, min, 25%, 50%, 75%, max 等统计量的摘要表。
        """
        if self.rfm_data is None:
            raise ValueError("请先调用 calculate_rfm() 计算 RFM 指标。")
        return self.rfm_data[['R', 'F', 'M']].describe()


class ClusterAnalyzer:
    """K-Means 聚类分析器，基于 RFM 数据进行客户分群。"""

    def __init__(self, rfm_data):
        """
        初始化聚类分析器。

        Parameters
        ----------
        rfm_data : pandas.DataFrame
            RFM 数据，需包含 MEMBER_NO, R, F, M 列。
        """
        # 优化：去掉 .copy() —— 后续操作都是新增列，不影响原 rfm_data；
        # 在真正需要写入的 fit/get_result_dataframe 中再按需复制
        self.rfm_data = rfm_data
        self.scaler = StandardScaler()
        self.rfm_scaled = None
        self.model = None
        self.labels = None
        self.label_names = None
        self.k = None

    def standardize(self):
        """
        使用 StandardScaler 对 RFM 数据标准化。

        Returns
        -------
        numpy.ndarray
            标准化后的 RFM 数据数组。
        """
        rfm_values = self.rfm_data[['R', 'F', 'M']].values.astype(np.float64)
        # 处理 NaN 和无穷大值
        rfm_values = np.nan_to_num(rfm_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.rfm_scaled = self.scaler.fit_transform(rfm_values)
        return self.rfm_scaled

    def elbow_method(self, k_range=range(2, 11)):
        """
        肘部法确定最佳 K 值。

        Parameters
        ----------
        k_range : range, optional
            K 值搜索范围，默认 range(2, 11)。

        Returns
        -------
        tuple
            (recommended_k, sse_list)
            - recommended_k : int，推荐的最佳 K 值。
            - sse_list : list，每个 K 对应的 SSE 列表。
        """
        if self.rfm_scaled is None:
            self.standardize()

        sse_list = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(self.rfm_scaled)
            sse_list.append(km.inertia_)

        # 使用"肘部"启发法：计算相邻两点间斜率变化最大的点
        deltas = np.diff(sse_list)
        delta_deltas = np.diff(deltas)
        recommended_idx = np.argmax(np.abs(delta_deltas)) + 2  # +2 因为 diff 操作减少的索引偏移
        recommended_k = list(k_range)[recommended_idx]

        return recommended_k, sse_list

    def fit(self, k=CLUSTERING_DEFAULTS['k'], random_state=CLUSTERING_DEFAULTS['random_state'], n_init=CLUSTERING_DEFAULTS['n_init']):
        """
        执行 K-Means 聚类。

        Parameters
        ----------
        k : int, optional
            聚类簇数。
        random_state : int or None, optional
            随机种子，确保结果可复现。设置为 None 则每次运行结果可能不同。
        n_init : int, optional
            算法运行次数，每次用不同初始化，选择最优结果。值越大越稳定但越慢。

        Returns
        -------
        numpy.ndarray
            聚类标签数组。
        """
        if self.rfm_scaled is None:
            self.standardize()

        self.k = k
        self.model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        self.labels = self.model.fit_predict(self.rfm_scaled)
        return self.labels

    # 各 K 值对应的分群标签（按级别从低到高排列，索引0为最低级别）
    LABEL_POOL = {
        2: ['低价值客户', '高价值客户'],
        3: ['低价值客户', '一般客户', '高价值客户'],
        4: ['低价值客户', '一般客户', '重要客户', '高价值客户'],
        5: ['低价值客户', '一般客户', '重要发展客户', '重要保持客户', '高价值客户'],
        6: ['低价值客户', '潜力客户', '一般客户', '重要发展客户', '重要保持客户', '高价值客户'],
        7: ['低价值客户', '新客户', '潜力客户', '一般客户', '重要发展客户', '重要保持客户', '高价值客户'],
        8: ['流失客户', '低价值客户', '一般保持客户', '一般发展客户', '潜力客户', '重要发展客户', '重要保持客户', '高价值客户'],
        9: ['流失客户', '低价值客户', '新客户', '一般保持客户', '一般发展客户', '潜力客户', '重要发展客户', '重要保持客户', '高价值客户'],
        10: ['休眠客户', '流失客户', '低价值客户', '新客户', '一般保持客户', '一般发展客户', '潜力客户', '重要发展客户', '重要保持客户', '高价值客户'],
    }

    def assign_labels(self):
        """
        根据聚类中心特征自动映射客户分群标签。

        综合得分 = F_normalized + M_normalized - R_normalized
        按得分升序排列（得分越低级别越低），重新分配 ID，使得 ID 越小级别越低。

        Returns
        -------
        dict
            {聚类编号: 分群名称} 的映射字典。
        """
        if self.model is None:
            raise ValueError("请先调用 fit() 执行聚类。")

        centers = self.model.cluster_centers_

        # 综合得分：F 越大越好、M 越大越好、R 越小越好
        scores = centers[:, 1] + centers[:, 2] - centers[:, 0]

        # 根据 K 值获取标签池，未定义的 K 值使用兜底方案
        # 标签池从低到高排列（索引0为最低级别）
        label_pool = self.LABEL_POOL.get(self.k, [f'客户群{i + 1}' for i in range(self.k)])

        # 按得分升序排列：得分最低的排在前面（低价值），得分最高的排在后面（高价值）
        sorted_indices = np.argsort(scores)

        # 构建旧ID到新ID的映射：得分最低的映射到0，依次递增
        old_to_new = {}
        label_mapping = {}
        for new_id, old_id in enumerate(sorted_indices):
            old_to_new[old_id] = new_id
            label_mapping[new_id] = label_pool[new_id]

        # 更新聚类标签：将旧的聚类ID映射为新的ID
        self.labels = np.array([old_to_new[label] for label in self.labels])
        self.label_names = np.array([label_mapping[label] for label in self.labels])

        # 更新聚类中心的顺序
        self.model.cluster_centers_ = centers[sorted_indices]

        return label_mapping

    def get_cluster_centers(self):
        """
        返回聚类中心坐标（标准化后的）。

        Returns
        -------
        pandas.DataFrame
            包含各簇中心 R、F、M 标准化值的 DataFrame。
        """
        if self.model is None:
            raise ValueError("请先调用 fit() 执行聚类。")

        centers = self.model.cluster_centers_
        return pd.DataFrame(centers, columns=['R_scaled', 'F_scaled', 'M_scaled'])

    def get_cluster_summary(self):
        """
        返回各簇的 RFM 均值、客户数量和占比。

        Returns
        -------
        pandas.DataFrame
            包含各簇 R_mean, F_mean, M_mean, 客户数量, 占比 的汇总表。
        """
        if self.labels is None:
            raise ValueError("请先调用 fit() 执行聚类。")

        # 优化：用 groupby 直接对原数据聚合，避免 copy 整表后再加列
        # 仅需为 groupby 提供 Cluster 序列
        tmp = self.rfm_data[['R', 'F', 'M', 'MEMBER_NO']].copy()
        tmp['Cluster'] = self.labels

        summary = tmp.groupby('Cluster').agg(
            R_mean=('R', 'mean'),
            F_mean=('F', 'mean'),
            M_mean=('M', 'mean'),
            客户数量=('MEMBER_NO', 'count')
        )
        summary['占比'] = (summary['客户数量'] / summary['客户数量'].sum() * 100).round(2)
        return summary

    def get_result_dataframe(self):
        """
        返回完整的聚类结果 DataFrame。

        Returns
        -------
        pandas.DataFrame
            包含 MEMBER_NO, R, F, M, R_scaled, F_scaled, M_scaled, Cluster, 客户分群 列。
        """
        if self.labels is None:
            raise ValueError("请先调用 fit() 执行聚类。")
        if self.label_names is None:
            self.assign_labels()

        # 优化：只 copy 必要列，避免 copy 整表
        result = self.rfm_data[['MEMBER_NO', 'R', 'F', 'M']].copy()

        # 添加标准化值
        result['R_scaled'] = self.rfm_scaled[:, 0]
        result['F_scaled'] = self.rfm_scaled[:, 1]
        result['M_scaled'] = self.rfm_scaled[:, 2]

        # 添加聚类标签
        result['Cluster'] = self.labels

        # 添加分群名称
        result['客户分群'] = self.label_names

        return result
