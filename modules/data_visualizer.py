"""数据可视化模块 - 客户分群RFM分析图表"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """数据可视化类，用于绘制客户分群分析图表"""

    def __init__(self, cluster_data):
        """
        初始化

        Parameters
        ----------
        cluster_data : pandas.DataFrame
            含分群标签的聚类结果，需包含以下列：
            MEMBER_NO, Recency, Frequency, Mileage, Cluster, Customer_Label
        """
        self.cluster_data = cluster_data

    # ------------------------------------------------------------------
    # 饼图：客户分群分布
    # ------------------------------------------------------------------
    def pie_chart(self, figsize=(8, 6)):
        """
        客户分群分布饼图

        - 显示各分群客户数量及百分比
        - 突出显示最大分群

        Returns
        -------
        matplotlib.figure.Figure
        """
        counts = self.cluster_data['Customer_Label'].value_counts()
        labels = counts.index.tolist()
        sizes = counts.values

        # 最大分群突出显示
        explode = [0.05 if i == np.argmax(sizes) else 0 for i in range(len(sizes))]

        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            shadow=True,
        )
        ax.set_title('客户分群分布', fontsize=14, fontweight='bold')
        ax.axis('equal')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 三维散点图：RFM (交互式)
    # ------------------------------------------------------------------
    def scatter_3d(self, figsize=(10, 8)):
        """
        RFM三维散点图（交互式）

        - 以 Recency / Frequency / Mileage 为三个轴
        - 不同颜色代表不同分群
        - 支持旋转、缩放、悬停显示信息、选择数据点

        Returns
        -------
        plotly.graph_objects.Figure
        """
        # 数据采样策略：数据量超过 200k 时进行分层采样
        SAMPLE_THRESHOLD = 200000
        total_rows = len(self.cluster_data)
        
        if total_rows > SAMPLE_THRESHOLD:
            # 按分群比例分层采样
            sample_ratio = SAMPLE_THRESHOLD / total_rows
            sampled_data = self.cluster_data.groupby('Customer_Label', group_keys=False).apply(
                lambda x: x.sample(frac=sample_ratio, random_state=42)
            )
            is_sampled = True
        else:
            sampled_data = self.cluster_data
            is_sampled = False
        
        unique_labels = sorted(sampled_data['Customer_Label'].unique())
        n_labels = len(unique_labels)
        
        # 生成足够数量的颜色（支持任意 k 值）
        if n_labels <= 10:
            colors = px.colors.qualitative.Plotly[:n_labels]
        else:
            # 超过10个分群时，使用 matplotlib 颜色映射生成
            cmap = plt.cm.get_cmap('tab20', n_labels)
            colors = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]})' 
                      for c in [cmap(i) for i in range(n_labels)]]
        
        fig = go.Figure()
        
        for idx, label in enumerate(unique_labels):
            subset = sampled_data[sampled_data['Customer_Label'] == label]
            
            # 准备悬停信息
            hover_text = [
                f"会员编号: {member_no}<br>"
                f"Recency: {r} 天<br>"
                f"Frequency: {f} 次<br>"
                f"Mileage: {m:,.0f} km<br>"
                f"分群: {label}"
                for member_no, r, f, m in zip(
                    subset['MEMBER_NO'], 
                    subset['Recency'], 
                    subset['Frequency'], 
                    subset['Mileage']
                )
            ]
            
            fig.add_trace(go.Scatter3d(
                x=subset['Recency'],
                y=subset['Frequency'],
                z=subset['Mileage'],
                mode='markers',
                name=label,
                marker=dict(
                    size=2.5,
                    color=colors[idx],
                    opacity=1,
                    line=dict(width=0.5, color='white'),
                ),
                text=hover_text,
                hoverinfo='text',
            ))
        
        # 标题：采样时显示采样信息
        if is_sampled:
            title_text = f'RFM 三维散点图（已采样 {len(sampled_data):,}/{total_rows:,}）'
        else:
            title_text = 'RFM 三维散点图'
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=16),
                x=0.02,
                xanchor='left',
            ),
            scene=dict(
                xaxis_title='Recency (最近乘机距今天数)',
                yaxis_title='Frequency (乘机次数)',
                zaxis_title='Mileage (总飞行里程)',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray'),
                zaxis=dict(gridcolor='lightgray'),
            ),
            legend=dict(
                title=dict(text='客户分群', font=dict(size=16)),
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01,
                font=dict(size=14),
                itemsizing='constant',
                itemwidth=50,
                bordercolor='gray',
                borderwidth=1,
                bgcolor='rgba(255,255,255,0.8)',
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=700,
            # 添加交互模式栏
            modebar=dict(
                bgcolor='rgba(220,220,220,0.8)',
            ),
        )
        
        return fig

    # ------------------------------------------------------------------
    # 分组柱状图：各群体 RFM 均值（双Y轴）
    # ------------------------------------------------------------------
    def rfm_bar_chart(self, figsize=(12, 6)):
        """
        各群体RFM指标分组柱状图

        - 按分群计算 R/F/M 均值
        - Mileage 缩放 1:100 以保证各柱子可比
        - Frequency 使用右侧次坐标轴
        - 添加数值标签

        Returns
        -------
        matplotlib.figure.Figure
        """
        rfm_means = (
            self.cluster_data
            .groupby('Customer_Label')[['Recency', 'Frequency', 'Mileage']]
            .mean()
        )

        # Mileage 缩放 1:100
        rfm_means['Mileage'] = rfm_means['Mileage'] / 100

        labels = rfm_means.index.tolist()
        x = np.arange(len(labels))
        width = 0.25

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()  # 创建次坐标轴

        # 左轴：Recency 和 Mileage
        bars_r = ax1.bar(x - width, rfm_means['Recency'], width, label='Recency (天)', color='#1f77b4')
        bars_m = ax1.bar(x + width, rfm_means['Mileage'], width, label='Mileage (×100 km)', color='#2ca02c')
        
        # 右轴：Frequency
        bars_f = ax2.bar(x, rfm_means['Frequency'], width, label='Frequency (次)', color='#ff7f0e', alpha=0.8)

        # 数值标签 - 左轴
        for bars in [bars_r, bars_m]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(
                    f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
        
        # 数值标签 - 右轴
        for bar in bars_f:
            height = bar.get_height()
            ax2.annotate(
                f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=11)
        ax1.set_ylabel('Recency / Mileage 均值', fontsize=12, color='#1f77b4')
        ax2.set_ylabel('Frequency 均值 (次)', fontsize=12, color='#ff7f0e')
        ax1.set_title('各分群 RFM 指标均值对比', fontsize=14, fontweight='bold')
        
        # 合并图例 - 移动到图表外部底部横向排列
        bars_all = [bars_r, bars_f, bars_m]
        labels_all = [bar.get_label() for bar in [bars_r, bars_f, bars_m]]
        ax1.legend(bars_all, labels_all, fontsize=10, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
        
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 相关性热力图
    # ------------------------------------------------------------------
    def correlation_heatmap(self, figsize=(8, 6)):
        """
        RFM相关性热力图

        - 计算 R/F/M 相关系数矩阵
        - 使用 seaborn 绘制，coolwarm 配色
        - 显示数值

        Returns
        -------
        matplotlib.figure.Figure
        """
        corr = self.cluster_data[['Recency', 'Frequency', 'Mileage']].corr()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title('RFM 相关性热力图', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 箱线图：各分群RFM指标分布
    # ------------------------------------------------------------------
    def boxplot_chart(self, figsize=(12, 6)):
        """
        各分群RFM指标箱线图

        - 展示各分群的 Recency / Frequency / Mileage 分布
        - 显示中位数、四分位数和异常值

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics = ['Recency', 'Frequency', 'Mileage']
        titles = ['Recency (天)', 'Frequency (次)', 'Mileage (km)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            data_to_plot = []
            labels = []
            for label in sorted(self.cluster_data['Customer_Label'].unique()):
                subset = self.cluster_data[self.cluster_data['Customer_Label'] == label][metric]
                data_to_plot.append(subset.values)
                labels.append(label)
            
            bp = axes[idx].boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
        
        fig.suptitle('各分群 RFM 指标分布箱线图', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 雷达图：各分群特征对比
    # ------------------------------------------------------------------
    def radar_chart(self, figsize=(10, 8)):
        """
        各分群特征雷达图

        - 展示各分群在 R/F/M 三个维度上的标准化均值
        - 直观对比各分群特征差异

        Returns
        -------
        matplotlib.figure.Figure
        """
        # 计算各分群均值并标准化到 0-1
        rfm_means = (
            self.cluster_data
            .groupby('Customer_Label')[['Recency', 'Frequency', 'Mileage']]
            .mean()
        )
        
        # 标准化（Min-Max）
        rfm_normalized = (rfm_means - rfm_means.min()) / (rfm_means.max() - rfm_means.min())
        
        # 注意：Recency 越小越好，取反
        rfm_normalized['Recency'] = 1 - rfm_normalized['Recency']
        
        labels = ['Recency\n(近期活跃)', 'Frequency\n(消费频率)', 'Mileage\n(飞行里程)']
        num_vars = len(labels)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(rfm_normalized)))
        
        for idx, (label, row) in enumerate(rfm_normalized.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('各分群特征雷达图', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 小提琴图：各分群数据分布
    # ------------------------------------------------------------------
    def violin_chart(self, figsize=(12, 6)):
        """
        各分群RFM指标小提琴图

        - 展示各分群数据的分布密度
        - 比箱线图更详细地展示数据分布形态

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics = ['Recency', 'Frequency', 'Mileage']
        titles = ['Recency (天)', 'Frequency (次)', 'Mileage (km)']
        
        unique_labels = sorted(self.cluster_data['Customer_Label'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            parts = axes[idx].violinplot(
                [self.cluster_data[self.cluster_data['Customer_Label'] == label][metric].values 
                 for label in unique_labels],
                positions=range(len(unique_labels)),
                showmeans=True,
                showmedians=True,
            )
            
            # 设置颜色
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            axes[idx].set_xticks(range(len(unique_labels)))
            axes[idx].set_xticklabels(unique_labels, rotation=45, fontsize=9)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        fig.suptitle('各分群 RFM 指标分布小提琴图', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 散点矩阵：变量间关系
    # ------------------------------------------------------------------
    def scatter_matrix_chart(self, figsize=(12, 10)):
        """
        RFM散点矩阵图

        - 展示 Recency / Frequency / Mileage 之间的两两关系
        - 对角线显示各变量的分布直方图
        - 不同颜色代表不同分群

        Returns
        -------
        matplotlib.figure.Figure
        """
        unique_labels = sorted(self.cluster_data['Customer_Label'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        metrics = ['Recency', 'Frequency', 'Mileage']
        
        for i, metric_i in enumerate(metrics):
            for j, metric_j in enumerate(metrics):
                ax = axes[i][j]
                
                if i == j:
                    # 对角线：直方图
                    for label, color in zip(unique_labels, colors):
                        subset = self.cluster_data[self.cluster_data['Customer_Label'] == label]
                        ax.hist(subset[metric_i], bins=20, alpha=0.5, label=label, color=color)
                    ax.set_ylabel('频数')
                else:
                    # 非对角线：散点图
                    for label, color in zip(unique_labels, colors):
                        subset = self.cluster_data[self.cluster_data['Customer_Label'] == label]
                        ax.scatter(subset[metric_j], subset[metric_i], 
                                  alpha=0.5, s=10, label=label, color=color)
                
                # 设置标签
                if i == 2:
                    ax.set_xlabel(metric_j)
                if j == 0:
                    ax.set_ylabel(metric_i)
                
                ax.grid(True, alpha=0.3)
        
        # 添加图例
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color, markersize=8, label=label) 
                   for label, color in zip(unique_labels, colors)]
        fig.legend(handles, unique_labels, loc='upper right', fontsize=9)
        
        fig.suptitle('RFM 散点矩阵图', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 保存图表
    # ------------------------------------------------------------------
    def save_figure(self, fig, filename, fmt='png'):
        """
        保存图表为文件

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        filename : str
            保存路径
        fmt : str
            图片格式，默认 'png'
        """
        fig.savefig(filename, format=fmt, dpi=150, bbox_inches='tight')
