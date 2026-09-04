"""客户分析与智能查询 - Streamlit 主界面"""

import os
import time
import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules import DataCleaner, RFMAnalyzer, ClusterAnalyzer, DataVisualizer, NL2SQLQueryEngine, DatabaseManager, SmartAssistant, ReportGenerator
from config.settings import DEFAULT_DATA_FILE, DEFAULT_LLM_CONFIG, CLEANING_DEFAULTS, CLUSTERING_DEFAULTS, MYSQL_CONFIG, CHAT_TEMPLATES, QUERY_DB_EXAMPLES, REPORT_TYPES, REPORT_DETAIL_LEVELS

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="客户分析与智能查询系统",
    page_icon="✈️",
    layout="wide",
)

# ============================================================
# session_state 初始化
# ============================================================
STATE_KEYS = {
    'cleaner': None,          # DataCleaner 实例
    'clean_data': None,       # 清洗后 DataFrame
    'rfm_df': None,           # RFM DataFrame
    'rfm_summary': None,      # RFM 摘要
    'cluster': None,          # ClusterAnalyzer 实例
    'cluster_result': None,   # 聚类结果 DataFrame（含可视化所需列名）
    'cluster_summary': None,  # 聚类汇总
    'recommended_k': None,    # 肘部法推荐 K
    'sse_list': None,         # SSE 列表
    'nl2sql_engine': None,    # NL2SQLQueryEngine 实例
}
for k, v in STATE_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 侧边栏导航
# ============================================================
st.sidebar.markdown(
    """
    <style>
    /* 隐藏本元素自身，不占垂直空间 */
    .element-container:has(> style) { display: none; }
    /* 减小侧边栏顶部内边距 */
    section[data-testid="stSidebar"] > div { padding-top: 0.5rem; }
    /* 减小侧边栏所有标题的上边距 */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 0;
        padding-top: 0;
    }
    /* 缩小侧边栏内 markdown 分隔线上方间距 */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    /* 缩小侧边栏 radio 组件的上下间距 */
    section[data-testid="stSidebar"] [data-testid="stRadio"] {
        padding-top: 0;
        padding-bottom: 0;
        margin-top: 0;
        margin-bottom: 0;
    }
    /* 调大侧边栏宽度 */
    section[data-testid="stSidebar"] {
        min-width: 24rem;
        max-width: 24rem;
    }
    /* 隐藏侧边栏收起按钮，禁止收起侧边栏 */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] * {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("航空客户智能分析与查询平台")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航",
    ["首页", "数据加载与清洗", "数据分析", "数据可视化", "数据库管理", "智能查询", "智能客服", "智能报告"],
)

st.sidebar.markdown("---")



# ============================================================
# 工具函数
# ============================================================
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8-sig')


def ensure_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 列类型与 Streamlit Arrow 序列化兼容。"""
    df = df.copy()
    for col in df.columns:
        # 处理 object 类型列
        if df[col].dtype == 'object':
            # 尝试转为数值类型
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isna().all() and converted.notna().sum() > 0:
                df[col] = converted
        # 处理 pandas nullable 整数类型（Int64, Int32 等）转为标准 numpy 类型
        elif hasattr(df[col].dtype, 'name') and 'Int' in str(df[col].dtype):
            df[col] = df[col].astype('float64')
    return df


def build_visualizer_df(result_df: pd.DataFrame) -> pd.DataFrame:
    """将 ClusterAnalyzer 输出的列名转换为 DataVisualizer 期望的列名。"""
    df = result_df.rename(columns={
        'R': 'Recency',
        'F': 'Frequency',
        'M': 'Mileage',
        '客户分群': 'Customer_Label',
    })
    return df


def paginated_data_preview(df: pd.DataFrame, key_prefix: str,
                           search_placeholder: str = "输入关键词搜索...",
                           filter_column: str = None,
                           search_columns: list = None,
                           default_rows: int = 30):
    """
    通用分页数据预览组件：定向列搜索 + 可选列筛选 + 分页 + 显示数量。

    Parameters
    ----------
    df : pd.DataFrame
        待展示的数据。
    key_prefix : str
        Streamlit widget key 前缀，保证唯一性。
    search_placeholder : str
        搜索框占位文字。
    filter_column : str, optional
        可选的筛选列名（通常是分群标签列），为 None 则不显示筛选。
    search_columns : list, optional
        限定搜索的列名列表。为 None 时搜索所有非浮点数列。
    default_rows : int
        默认每页显示行数。
    """
    if df.empty:
        st.info("暂无数据。")
        return

    # ---------- 确定搜索列 ----------
    if search_columns is not None:
        cols_for_search = [c for c in search_columns if c in df.columns]
    else:
        # 默认排除浮点数列，避免小数位产生误匹配
        cols_for_search = [c for c in df.columns
                           if not pd.api.types.is_float_dtype(df[c])]

    # ---------- 控制栏 ----------
    ctrl_cols = st.columns([3, 2, 1, 1])

    with ctrl_cols[0]:
        search_query = st.text_input(
            "搜索", placeholder=search_placeholder,
            key=f"{key_prefix}_search",
        )
    with ctrl_cols[1]:
        if filter_column and filter_column in df.columns:
            options = ['全部'] + sorted(df[filter_column].dropna().unique().tolist())
            selected_filter = st.selectbox(
                f"按 {filter_column} 筛选", options,
                key=f"{key_prefix}_filter",
            )
        else:
            selected_filter = '全部'
    with ctrl_cols[2]:
        max_rows = max(min(len(df), 10000), 1)
        show_rows = st.number_input(
            "每页条数", min_value=1, max_value=max_rows,
            value=min(default_rows, max_rows), step=10,
            key=f"{key_prefix}_rows",
        )
    with ctrl_cols[3]:
        st.write("")  # 占位对齐
        st.write("")

    # ---------- 筛选 + 搜索 ----------
    filtered = df.copy()
    if selected_filter != '全部' and filter_column:
        filtered = filtered[filtered[filter_column] == selected_filter]

    if search_query.strip():
        keyword = search_query.strip()
        # 优化：用向量化 str.contains 逐列匹配，any(axis=1) 合并
        # 比 .astype(str) + .apply() 快 5-10 倍，且内存峰值低
        mask = None
        for col in cols_for_search:
            col_series = filtered[col].astype(str)
            col_mask = col_series.str.contains(keyword, case=False, na=False)
            if mask is None:
                mask = col_mask.values
            else:
                mask |= col_mask.values
        if mask is not None:
            filtered = filtered[mask]

    total = len(filtered)
    total_pages = max(1, (total + show_rows - 1) // show_rows)

    # ---------- 分页控制 ----------
    pg1, pg2, pg3 = st.columns([1, 2, 1])
    with pg2:
        current_page = st.number_input(
            "页码", min_value=1, max_value=total_pages, value=1, step=1,
            key=f"{key_prefix}_page",
        )

    start = (current_page - 1) * show_rows
    end = min(start + show_rows, total)
    page_df = filtered.iloc[start:end]

    st.caption(f"共 **{total}** 条结果 ｜ 第 **{current_page}** / **{total_pages}** 页 ｜ 当前显示第 {start + 1}–{end} 条")
    st.dataframe(ensure_arrow_compatible(page_df), width='stretch')


def display_label_counts(label_counts, key_prefix="default"):
    """
    显示客户分群统计，支持按级别或人数排序，可选择升序或降序。

    Parameters
    ----------
    label_counts : pandas.Series
        各标签的计数。
    key_prefix : str
        Streamlit 组件的 key 前缀，避免重复。
    """
    # 排序选择器
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.radio(
            "排序依据",
            ["按级别", "按人数"],
            horizontal=True,
            key=f"{key_prefix}_sort_by"
        )
    with col2:
        sort_order = st.radio(
            "排序顺序",
            ["升序", "降序"],
            horizontal=True,
            key=f"{key_prefix}_sort_order"
        )

    ascending = sort_order == "升序"

    if sort_by == "按人数":
        label_counts = label_counts.sort_values(ascending=ascending)
    else:
        # 按级别排序：使用预定义的标签顺序
        label_order = [
            '休眠客户', '流失客户', '低价值客户', '新客户',
            '一般保持客户', '一般发展客户', '潜力客户',
            '重要发展客户', '重要保持客户', '高价值客户',
            '一般客户', '重要客户'
        ]
        # 按照标签顺序排序，未定义的标签放在最后
        sorted_labels = []
        for label in label_order:
            if label in label_counts.index:
                sorted_labels.append(label)
        # 添加未在预定义顺序中的标签
        for label in label_counts.index:
            if label not in sorted_labels:
                sorted_labels.append(label)
        if not ascending:
            sorted_labels = sorted_labels[::-1]
        label_counts = label_counts.reindex(sorted_labels)

    # 显示统计
    n_labels = len(label_counts)
    if n_labels >= 6:
        per_row = 5
        items = list(label_counts.items())
        for i in range(0, len(items), per_row):
            chunk = items[i:i + per_row]
            cols = st.columns(per_row)
            for col, (label, cnt) in zip(cols, chunk):
                col.metric(label, f"{cnt:,} 人")
    else:
        count_cols = st.columns(n_labels)
        for col, (label, cnt) in zip(count_cols, label_counts.items()):
            col.metric(label, f"{cnt:,} 人")


# ============================================================
# 辅助函数
# ============================================================
def _sanitize_mcp_data(data):
    """将 MCP 查询结果数据转换为可安全存储在 session_state 中的原生 Python 类型。

    处理 numpy 类型、Decimal、datetime 等不可序列化的类型。
    """
    import json
    import decimal
    import numpy as np

    def _default_encoder(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    if data is None:
        return None
    try:
        return json.loads(json.dumps(data, default=_default_encoder))
    except (TypeError, ValueError):
        return data


# ============================================================
# 首页
# ============================================================
def page_home():
    st.title("客户分析与智能查询平台")
    st.markdown("---")

    st.markdown("""
    本平台基于航空客户数据，集成 **数据清洗**、**RFM 模型分析**、**K-Means 聚类**、
    **数据可视化** 和 **自然语言智能查询** 五大功能模块，帮助您深入理解客户行为，
    精准制定营销策略。
    """)
    st.markdown("---")

    # 数据集概览（如果已加载）
    if st.session_state['clean_data'] is not None:
        df = st.session_state['clean_data']
        st.subheader("当前数据集概览")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("记录数", f"{df.shape[0]:,}")
        m2.metric("特征数", f"{df.shape[1]}")
        m3.metric("缺失值", f"{df.isnull().sum().sum():,}")
        m4.metric("数值列", f"{df.select_dtypes(include='number').shape[1]}")
        paginated_data_preview(df, key_prefix="home_preview", default_rows=20)
    else:
        st.info("尚未完成数据清洗，请前往 **数据加载与清洗** 页面加载并清洗数据。")


# ============================================================
# 数据清洗页面
# ============================================================
def page_cleaning():
    st.title("数据加载与清洗")

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("上传数据")
        uploaded = st.file_uploader("上传 CSV 文件", type=['csv'])
        use_default = st.checkbox("使用默认数据 (air.csv)", value=True)
        
        st.markdown("---")

        st.header("清洗参数")

        missing_strategy = st.selectbox(
            "缺失值策略",
            ['drop', 'mean', 'median', 'mode', 'do nothing'],
            index=['drop', 'mean', 'median', 'mode', 'do nothing'].index(CLEANING_DEFAULTS['missing_strategy']),
        )
        outlier_method = st.selectbox(
            "异常值检测方法",
            ['iqr', 'zscore', 'do nothing'],
            index=['iqr', 'zscore', 'do nothing'].index(CLEANING_DEFAULTS['outlier_method']),
        )
        # 当检测方法为 do nothing 时，处理策略自动设为 do nothing 并禁用
        outlier_disabled = outlier_method == 'do nothing'
        if outlier_disabled:
            outlier_strategy = 'do nothing'
            st.selectbox(
                "异常值处理策略",
                ['do nothing'],
                index=0,
                disabled=True,
                help="检测方法为 do nothing 时，处理策略自动设为 do nothing",
            )
        else:
            outlier_strategy = st.selectbox(
                "异常值处理策略",
                ['remove', 'clip', 'do nothing'],
                index=['remove', 'clip', 'do nothing'].index(CLEANING_DEFAULTS['outlier_strategy']),
            )

    # ---------- 主区域 ----------
    # 1. 加载数据
    file_path = None
    file_bytes = None
    pending_upload_name = None  # 待加载的上传文件名（尚未点击加载）

    # 检查是否有新上传的文件
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        pending_upload_name = uploaded.name
        # 注意：此处不修改 session_state 中的"已加载"状态
        # 仅当用户点击"加载数据"并成功加载后，才更新 loaded_data_source
        loaded_src = st.session_state.get('loaded_data_source')
        loaded_name = st.session_state.get('loaded_data_source_name')
        if not (loaded_src == 'uploaded' and loaded_name == uploaded.name):
            # 当前上传文件尚未加载，提示用户点击加载
            st.info(
                f"已上传文件: **{uploaded.name}** ({len(file_bytes) / 1024:.1f} KB)"
                f" — 请点击下方「加载数据」按钮加载"
            )
    elif use_default:
        if os.path.exists(DEFAULT_DATA_FILE):
            file_path = DEFAULT_DATA_FILE
        else:
            st.error(f"默认数据文件不存在: {DEFAULT_DATA_FILE}")
            return
    else:
        st.warning("请上传 CSV 文件或勾选使用默认数据。")
        return

    # 加载按钮
    if st.button("加载数据", width='stretch'):
        try:
            with st.spinner("正在加载数据..."):
                cleaner = DataCleaner()
                if file_bytes is not None:
                    # 将上传内容写入临时文件
                    tmp_path = os.path.join(os.path.dirname(__file__), '_tmp_upload.csv')
                    with open(tmp_path, 'wb') as f:
                        f.write(file_bytes)
                    cleaner.load_data(tmp_path)
                    os.remove(tmp_path)
                    # 加载成功后，记录数据来源为"已上传"
                    st.session_state['loaded_data_source'] = 'uploaded'
                    st.session_state['loaded_data_source_name'] = pending_upload_name
                else:
                    cleaner.load_data(file_path)
                    # 记录数据来源为"默认"
                    st.session_state['loaded_data_source'] = 'default'
                    st.session_state['loaded_data_source_name'] = DEFAULT_DATA_FILE
                # 立即转换数据类型，确保概览显示正确的类型
                cleaner.convert_dtypes()
                st.session_state['cleaner'] = cleaner
                # 重置清洗状态，因为重新加载了原始数据
                st.session_state['data_cleaned'] = None
                st.session_state['clean_data'] = None
                # 使 CSV 导出缓存失效（数据已变更）
                st.session_state['clean_data_csv_cache'] = None
            st.success("数据加载成功！")
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            return

    cleaner = st.session_state['cleaner']

    # 显示当前已加载数据的来源（仅当真正加载过数据后才显示）
    if cleaner is not None:
        src = st.session_state.get('loaded_data_source', 'default')
        src_name = st.session_state.get('loaded_data_source_name', DEFAULT_DATA_FILE)
        if src == 'uploaded':
            st.info(f"当前已加载数据来源: **{src_name}** (已上传)")
        else:
            st.info(f"当前已加载数据来源: **{src_name}** (默认)")
    else:
        st.info("请点击上方按钮加载数据。")
        return

    # 2. 数据概览（始终显示原始导入文件的信息）
    st.subheader("数据概览（原始数据）")
    try:
        # 使用原始数据计算概览，而不是清洗后的数据
        raw_data = cleaner.raw_data if cleaner.raw_data is not None else cleaner.data
        
        c1, c2 = st.columns(2)
        c1.metric("行数", f"{raw_data.shape[0]:,}")
        c2.metric("列数", f"{raw_data.shape[1]}")
    except Exception as e:
        st.error(f"获取数据概览失败: {e}")

    st.markdown("---")

    # 3. 执行清洗
    all_do_nothing = (missing_strategy == 'do nothing' and 
                      outlier_method == 'do nothing' and 
                      outlier_strategy == 'do nothing')
    
    if st.button("执行清洗", width='stretch'):
        try:
            # 每次清洗时从原始数据开始，避免重复处理
            if cleaner.raw_data is not None:
                cleaner.data = cleaner.raw_data.copy()
                cleaner.outlier_mask = None

            if all_do_nothing:
                # 不清洗，直接使用原始数据
                st.session_state['clean_data'] = cleaner.data.copy()
                st.session_state['data_cleaned'] = False
                # 数据已变更，使 CSV 导出缓存失效
                st.session_state['clean_data_csv_cache'] = None
                st.info("数据未进行清洗，将使用原始数据。")
            else:
                # 记录清洗前的缺失值和异常值数量
                before_missing = cleaner.data.isnull().sum().sum()
                before_rows = len(cleaner.data)

                with st.spinner("正在执行数据清洗流水线..."):
                    cleaner.run_pipeline(
                        missing_strategy=missing_strategy,
                        outlier_method=outlier_method,
                        outlier_strategy=outlier_strategy,
                    )
                    clean_data = cleaner.get_clean_data()
                    st.session_state['clean_data'] = clean_data
                    st.session_state['data_cleaned'] = True
                    # 清洗后需重新分析，清除下游缓存
                    st.session_state['rfm_df'] = None
                    st.session_state['cluster_result'] = None
                    # 数据已变更，使 CSV 导出缓存失效
                    st.session_state['clean_data_csv_cache'] = None

                # 显示清洗效果
                after_missing = clean_data.isnull().sum().sum()
                after_rows = len(clean_data)
                col1, col2, col3 = st.columns(3)
                col1.metric("行数变化", f"{before_rows} → {after_rows}",
                            delta=f"{after_rows - before_rows}",delta_color="inverse")
                col2.metric("缺失值", f"{before_missing} → {after_missing}",
                            delta=f"{after_missing - before_missing}",delta_color="inverse")
                col3.metric("最终数据", f"{clean_data.shape[0]} 行 × {clean_data.shape[1]} 列")
                st.success("清洗完成！")
        except Exception as e:
            st.error(f"数据清洗失败: {e}")

    # 4. 数据质量报告
    st.subheader("数据质量报告")
    
    # 显示数据是否经过清洗的提示
    if st.session_state.get('data_cleaned') is False:
        st.warning("**数据未进行清洗** - 以下报告基于原始数据。")
    elif st.session_state.get('data_cleaned') is True:
        st.success("✓ **数据已清洗** - 以下报告基于清洗后的数据。")
    
    try:
        report = cleaner.generate_quality_report()
        
        # 计算缺失值总数
        total_missing = report['missing_info']['missing_count'].sum()
        # 计算异常值点总数
        total_outliers = sum(info['outlier_count'] for info in report['outlier_summary'].values())
        
        # 第一行：总行数、总列数、重复行数
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("总行数", f"{report['total_rows']:,}")
        rc2.metric("总列数", f"{report['total_columns']}")
        rc3.metric("重复行数", f"{report['duplicate_rows']:,}")
        
        # 第二行：缺失值、异常值点、异常值字段数
        rc4, rc5, rc6 = st.columns(3)
        rc4.metric("缺失值", f"{total_missing:,}")
        rc5.metric("异常值点", f"{total_outliers:,}")
        rc6.metric("异常值字段数", f"{len(report['outlier_summary'])}")

        with st.expander("缺失值详情"):
            st.dataframe(report['missing_info'], width='stretch')
        with st.expander("数值列统计"):
            st.dataframe(report['numeric_stats'], width='stretch')
        with st.expander("异常值摘要"):
            if report['outlier_summary']:
                outlier_df = pd.DataFrame(report['outlier_summary']).T
                st.dataframe(outlier_df, width='stretch')
            else:
                st.info("未检测到异常值。")
    except Exception as e:
        st.error(f"生成质量报告失败: {e}")

    # 5. 下载清洗数据
    # 优化：仅在用户主动点击"生成下载文件"时才执行 CSV 转换，
    # 避免每次页面渲染都对大 DataFrame 执行 to_csv 造成卡顿
    if st.session_state['clean_data'] is not None:
        st.markdown("---")
        clean_data = st.session_state['clean_data']
        # 用 id + 行数 作为缓存键，检测 clean_data 是否变化
        cache_key = (id(clean_data), len(clean_data))
        csv_cache = st.session_state.get('clean_data_csv_cache')

        if csv_cache is None or csv_cache.get('key') != cache_key:
            # 缓存不存在或已失效：显示"生成下载文件"按钮
            if st.button(
                "生成下载文件 (CSV)",
                width='stretch',
                help="点击后将清洗后数据转换为 CSV 格式。大数量数据转换可能需要数秒。",
            ):
                csv_bytes = convert_df_to_csv(clean_data)
                st.session_state['clean_data_csv_cache'] = {
                    'key': cache_key,
                    'bytes': csv_bytes,
                    'rows': len(clean_data),
                }
                st.rerun()
        else:
            # 缓存有效：直接显示下载按钮
            st.download_button(
                label=f"下载清洗后数据 (CSV, {csv_cache['rows']:,} 行)",
                data=csv_cache['bytes'],
                file_name="clean_data.csv",
                mime="text/csv",
                width='stretch',
            )
            # 提供"重新生成"入口（数据未变但用户想重新生成）
            if st.button("重新生成下载文件", help="强制重新生成 CSV 文件"):
                st.session_state['clean_data_csv_cache'] = None
                st.rerun()


# ============================================================
# 数据分析页面
# ============================================================
def page_analysis():
    st.title("数据分析")

    clean_data = st.session_state['clean_data']
    if clean_data is None:
        st.warning("请先在 **数据加载与清洗** 页面加载并清洗数据。")
        return

    # ---------- 1. RFM 计算 ----------
    st.subheader("1. RFM 指标计算")
    
    # 基准日期设置
    reference_date = st.date_input(
        "基准日期 (Recency)",
        value=None,
        help="计算 Recency 的参考日期。留空则使用数据集中的最晚日期。"
    )
    
    if st.session_state['rfm_df'] is None:
        if st.button("计算 RFM 指标", width='stretch'):
            try:
                with st.spinner("正在计算 RFM 指标..."):
                    ref = str(reference_date) if reference_date else None
                    rfm = RFMAnalyzer(clean_data, reference_date=ref)
                    rfm_df = rfm.calculate_rfm()
                    summary = rfm.get_rfm_summary()
                    st.session_state['rfm_df'] = rfm_df
                    st.session_state['rfm_summary'] = summary
                    # 清除下游缓存
                    st.session_state['cluster'] = None
                    st.session_state['cluster_result'] = None
                st.rerun()
            except Exception as e:
                st.error(f"RFM 计算失败: {e}")
    else:
        st.success("RFM 指标已计算。")

    rfm_df = st.session_state['rfm_df']
    if rfm_df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("平均 Recency (天)", f"{rfm_df['R'].mean():.1f}")
        c2.metric("平均 Frequency (次)", f"{rfm_df['F'].mean():.1f}")
        c3.metric("平均 Mileage (km)", f"{rfm_df['M'].mean():,.0f}")

        with st.expander("RFM 统计摘要"):
            st.dataframe(st.session_state['rfm_summary'], width='stretch')
        with st.expander("RFM 数据预览"):
            paginated_data_preview(rfm_df, key_prefix="rfm_preview", search_placeholder="搜索会员编号等...")

    st.markdown("---")

    # ---------- 2. 聚类分析 ----------
    st.subheader("2. 聚类分析")

    # 聚类参数设置
    with st.expander("聚类参数设置", expanded=True):
        col_params = st.columns(3)
        with col_params[0]:
            random_state = st.number_input(
                "随机种子 (random_state)",
                min_value=0,
                value=CLUSTERING_DEFAULTS['random_state'],
                help="设置随机种子以确保结果可复现。设置为 0 则每次运行结果可能不同。"
            )
            # 如果用户输入0，将其转换为None
            if random_state == 0:
                random_state = None
        with col_params[1]:
            n_init = st.number_input(
                "初始化次数 (n_init)",
                min_value=1,
                max_value=100,
                value=CLUSTERING_DEFAULTS['n_init'],
                help="算法运行次数，每次用不同初始化，选择最优结果。值越大越稳定但越慢。"
            )
        with col_params[2]:
            k_for_fit = st.number_input(
                "聚类数量 (k)",
                min_value=2,
                max_value=10,
                value=CLUSTERING_DEFAULTS['k'],
                help="要将数据分成的簇数。"
            )

    col_elbow, col_fit = st.columns(2)

    elbow_clicked = False
    fit_clicked = False

    # 肘部法
    with col_elbow:
        elbow_clicked = st.button("肘部法分析", width='stretch', disabled=rfm_df is None)

    # 执行聚类
    with col_fit:
        fit_clicked = st.button("执行聚类", width='stretch', disabled=rfm_df is None)

    # 状态消息放在列外部，避免列内高度变化导致布局抖动
    elbow_status = st.empty()
    fit_status = st.empty()

    if elbow_clicked:
        try:
            elbow_status.markdown("**正在运行肘部法...**")
            cluster = ClusterAnalyzer(rfm_df)
            cluster.standardize()
            recommended_k, sse_list = cluster.elbow_method()
            st.session_state['cluster'] = cluster
            st.session_state['recommended_k'] = recommended_k
            st.session_state['sse_list'] = sse_list
            elbow_status.success(f"✓ 推荐聚类数 K = {recommended_k}")
        except Exception as e:
            elbow_status.error(f"✗ 肘部法分析失败: {e}")

    if fit_clicked:
        try:
            fit_status.markdown(f"**正在执行 K={k_for_fit} 聚类...**")
            cluster = st.session_state.get('cluster') or ClusterAnalyzer(rfm_df)
            if cluster.rfm_scaled is None:
                cluster.standardize()
            cluster.fit(k=k_for_fit, random_state=random_state, n_init=n_init)
            label_mapping = cluster.assign_labels()
            result_df = cluster.get_result_dataframe()
            viz_df = build_visualizer_df(result_df)
            summary = cluster.get_cluster_summary()

            st.session_state['cluster'] = cluster
            st.session_state['cluster_result'] = viz_df
            st.session_state['cluster_summary'] = summary
            # 格式化 label_mapping，将 np.int64 转换为普通整数
            formatted_mapping = {int(k): v for k, v in label_mapping.items()}
            fit_status.success(f"✓ 聚类完成！K={k_for_fit}，分群映射: {formatted_mapping}")
        except Exception as e:
            fit_status.error(f"✗ 聚类分析失败: {e}")

    # 肘部法图表放在列外部
    if st.session_state['sse_list'] is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        k_range = list(range(2, 2 + len(st.session_state['sse_list'])))
        ax.plot(k_range, st.session_state['sse_list'], 'bo-', linewidth=2)
        ax.axvline(
            x=st.session_state['recommended_k'], color='r',
            linestyle='--', label=f"推荐 K={st.session_state['recommended_k']}"
        )
        ax.set_xlabel('聚类数 K')
        ax.set_ylabel('SSE (误差平方和)')
        ax.set_title('肘部法确定最佳 K 值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ---------- 3. 聚类结果展示 ----------
    st.subheader("3. 聚类结果")

    if st.session_state['cluster_result'] is not None:
        viz_df = st.session_state['cluster_result']

        # 分群统计
        label_counts = viz_df['Customer_Label'].value_counts()
        st.markdown("**客户分群统计:**")
        display_label_counts(label_counts, key_prefix="analysis")

        if st.session_state['cluster_summary'] is not None:
            with st.expander("聚类中心汇总"):
                st.dataframe(st.session_state['cluster_summary'], width='stretch')

        with st.expander("聚类结果数据预览"):
            paginated_data_preview(
                viz_df, key_prefix="cluster_preview",
                search_placeholder="搜索会员编号等...",
                filter_column='Customer_Label',
            )
    else:
        st.info("请先执行 RFM 计算和聚类分析。")


# ============================================================
# 数据可视化页面
# ============================================================
def page_visualization():
    st.title("数据可视化")

    viz_df = st.session_state['cluster_result']
    if viz_df is None:
        st.warning("请先在 **数据分析** 页面完成聚类分析。")
        return

    # ---------- 侧边栏 ----------
    chart_options = {
        "客户分群饼图": "pie",
        "RFM 三维散点图": "scatter3d",
        "RFM 分组柱状图": "bar",
        "RFM 相关性热力图": "heatmap",
        "各分群箱线图": "boxplot",
        "各分群特征雷达图": "radar",
        "各分群小提琴图": "violin",
        "RFM 散点矩阵图": "scatter_matrix",
    }
    with st.sidebar:
        st.header("图表设置")
        selected = st.multiselect(
            "选择要展示的图表",
            list(chart_options.keys()),
            default=list(chart_options.keys()),
        )

    if not selected:
        st.info("请在侧边栏选择至少一种图表。")
        return

    try:
        viz = DataVisualizer(viz_df)
    except Exception as e:
        st.error(f"初始化可视化失败: {e}")
        return

    chart_func_map = {
        "pie": viz.pie_chart,
        "scatter3d": viz.scatter_3d,
        "bar": viz.rfm_bar_chart,
        "heatmap": viz.correlation_heatmap,
        "boxplot": viz.boxplot_chart,
        "radar": viz.radar_chart,
        "violin": viz.violin_chart,
        "scatter_matrix": viz.scatter_matrix_chart,
    }

    for name in selected:
        key = chart_options[name]
        st.subheader(name)
        try:
            with st.spinner(f"正在生成 {name} ..."):
                fig = chart_func_map[key]()
            
            # 根据图表类型使用不同的显示方法
            if key == "scatter3d":
                # plotly 交互式图表
                st.plotly_chart(fig, use_container_width=True)
            else:
                # matplotlib 静态图表
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
        except Exception as e:
            st.error(f"生成 {name} 失败: {e}")


# ============================================================
# 数据库管理页面
# ============================================================
def page_database():
    st.title("数据库管理")

    # ---------- session_state 初始化 ----------
    if 'db_manager' not in st.session_state:
        st.session_state['db_manager'] = None
    if 'db_connected' not in st.session_state:
        st.session_state['db_connected'] = False

    # ---------- 自动连接数据库（使用 .env 配置） ----------
    if not st.session_state['db_connected']:
        try:
            db_manager = DatabaseManager(**MYSQL_CONFIG)
            db_manager.connect()
            db_manager.create_tables()
            st.session_state['db_manager'] = db_manager
            st.session_state['db_connected'] = True
        except Exception as e:
            st.error(f"数据库连接失败: {e}　|　请检查 .env 文件中的 MySQL 配置。")
            st.session_state['db_connected'] = False
            # 提供手动重连按钮
            if st.button("重新连接数据库", type="primary"):
                st.session_state['db_connected'] = False
                st.rerun()
            return

    # ---------- 连接健康检查与自动恢复 ----------
    db_manager = st.session_state.get('db_manager')
    if db_manager is not None:
        # 主动检测连接状态，失效时自动尝试重连
        if not db_manager.ensure_connection():
            st.session_state['db_connected'] = False
            st.error("数据库连接已失效，且自动重连失败。请检查 MySQL 服务是否运行。")
            if st.button("手动重连", type="primary"):
                try:
                    db_manager.reconnect()
                    db_manager.create_tables()
                    st.session_state['db_connected'] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"重连失败: {e}")
            return
        else:
            st.session_state['db_connected'] = True

    # ---------- 侧边栏：连接状态 ----------
    with st.sidebar:
        st.header("数据库连接状态")
        if db_manager is not None and db_manager.is_connected():
            st.success(f"已连接: {MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")
            if st.button("重新连接", help="强制重新建立数据库连接"):
                try:
                    db_manager.reconnect()
                    db_manager.create_tables()
                    st.success("重连成功！")
                except Exception as e:
                    st.error(f"重连失败: {e}")
                    st.session_state['db_connected'] = False
                    st.rerun()
        else:
            st.error("数据库未连接")

    # ---------- 主区域 ----------
    if not st.session_state['db_connected']:
        st.info("数据库未连接。")
        return

    db_manager = st.session_state['db_manager']

    # 数据库统计信息
    st.subheader("数据库概览")
    try:
        stats = db_manager.get_table_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("分析记录", f"{stats.get('customer_clusters', 0):,}")
        col2.metric("原始数据", f"{stats.get('member_data', 0):,}")
        col3.metric("批次数量", f"{stats.get('batch_count', 0)}")
    except Exception as e:
        st.error(f"获取统计信息失败: {e}")

    st.markdown("---")

    # 功能选项卡
    tab1, tab2, tab3 = st.tabs(["聚类结果入库", "数据查询", "数据管理"])

    # ---------- Tab 1: 聚类结果入库 ----------
    with tab1:
        # 显示入库成功消息（如果有）
        if 'import_success_msg' in st.session_state:
            st.success(st.session_state['import_success_msg'])
            del st.session_state['import_success_msg']
        
        st.subheader("聚类结果入库")
        
        cluster_result = st.session_state.get('cluster_result')
        rfm_df = st.session_state.get('rfm_df')
        
        if cluster_result is None:
            st.warning("请先在 **数据分析** 页面完成聚类分析。")
        else:
            # 显示聚类结果预览
            viz_df = cluster_result
            label_counts = viz_df['Customer_Label'].value_counts()
            
            st.markdown("**当前聚类结果:**")
            display_label_counts(label_counts, key_prefix="db_import")
            
            # 批次编号
            batch_no = st.text_input(
                "批次编号（不超过30个字符）",
                value=f"CLUSTER_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="用于标识本次入库的批次，同一批次数据可重复导入（会更新已有记录）",
                max_chars=30,
            )
            
            # 选择要入库的类别
            st.markdown("**选择要入库的客户类别:**")
            
            all_types = sorted(label_counts.index.tolist())
            select_all = st.checkbox("全选", value=True)
            
            if select_all:
                selected_types = all_types
            else:
                selected_types = st.multiselect(
                    "选择客户类别",
                    all_types,
                    default=all_types[:1] if all_types else [],
                )
            
            if selected_types:
                # 过滤选中的数据
                filtered_df = viz_df[viz_df['Customer_Label'].isin(selected_types)]
                
                st.info(f"已选择 **{len(selected_types)}** 个类别，共 **{len(filtered_df):,}** 条记录。")
                
                # 入库选项
                analysis_date = st.date_input("分析日期", value=datetime.date.today())
                
                if st.button("执行入库", width='stretch', type="primary"):
                    try:
                        # 进度展示用 placeholder
                        progress_ph = st.empty()
                        progress_bar = st.progress(0, text="准备写入...")

                        def _progress(done, total, label):
                            if total > 0:
                                pct = min(done / total, 1.0)
                                progress_bar.progress(
                                    pct,
                                    text=f"{label}：{done:,} / {total:,}（{pct*100:.1f}%）"
                                )

                        # 准备聚类数据（需要将列名转回中文）
                        insert_df = filtered_df.rename(columns={
                            'Customer_Label': '客户分群'
                        })

                        # 批量插入客户分群数据
                        progress_bar.progress(0, text="正在写入客户分群数据...")
                        cluster_count = db_manager.batch_insert_customer_clusters(
                            insert_df, batch_no, cluster_date=analysis_date,
                            progress_callback=lambda d, t: _progress(d, t, "客户分群"),
                        )

                        # 批量插入RFM分析数据（默认同时入库）
                        rfm_count = 0
                        if rfm_df is not None:
                            # 只插入选中类别的RFM数据
                            selected_members = filtered_df['MEMBER_NO'].unique()
                            rfm_filtered = rfm_df[rfm_df['MEMBER_NO'].isin(selected_members)]
                            rfm_count = db_manager.batch_insert_rfm_analysis(
                                rfm_filtered, batch_no, analysis_date=analysis_date,
                                progress_callback=lambda d, t: _progress(d, t, "RFM 分析"),
                            )

                        # 批量插入原始客户数据（默认同时入库）
                        member_count = 0
                        clean_data = st.session_state.get('clean_data')
                        if clean_data is not None:
                            # 只插入选中类别的会员数据
                            selected_members = filtered_df['MEMBER_NO'].unique()
                            member_filtered = clean_data[clean_data['MEMBER_NO'].isin(selected_members)]
                            member_count = db_manager.batch_insert_member_data(
                                member_filtered, batch_no, import_date=analysis_date,
                                progress_callback=lambda d, t: _progress(d, t, "原始数据"),
                            )

                        progress_bar.empty()
                        progress_ph.empty()
                        # 保存成功消息到 session_state，刷新后显示
                        st.session_state['import_success_msg'] = f"入库成功！客户分群: {cluster_count} 条，RFM: {rfm_count} 条，原始数据: {member_count} 条，批次号: {batch_no}"
                        # 刷新页面以更新概览数据
                        st.rerun()
                    except Exception as e:
                        st.error(f"入库失败: {e}")
            else:
                st.warning("请至少选择一个客户类别。")

    # ---------- Tab 2: 数据查询 ----------
    with tab2:
        st.subheader("数据查询")
        
        query_type = st.selectbox(
            "查询类型",
            ["RFM分析结果", "客户分群结果", "原始客户数据"]
        )
        
        if query_type == "RFM分析结果":
            # 获取批次列表
            try:
                batch_list = db_manager.get_batch_list('rfm_analysis')
                if not batch_list:
                    st.info("暂无RFM分析数据。")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_batch = st.selectbox("选择批次（必选）", batch_list)
                    with col2:
                        member_no = st.text_input("会员编号搜索（留空查询全部）")
                    
                    limit = st.number_input("显示条数", min_value=10, max_value=10000, value=100)
                    
                    if st.button("查询", key="query_rfm"):
                        member_filter = member_no.strip() if member_no.strip() else None
                        result = db_manager.get_rfm_analysis(
                            batch_no=selected_batch, member_no=member_filter, limit=limit
                        )
                        st.dataframe(result, width='stretch')
                        st.caption(f"共 {len(result)} 条记录")
            except Exception as e:
                st.error(f"查询失败: {e}")
        
        elif query_type == "原始客户数据":
            try:
                batch_list = db_manager.get_batch_list('member_data')
                if not batch_list:
                    st.info("暂无原始客户数据。")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_batch = st.selectbox("选择批次（必选）", batch_list, key="member_data_batch")
                    with col2:
                        member_no = st.text_input("会员编号搜索（留空查询全部）", key="member_data_member")
                    
                    limit = st.number_input("显示条数", min_value=10, max_value=10000, value=100, key="member_data_limit")
                    
                    if st.button("查询", key="query_member_data"):
                        member_filter = member_no.strip() if member_no.strip() else None
                        result = db_manager.get_member_data(
                            batch_no=selected_batch, member_no=member_filter, limit=limit
                        )
                        st.dataframe(result, width='stretch')
                        st.caption(f"共 {len(result)} 条记录")
            except Exception as e:
                st.error(f"查询失败: {e}")
        elif query_type == "客户分群结果":
            try:
                batch_list = db_manager.get_batch_list('customer_clusters')
                if not batch_list:
                    st.info("暂无客户分群数据。")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        selected_batch = st.selectbox("选择批次（必选）", batch_list)
                    with col2:
                        type_filter = st.text_input("客户类型筛选（留空查询全部）")
                    with col3:
                        member_no = st.text_input("会员编号搜索（留空查询全部）")
                    
                    limit = st.number_input("显示条数", min_value=10, max_value=10000, value=100, key="cluster_limit")
                    
                    if st.button("查询", key="query_cluster"):
                        type_filter_val = type_filter.strip() if type_filter.strip() else None
                        member_filter = member_no.strip() if member_no.strip() else None
                        result = db_manager.get_customer_clusters(
                            batch_no=selected_batch, customer_type=type_filter_val,
                            member_no=member_filter, limit=limit
                        )
                        # 保存查询结果到 session_state
                        st.session_state['cluster_query_result'] = result
                        st.session_state['cluster_query_batch'] = selected_batch
                        st.session_state['cluster_query_type'] = type_filter_val
                        st.session_state['cluster_query_member'] = member_filter
                    
                    # 显示查询结果（如果存在）
                    if 'cluster_query_result' in st.session_state and not st.session_state['cluster_query_result'].empty:
                        result = st.session_state['cluster_query_result']
                        batch_filter = st.session_state.get('cluster_query_batch')
                        type_filter_val = st.session_state.get('cluster_query_type')
                        member_filter = st.session_state.get('cluster_query_member')

                        # 数据源选择
                        data_source = st.radio(
                            "总览数据源",
                            ["当前显示数据", "整个批次数据"],
                            horizontal=True,
                            key="cluster_data_source"
                        )

                        st.markdown("**查询结果总览:**")
                        if data_source == "当前显示数据":
                            # 使用当前显示的数据
                            label_counts = result['customer_type'].value_counts()
                        else:
                            # 优化：直接使用 GROUP BY 聚合查询，避免将百万级数据载入内存
                            label_counts = db_manager.get_customer_type_counts(
                                batch_no=batch_filter,
                                customer_type=type_filter_val,
                                member_no=member_filter,
                            )

                        display_label_counts(label_counts, key_prefix="query_cluster")

                        st.dataframe(result, width='stretch')
                        st.caption(f"共 {len(result)} 条记录")
            except Exception as e:
                st.error(f"查询失败: {e}")

    # ---------- Tab 3: 数据管理 ----------
    with tab3:
        # 显示删除结果消息（如果有）
        delete_msg = st.session_state.get('delete_result_msg')
        if delete_msg:
            if delete_msg['type'] == 'success':
                st.success(delete_msg['text'])
            else:
                st.warning(delete_msg['text'])
                for err in delete_msg.get('errors', []):
                    st.error(f"删除失败 - {err}")
            del st.session_state['delete_result_msg']

        st.subheader("数据删除")
        
        st.warning("注意：删除操作不可恢复，请谨慎操作。")
        
        # 获取两个表的批次列表并合并
        try:
            rfm_batches = set(db_manager.get_batch_list('rfm_analysis'))
            cluster_batches = set(db_manager.get_batch_list('customer_clusters'))
            member_batches = set(db_manager.get_batch_list('member_data'))
            all_batches = sorted(rfm_batches | cluster_batches | member_batches)  # 合并去重
            
            if not all_batches:
                st.info("暂无分析数据。")
            else:
                st.markdown(f"**现有批次列表 ({len(all_batches)} 个):**")
                for batch in all_batches:
                    st.code(batch)
                
                selected_batches = st.multiselect("选择要删除的批次", all_batches)
                
                if selected_batches:
                    # 删除按钮
                    if st.button(f"删除选中的 {len(selected_batches)} 个批次", type="secondary"):
                        # 使用 session_state 进行二次确认
                        st.session_state['confirm_delete_batches'] = selected_batches
                        st.rerun()
                
                if st.session_state.get('confirm_delete_batches'):
                    batches_to_delete = st.session_state['confirm_delete_batches']
                    st.warning(f"确认要删除以下 {len(batches_to_delete)} 个批次吗？此操作不可恢复！")
                    for batch in batches_to_delete:
                        st.code(batch)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        confirm_delete = st.button("确认删除", type="primary", key="confirm_delete_yes")
                    with col2:
                        cancel_delete = st.button("取消", key="confirm_delete_no")

                    if cancel_delete:
                        st.session_state['confirm_delete_batches'] = None
                        st.rerun()

                    if confirm_delete:
                        total_deleted = 0
                        errors = []
                        # 进度条放在 columns 容器外，拉伸至整个页面宽度
                        progress_bar = st.progress(0, text="开始删除...")
                        # 预先统计三个表的总行数，用于进度展示
                        total_to_delete = 0
                        for batch in batches_to_delete:
                            try:
                                total_to_delete += db_manager.count_batch_rows('rfm_analysis', batch)
                                total_to_delete += db_manager.count_batch_rows('customer_clusters', batch)
                                total_to_delete += db_manager.count_batch_rows('member_data', batch)
                            except Exception:
                                pass

                        # 使用 list 作为可变容器，避免闭包对不可变 int 的延迟绑定
                        deleted_so_far = [0]
                        for batch in batches_to_delete:
                            try:
                                # 同时删除三个表的数据（分批删除，每批 5000 行）
                                for table in ['rfm_analysis', 'customer_clusters', 'member_data']:
                                    lbl_table = table
                                    lbl_batch = batch

                                    def _on_prog(d, t, _t=lbl_table, _b=lbl_batch, _ds=deleted_so_far):
                                        cur = _ds[0] + d
                                        pct = min(cur / max(total_to_delete, 1), 1.0)
                                        progress_bar.progress(
                                            pct,
                                            text=f"删除 {_t} 批次 {_b}：{d:,} / {t:,} 行"
                                        )

                                    deleted = db_manager.delete_batch(
                                        table, batch, progress_callback=_on_prog
                                    )
                                    deleted_so_far[0] += deleted
                                    total_deleted += deleted

                            except Exception as e:
                                errors.append(f"{batch}: {e}")

                        progress_bar.empty()
                        st.session_state['confirm_delete_batches'] = None
                        # 清空导出缓存（已删除的批次数据已失效）
                        st.session_state['export_batch_data'] = None
                        st.session_state['export_data_cache'] = {}
                        if errors:
                            st.session_state['delete_result_msg'] = {
                                'type': 'warning',
                                'text': f"部分删除完成，共删除 {total_deleted} 条记录。",
                                'errors': errors,
                            }
                        else:
                            st.session_state['delete_result_msg'] = {
                                'type': 'success',
                                'text': f"成功删除 {len(batches_to_delete)} 个批次，共 {total_deleted} 条记录。",
                                'errors': [],
                            }
                        st.rerun()

                # 危险操作区：清空所有数据
                st.markdown("---")
                st.subheader("清空所有数据")
                st.error(
                    "⚠️ **危险操作**：清空所有数据将使用 TRUNCATE 删除三个表（rfm_analysis、"
                    "customer_clusters、member_data）的全部记录。此操作：\n"
                    "- **不可恢复**，无法回滚\n"
                    "- **会重置自增 ID**\n"
                    "- 比 DELETE 快得多（直接释放数据页）\n\n"
                    "如需删除特定批次，请使用上方的批次删除功能。"
                )

                # 使用 session_state 控制二次确认的显示
                if 'confirm_truncate_all' not in st.session_state:
                    st.session_state['confirm_truncate_all'] = False

                if not st.session_state['confirm_truncate_all']:
                    if st.button("清空所有数据", type="primary", key="btn_truncate_all"):
                        st.session_state['confirm_truncate_all'] = True
                        st.rerun()
                else:
                    st.warning("再次确认：确定要清空所有数据吗？此操作不可恢复！")
                    col_trunc_yes, col_trunc_no = st.columns(2)
                    with col_trunc_yes:
                        if st.button("确认清空", type="primary", key="confirm_truncate_yes"):
                            try:
                                with st.spinner("正在清空所有数据..."):
                                    result = db_manager.truncate_all_tables()
                                # 清空导出缓存与批次选择缓存
                                st.session_state['export_batch_data'] = None
                                st.session_state['export_data_cache'] = {}
                                st.session_state['confirm_truncate_all'] = False
                                st.session_state['confirm_delete_batches'] = None
                                # 统计结果
                                success_tables = [t for t, ok in result.items() if ok]
                                failed_tables = [t for t, ok in result.items() if not ok]
                                if failed_tables:
                                    st.session_state['delete_result_msg'] = {
                                        'type': 'warning',
                                        'text': f"部分清空完成，成功: {', '.join(success_tables)}；失败: {', '.join(failed_tables)}",
                                        'errors': [],
                                    }
                                else:
                                    st.session_state['delete_result_msg'] = {
                                        'type': 'success',
                                        'text': f"已成功清空所有数据（{', '.join(success_tables)}）。",
                                        'errors': [],
                                    }
                                st.rerun()
                            except Exception as e:
                                st.session_state['confirm_truncate_all'] = False
                                st.error(f"清空失败: {e}")
                                st.rerun()
                    with col_trunc_no:
                        if st.button("取消", key="confirm_truncate_no"):
                            st.session_state['confirm_truncate_all'] = False
                            st.rerun()

                st.markdown("---")

                # 导出功能
                st.subheader("数据导出")
                if 'export_data_cache' not in st.session_state:
                    st.session_state['export_data_cache'] = {}
                export_batch = st.selectbox("选择要导出的批次", all_batches, key="export_batch")
                if st.button("导出CSV文件", key="export_query_btn"):
                    try:
                        # 优化：用流式导出，避免一次性载入百万行
                        # 先统计行数
                        rfm_total = db_manager.count_batch_rows('rfm_analysis', export_batch)
                        cluster_total = db_manager.count_batch_rows('customer_clusters', export_batch)
                        member_total = db_manager.count_batch_rows('member_data', export_batch)

                        export_dir = os.path.join(os.path.dirname(__file__), '_exports')
                        os.makedirs(export_dir, exist_ok=True)

                        progress_bar = st.progress(0, text="准备导出...")
                        tmp_paths = {}

                        for label, table, total in [
                            ('rfm', 'rfm_analysis', rfm_total),
                            ('cluster', 'customer_clusters', cluster_total),
                            ('member', 'member_data', member_total),
                        ]:
                            if total == 0:
                                continue
                            tmp_path = os.path.join(export_dir, f"{table}_{export_batch}.csv")

                            def _on_prog(done, t, lbl=label):
                                if t > 0:
                                    pct = min(done / t, 1.0)
                                    progress_bar.progress(
                                        pct,
                                        text=f"导出 {lbl}：{done:,} / {t:,}（{pct*100:.1f}%）"
                                    )

                            db_manager.stream_export_to_csv(
                                table, export_batch, tmp_path,
                                progress_callback=_on_prog,
                            )
                            tmp_paths[label] = (tmp_path, total)

                        progress_bar.empty()

                        # 读取文件到内存供 download_button（已分批写入，文件大小可控）
                        # 若文件极大，建议用户直接从 _exports 目录复制
                        cache = {}
                        for label, (path, cnt) in tmp_paths.items():
                            with open(path, 'rb') as f:
                                cache[label] = (f.read(), cnt)

                        st.session_state['export_data_cache'][export_batch] = cache
                        st.session_state['export_batch_data'] = export_batch
                        st.rerun()
                    except Exception as e:
                        st.error(f"导出失败: {e}")

                export_data_batch = st.session_state.get('export_batch_data')
                if export_data_batch:
                    cached = st.session_state['export_data_cache'].get(export_data_batch)
                    if cached:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if 'rfm' in cached:
                                rfm_bytes, rfm_cnt = cached['rfm']
                                st.download_button(
                                    label=f"下载 RFM分析结果 ({rfm_cnt:,} 条)",
                                    data=rfm_bytes,
                                    file_name=f"rfm_analysis_{export_data_batch}.csv",
                                    mime="text/csv",
                                    key="download_rfm"
                                )
                        with col_b:
                            if 'cluster' in cached:
                                cluster_bytes, cluster_cnt = cached['cluster']
                                st.download_button(
                                    label=f"下载 客户分群结果 ({cluster_cnt:,} 条)",
                                    data=cluster_bytes,
                                    file_name=f"customer_clusters_{export_data_batch}.csv",
                                    mime="text/csv",
                                    key="download_cluster"
                                )
                        with col_c:
                            if 'member' in cached:
                                member_bytes, member_cnt = cached['member']
                                st.download_button(
                                    label=f"下载 原始客户数据 ({member_cnt:,} 条)",
                                    data=member_bytes,
                                    file_name=f"member_data_{export_data_batch}.csv",
                                    mime="text/csv",
                                    key="download_member"
                                )


 
        except Exception as e:
            st.error(f"获取批次列表失败: {e}")


# ============================================================
# 智能查询页面
# ============================================================
def page_query():
    st.title("智能查询")

    api_key = DEFAULT_LLM_CONFIG.get('api_key', '')
    endpoint = DEFAULT_LLM_CONFIG.get('endpoint', '')
    model = DEFAULT_LLM_CONFIG.get('model', '')

    # ---------- 模式提示（共用） ----------
    def _show_mode_hint(engine, force_rule=False, smart_mode=True):
        if not smart_mode:
            st.info("当前为 **手动 SQL 模式**，请直接输入 SELECT 查询语句执行。")
        elif force_rule and engine.llm_available:
            st.warning(f"LLM 已手动关闭（{engine.model}），当前为 **规则模式**。仅支持预定义的常见查询。")
        elif engine.llm_available:
            st.success(f"当前为 **LLM 智能模式**：{engine.model}")
        else:
            st.info("当前为 **规则模式**（未配置 API Key）。仅支持预定义的常见查询。如需更强大的查询能力，请在项目根目录 `.env` 文件中配置 `LLM_API_KEY`。")
            with st.expander(".env 配置说明"):
                st.code(
                    "# 在项目根目录创建 .env 文件，写入以下内容：\n"
                    "LLM_API_KEY=your-api-key-here\n"
                    "LLM_ENDPOINT=https://api.openai.com/v1\n"
                    "LLM_MODEL=gpt-3.5-turbo\n\n"
                    "# 支持任何 OpenAI 兼容接口，例如：\n"
                    "# - 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                    "# - DeepSeek: https://api.deepseek.com\n"
                    "# - OpenAI:   https://api.openai.com/v1",
                    language='bash',
                )

    # ---------- 侧边栏：设置与示例 ----------
    db_examples = QUERY_DB_EXAMPLES

    with st.sidebar:
        # 初始化持久化设置值（使用独立 key，不与 widget key 冲突）
        if "_persist_smart_mode" not in st.session_state:
            st.session_state["_persist_smart_mode"] = True
        if "_persist_use_llm" not in st.session_state:
            st.session_state["_persist_use_llm"] = bool(DEFAULT_LLM_CONFIG.get('api_key', ''))

        def _sync_query_settings():
            st.session_state["_persist_smart_mode"] = st.session_state["smart_mode_toggle"]
            st.session_state["_persist_use_llm"] = st.session_state["use_llm_toggle"]

        # 智能匹配模式开关
        smart_mode = st.toggle(
            "智能匹配模式",
            key="smart_mode_toggle",
            value=st.session_state["_persist_smart_mode"],
            on_change=_sync_query_settings,
            help="开启：输入自然语言问题，由 LLM 或规则自动生成 SQL；关闭：手动输入 SQL 查询语句直接执行。",
        )

        # LLM 模式开关（智能匹配关闭时禁用，手动SQL模式无需LLM）
        has_api_key = bool(DEFAULT_LLM_CONFIG.get('api_key', ''))
        use_llm = st.toggle(
            "使用 LLM ",
            key="use_llm_toggle",
            value=st.session_state["_persist_use_llm"],
            disabled=not smart_mode or not has_api_key,
            on_change=_sync_query_settings,
        )
        if not smart_mode:
            st.caption("手动 SQL 模式下无需 LLM")
        elif not has_api_key:
            st.caption("未配置 API Key，已锁定为规则模式")


        # 仅在智能匹配模式下显示示例选择器
        if smart_mode:
            def _on_db_example_change():
                sel = st.session_state.get("db_example", "")
                if sel:
                    st.session_state["question_db"] = sel
            st.selectbox("选择示例问题", [""] + db_examples, key="db_example",
                         on_change=_on_db_example_change)

    st.subheader("MySQL 数据库查询")
    st.caption("查询已入库的 RFM 分析结果、客户分群数据和原始客户数据，需要先连接数据库。")

    db_manager = st.session_state.get('db_manager')
    db_connected = st.session_state.get('db_connected', False)

    if not db_connected or db_manager is None:
        st.warning("请先进入一次 **数据库管理** 页面来连接 MySQL 数据库。")
    else:
        # 初始化数据库查询引擎
        engine_db = st.session_state.get('nl2sql_engine_db')
        engine_db_id = st.session_state.get('nl2sql_engine_db_conn_id')
        current_db_id = id(db_manager.connection)

        if engine_db is None or engine_db_id != current_db_id:
            try:
                engine_db = NL2SQLQueryEngine(
                    db_manager=db_manager,
                    api_key=api_key, endpoint=endpoint, model=model,
                )
                st.session_state['nl2sql_engine_db'] = engine_db
                st.session_state['nl2sql_engine_db_conn_id'] = current_db_id
            except Exception as e:
                st.error(f"初始化数据库查询引擎失败: {e}")
                engine_db = None

        if engine_db:
            smart_mode = st.session_state.get("smart_mode_toggle", True)
            _show_mode_hint(engine_db, force_rule=not use_llm, smart_mode=smart_mode)

            # 获取可用批次列表（仅智能匹配模式下需要）
            batch_no = None
            if smart_mode:
                try:
                    rfm_batches = db_manager.get_batch_list('rfm_analysis')
                    cluster_batches = db_manager.get_batch_list('customer_clusters')
                    member_batches = db_manager.get_batch_list('member_data')
                    batch_list = sorted(set(rfm_batches) | set(cluster_batches) | set(member_batches))
                except Exception:
                    batch_list = []

                if batch_list:
                    selected_batch = st.selectbox(
                        "选择数据批次（查询前必须选择）",
                        batch_list,
                        index=len(batch_list) - 1,
                        key="db_selected_batch",
                    )
                    batch_no = selected_batch
                else:
                    st.info("暂无已入库的批次数据，请先在数据管理页面入库数据。")

            if smart_mode:
                # 智能匹配模式：自然语言问题输入
                with st.form("form_db_query"):
                    if "question_db" not in st.session_state:
                        st.session_state["question_db"] = ""
                    question_db = st.text_input(
                        "输入您的问题（按 Enter 提交）",
                        placeholder="例如：各客户类型有多少客户",
                        key="question_db",
                    )
                    submitted_db = st.form_submit_button("执行查询", width='stretch')

                if submitted_db and question_db.strip():
                    try:
                        force_rule = not use_llm
                        use_llm_mode = engine_db.llm_available and not force_rule

                        mode_text = "调用大模型生成SQL" if use_llm_mode else "规则匹配生成SQL"
                        with st.spinner(f"正在{mode_text}并执行查询..."):
                            sql, result_df, status, error_msg = engine_db.query(question_db.strip(), batch_no=batch_no, force_rule=force_rule)

                        st.subheader("生成的 SQL")
                        if sql:
                            st.code(sql, language='sql')
                        else:
                            st.warning("未能生成 SQL 语句。")

                        if status == 'success' and result_df is not None:
                            st.subheader("查询结果")
                            st.success(f"查询成功，共 {len(result_df)} 条记录。")
                            paginated_data_preview(result_df, key_prefix="db_query_result", default_rows=20)
                            csv_bytes = convert_df_to_csv(result_df)
                            st.download_button(
                                label="下载查询结果 (CSV)", data=csv_bytes,
                                file_name="db_query_result.csv", mime="text/csv",
                            )
                        else:
                            st.error(f"查询失败: {error_msg}")
                    except Exception as e:
                        st.error(f"查询过程异常: {e}")
            else:
                # 手动SQL模式
                with st.form("form_db_sql"):
                    if "manual_sql_db" not in st.session_state:
                        st.session_state["manual_sql_db"] = ""
                    manual_sql_db = st.text_area(
                        "输入 SQL 查询语句",
                        placeholder="SELECT * FROM rfm_analysis LIMIT 10",
                        key="manual_sql_db",
                        height=120,
                        help="直接输入 SELECT 查询语句执行，仅允许 SELECT 查询。",
                    )
                    submitted_db_sql = st.form_submit_button("执行 SQL", width='stretch')

                if submitted_db_sql and manual_sql_db.strip():
                    try:
                        sql_input = manual_sql_db.strip()
                        is_valid, err_msg = engine_db.validate_sql(sql_input)
                        if not is_valid:
                            st.error(f"SQL 校验失败: {err_msg}")
                        else:
                            with st.spinner("正在执行 SQL 查询..."):
                                result_df, error_msg = engine_db.execute_query(sql_input)

                            st.subheader("执行的 SQL")
                            st.code(sql_input, language='sql')

                            if result_df is not None:
                                st.subheader("查询结果")
                                st.success(f"查询成功，共 {len(result_df)} 条记录。")
                                paginated_data_preview(result_df, key_prefix="db_query_result", default_rows=20)
                                csv_bytes = convert_df_to_csv(result_df)
                                st.download_button(
                                    label="下载查询结果 (CSV)", data=csv_bytes,
                                    file_name="db_query_result.csv", mime="text/csv",
                                )
                            else:
                                st.error(f"查询失败: {error_msg}")
                    except Exception as e:
                        st.error(f"查询过程异常: {e}")

    # ---------- 查询历史 ----------
    st.markdown("---")
    st.subheader("本次查询历史")

    all_history = []
    engine_db = st.session_state.get('nl2sql_engine_db')
    if engine_db:
        for h in engine_db.get_history():
            all_history.append(h)

    if all_history:
        for i, record in enumerate(reversed(all_history)):
            status_icon = "✓" if record['status'] == 'success' else "✗"
            with st.expander(f"{status_icon} {record['question'][:50]}..."):
                st.markdown(f"**问题:** {record['question']}")
                if record['sql']:
                    st.code(record['sql'], language='sql')
                if record['status'] == 'success' and record['result_preview'] is not None:
                    st.dataframe(record['result_preview'], width='stretch')
                elif record['error_message']:
                    st.error(record['error_message'])

        if st.button("清空历史"):
            if engine_db:
                engine_db.clear_history()
            st.rerun()
    else:
        st.info("暂无查询记录。")


# ============================================================
# 智能客服页面
# ============================================================
def page_smart_assistant():
    st.title("智能客服")

    api_key = DEFAULT_LLM_CONFIG.get('api_key', '')
    endpoint = DEFAULT_LLM_CONFIG.get('endpoint', '')
    model = DEFAULT_LLM_CONFIG.get('model', '')

    if not api_key:
        st.warning("未配置 API Key，智能客服无法使用。请在项目根目录 `.env` 文件中配置 `LLM_API_KEY`。")
        with st.expander(".env 配置说明"):
            st.code(
                "# 在项目根目录创建 .env 文件，写入以下内容：\n"
                "LLM_API_KEY=your-api-key-here\n"
                "LLM_ENDPOINT=https://api.openai.com/v1\n"
                "LLM_MODEL=gpt-3.5-turbo\n\n"
                "# 支持任何 OpenAI 兼容接口，例如：\n"
                "# - 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                "# - DeepSeek: https://api.deepseek.com\n"
                "# - OpenAI:   https://api.openai.com/v1",
                language='bash',
            )
        return

    # ---------- 侧边栏配置 ----------
    with st.sidebar:
        st.header("智能客服设置")

        # MCP 模式开关
        if "_persist_mcp_mode_assistant" not in st.session_state:
            st.session_state["_persist_mcp_mode_assistant"] = False

        def _sync_mcp_mode_assistant():
            st.session_state["_persist_mcp_mode_assistant"] = st.session_state["mcp_mode_toggle_assistant"]

        mcp_mode = st.toggle(
            "MCP 模式",
            key="mcp_mode_toggle_assistant",
            value=st.session_state["_persist_mcp_mode_assistant"],
            on_change=_sync_mcp_mode_assistant,
            help="开启后，LLM 通过 MCP 工具调用获取数据（并发查询，性能更优）；关闭则使用默认模式。",
        )

        # 初始化持久化设置值（使用独立 key，不与 widget key 冲突）
        # 批次选择
        batch_no = None
        db_manager = st.session_state.get('db_manager')
        if db_manager and db_manager.is_connected():
            try:
                # 获取RFM和聚类批次列表
                rfm_batches = db_manager.get_batch_list('rfm_analysis')
                cluster_batches = db_manager.get_batch_list('customer_clusters')
                all_batches = sorted(set(rfm_batches + cluster_batches), reverse=True)
                
                if all_batches:
                    selected_batch = st.selectbox(
                        "选择分析批次",
                        all_batches,
                        index=0,
                        key="assistant_batch_select",
                        help="选择特定批次进行分析"
                    )
                    batch_no = selected_batch
                    st.caption(f"当前批次: {batch_no}")
                else:
                    st.warning("数据库中暂无批次数据")
            except Exception as e:
                st.error(f"获取批次列表失败: {str(e)}")
        else:
            st.warning("数据库未连接，请先连接数据库")


        # 预定义模板
        st.markdown("---")
        st.header("快速提问")
        template_names = list(CHAT_TEMPLATES.keys())

        def _on_template_change():
            sel = st.session_state.get("assistant_template", "")
            if sel:
                st.session_state["assistant_pending_template"] = CHAT_TEMPLATES[sel]

        st.selectbox(
            "选择模板问题（立即发送）",
            [""] + template_names,
            key="assistant_template",
            on_change=_on_template_change,
        )

        # 清空对话按钮
        if st.button("新建对话", width='stretch', key="clear_assistant_chat"):
            st.session_state['assistant_chat_history'] = []
            st.rerun()

    # ---------- Session state 初始化 ----------
    if 'assistant_chat_history' not in st.session_state:
        st.session_state['assistant_chat_history'] = []

    # ---------- 初始化 SmartAssistant ----------
    assistant = st.session_state.get('assistant_instance')
    if assistant is None:
        db_mgr = st.session_state.get('db_manager')
        assistant = SmartAssistant(api_key=api_key, endpoint=endpoint, model=model, db_manager=db_mgr)
        st.session_state['assistant_instance'] = assistant

    st.success(f"LLM 模型：{model}")

    # ---------- 数据源状态检查 ----------
    is_data_ready = False
    db_connected = st.session_state.get('db_connected', False)
    db_manager = st.session_state.get('db_manager')
    if not db_connected or db_manager is None:
        st.warning("请先进入一次 **数据库管理** 页面来连接 MySQL 数据库。")
    else:
        is_data_ready = True
        st.info(f"已连接数据库：{MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")

    # ---------- 渲染历史对话 ----------
    chat_history = st.session_state['assistant_chat_history']
    for msg in chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            # MCP 执行路径展示
            if msg['role'] == 'assistant' and msg.get('mcp_tool_calls'):
                mcp_calls = msg['mcp_tool_calls']
                with st.expander(f"MCP 执行路径（共 {len(mcp_calls)} 轮工具调用）", expanded=False):
                    for idx, tc in enumerate(mcp_calls, 1):
                        call = tc['call']
                        result = tc['result']
                        round_num = call.get('round', idx)
                        st.markdown(f"#### 第 {idx} 轮（Round {round_num}）")
                        st.markdown(f"**工具名称：** `{call.get('tool_name', '')}`")
                        args = call.get('arguments', {})
                        if args:
                            questions = args.get('questions', [])
                            if questions:
                                st.markdown(f"**查询问题（{len(questions)} 条）：**")
                                for i, q in enumerate(questions, 1):
                                    st.markdown(f"  {i}. {q}")
                            if args.get('batch_no'):
                                st.caption(f"批次: {args['batch_no']}")
                        if result:
                            status_icon = "✅" if result.get('status') == 'success' else "❌"
                            st.markdown(f"**执行结果：** {status_icon} {result.get('result_summary', '')}")
                            sub_results = result.get('sub_results', [])
                            if sub_results:
                                st.markdown("**子请求执行详情：**")
                                for si, sr in enumerate(sub_results, 1):
                                    s_icon = "✅" if sr.get('status') == 'success' else "❌"
                                    st.markdown(f"  **{si}. {s_icon} {sr.get('question', '')}**")
                                    if sr.get('sql'):
                                        st.code(sr['sql'], language='sql')
                                    if sr.get('was_fixed'):
                                        st.warning(f"⚠️ 经过 {sr.get('retry_count', 0)} 次自修复后成功")
                                        with st.expander("查看修复前的原始 SQL 和错误", expanded=False):
                                            st.markdown("**原始 SQL：**")
                                            st.code(sr.get('original_sql', ''), language='sql')
                                            st.markdown(f"**原始错误：** ")
                                            st.error(sr.get('original_error', ''))
                                    elif sr.get('error'):
                                        st.error(f"错误: {sr['error']}")
                                        if sr.get('retry_count', 0) > 0:
                                            st.caption(f"（已尝试 {sr.get('retry_count', 0)} 次自修复均失败）")
                                    if sr.get('status') == 'success':
                                        st.caption(f"返回 {sr.get('row_count', 0)} 行数据")
                                        sr_data = sr.get('data')
                                        if sr_data:
                                            with st.expander(f"查看查询数据（{len(sr_data)} 行）", expanded=False):
                                                try:
                                                    st.dataframe(pd.DataFrame(sr_data), width='stretch')
                                                except Exception:
                                                    st.json(sr_data)
                        st.divider()
            # 折叠展示查询详情（SQL + 结果/错误配对显示）
            if msg['role'] == 'assistant' and msg.get('query_items'):
                items = msg['query_items']
                with st.expander(f"查看查询详情（共 {len(items)} 条）"):
                    for idx, item in enumerate(items):
                        st.markdown(f"**第 {idx+1} 条查询：**" + (" (已自动修复)" if item.get('fixed') else ""))
                        if item.get('sql'):
                            st.code(item['sql'], language='sql')
                        if item.get('error'):
                            st.error(item['error'])
                        elif item.get('data') is not None:
                            data_val = item['data']
                            if isinstance(data_val, list):
                                for df in data_val:
                                    st.dataframe(df, width='stretch')
                            else:
                                st.dataframe(data_val, width='stretch')
                        # 对经过修复的 SQL，展示原始 SQL 和原始错误
                        if item.get('fixed') and item.get('original_sql'):
                            with st.expander("查看修复前的原始 SQL 和错误", expanded=False):
                                st.markdown("**原始 SQL：**")
                                st.code(item['original_sql'], language='sql')
                                st.markdown(f"**原始错误：** ")
                                st.error(item.get('original_error', ''))
                        st.divider()

    # ---------- 处理模板注入 ----------
    pending_template = st.session_state.pop('assistant_pending_template', None)

    # ---------- 用户输入 ----------
    user_input = st.chat_input("输入您的问题...", key="assistant_chat_input")

    # 如果有模板注入，优先使用
    if pending_template and is_data_ready:
        user_input = pending_template

    if user_input and is_data_ready:
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)

        # 调用 SmartAssistant 流式输出
        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            def stream_generator():
                """处理流式事件并返回文本片段，同时将查询详情存入 session_state"""
                items = []
                mcp_tool_calls = []  # 收集所有 MCP 工具调用轮次
                dm = st.session_state.get('db_manager')
                mcp_enabled = st.session_state.get("_persist_mcp_mode_assistant", False)

                # 根据模式选择调用方法
                if mcp_enabled:
                    event_source = assistant.chat_stream_mcp(
                        user_message=user_input,
                        db_manager=dm,
                        batch_no=batch_no,
                    )
                else:
                    event_source = assistant.chat_stream(
                        user_message=user_input,
                        db_manager=dm,
                        batch_no=batch_no,
                    )

                chunk_started = False
                # 跟踪当前工具调用状态，用于状态流转显示
                current_questions = []
                question_statuses = {}  # {question: {'icon': str, 'detail': str}}

                def _render_questions(header):
                    """根据 question_statuses 构建多行状态文本并刷新 placeholder。"""
                    lines = [header]
                    for i, q in enumerate(current_questions, 1):
                        st_info = question_statuses.get(q, {'icon': '⏳', 'detail': ''})
                        lines.append(f"  {i}. {st_info['icon']} {q}{st_info['detail']}")
                    status_placeholder.info('\n'.join(lines))

                for event in event_source:
                    etype = event.get('type')
                    if etype == 'status':
                        if current_questions:
                            _render_questions(f"🔄 {event['content']}")
                        else:
                            status_placeholder.info(event['content'])
                    elif etype == 'tool_call':
                        round_num = event.get('round', 1)
                        tool_name = event.get('tool_name', '')
                        args = event.get('arguments', {})
                        questions = args.get('questions', [])
                        current_questions = questions
                        question_statuses = {q: {'icon': '⏳', 'detail': ''} for q in questions}
                        _render_questions(f"工具调用（第 {round_num} 轮）：{tool_name} — 正在生成查询...")
                        mcp_tool_calls.append({'call': event, 'result': None})
                    elif etype == 'tool_progress':
                        info = event.get('info', {})
                        q = info.get('question', '')
                        phase = info.get('phase', '')
                        if q in question_statuses:
                            if phase == 'generating_sql':
                                question_statuses[q] = {'icon': '🔢', 'detail': ' — 生成SQL中...'}
                            elif phase == 'executing':
                                retry = info.get('retry')
                                hint = f"（第{retry}次重试）" if retry else ''
                                question_statuses[q] = {'icon': '🔍', 'detail': f' — 执行查询{hint}...'}
                            elif phase == 'retrying':
                                attempt = info.get('attempt', 1)
                                max_retries = info.get('max_retries', 2)
                                question_statuses[q] = {'icon': '🔄', 'detail': f' — 自修复中（{attempt}/{max_retries}）...'}
                            elif phase == 'done':
                                r = info.get('result', {})
                                if r.get('status') == 'success':
                                    row_info = f" — 返回 {r.get('row_count', 0)} 行"
                                    fix_info = f"（自修复{r.get('retry_count', 0)}次）" if r.get('was_fixed') else ''
                                    question_statuses[q] = {'icon': '✅', 'detail': f"{row_info}{fix_info}"}
                                else:
                                    question_statuses[q] = {'icon': '❌', 'detail': f" — {r.get('error', '失败')[:50]}"}
                            _render_questions(f"🔄 正在执行查询（第 {event.get('round', 1)} 轮）...")
                    elif etype == 'tool_result':
                        round_num = event.get('round', 1)
                        summary = event.get('result_summary', '')
                        result_status = event.get('status', '')
                        sub_results = event.get('sub_results', [])
                        icon = "✅" if result_status == 'success' else "❌"

                        detail_lines = [f"{icon} 工具执行完成（第 {round_num} 轮）：{summary}"]
                        for si, sr in enumerate(sub_results, 1):
                            s_icon = "✅" if sr.get('status') == 'success' else "❌"
                            row_info = f" — 返回 {sr.get('row_count', 0)} 行" if sr.get('status') == 'success' else ''
                            fix_info = f"（自修复 {sr.get('retry_count', 0)} 次后成功）" if sr.get('was_fixed') else ''
                            detail_lines.append(f"  {si}. {s_icon} {sr.get('question', '')}{row_info}{fix_info}")
                        status_placeholder.info('\n'.join(detail_lines))

                        # 清除当前工具跟踪
                        current_questions = []
                        question_statuses = {}

                        if mcp_tool_calls:
                            mcp_tool_calls[-1]['result'] = event

                        # 每轮处理完成后延迟 2 秒，方便用户查看最终结果
                        time.sleep(2)
                    elif etype == 'sql':
                        if items and items[-1]['data'] is None and items[-1]['error'] is None:
                            items[-1]['sql'] = event['content']
                        else:
                            items.append({'sql': event['content'], 'data': None, 'error': None, 'fixed': False, 'original_sql': None, 'original_error': ''})
                    elif etype == 'sql_fixed':
                        if items:
                            items[-1]['fixed'] = True
                            items[-1]['original_sql'] = event.get('original_sql')
                            items[-1]['original_error'] = event.get('original_error', '')
                    elif etype == 'data':
                        if items:
                            items[-1]['data'] = event['content']
                    elif etype == 'sql_error':
                        if items:
                            items[-1]['error'] = event['content']
                        else:
                            items.append({'sql': None, 'data': None, 'error': None, 'fixed': False, 'original_sql': None, 'original_error': ''})
                    elif etype == 'chunk':
                        # 开始生成答案时清空实时状态显示
                        if not chunk_started:
                            status_placeholder.empty()
                            chunk_started = True
                        yield event['content']

                st.session_state['_pending_query_items'] = items
                # 保留 sub_results 中的 data 字段（经序列化处理），用于执行轨迹中展示查询数据
                sanitized_mcp_calls = []
                for tc in mcp_tool_calls:
                    sanitized_tc = {'call': tc['call'], 'result': None}
                    if tc['result']:
                        result_copy = dict(tc['result'])
                        sub_results = result_copy.get('sub_results', [])
                        if sub_results:
                            result_copy['sub_results'] = [
                                {**sr, 'data': _sanitize_mcp_data(sr.get('data'))}
                                for sr in sub_results
                            ]
                        sanitized_tc['result'] = result_copy
                    sanitized_mcp_calls.append(sanitized_tc)
                st.session_state['_pending_mcp_tool_calls'] = sanitized_mcp_calls

            # 使用 st.write_stream 流式显示文本
            reply_text = st.write_stream(stream_generator)

            # MCP 模式：展示完整的工具执行路径（所有轮次）
            mcp_tool_calls = st.session_state.pop('_pending_mcp_tool_calls', [])
            if mcp_tool_calls:
                with st.expander(f"MCP 执行路径（共 {len(mcp_tool_calls)} 轮工具调用）", expanded=False):
                    for idx, tc in enumerate(mcp_tool_calls, 1):
                        call = tc['call']
                        result = tc['result']
                        round_num = call.get('round', idx)
                        st.markdown(f"#### 第 {idx} 轮（Round {round_num}）")
                        st.markdown(f"**工具名称：** `{call.get('tool_name', '')}`")
                        args = call.get('arguments', {})
                        if args:
                            questions = args.get('questions', [])
                            if questions:
                                st.markdown(f"**查询问题（{len(questions)} 条）：**")
                                for i, q in enumerate(questions, 1):
                                    st.markdown(f"  {i}. {q}")
                            if args.get('batch_no'):
                                st.caption(f"批次: {args['batch_no']}")
                        if result:
                            status_icon = "✅" if result.get('status') == 'success' else "❌"
                            st.markdown(f"**执行结果：** {status_icon} {result.get('result_summary', '')}")
                            # 展示子请求详情
                            sub_results = result.get('sub_results', [])
                            if sub_results:
                                st.markdown("**子请求执行详情：**")
                                for si, sr in enumerate(sub_results, 1):
                                    s_icon = "✅" if sr.get('status') == 'success' else "❌"
                                    st.markdown(f"  **{si}. {s_icon} {sr.get('question', '')}**")
                                    if sr.get('sql'):
                                        st.code(sr['sql'], language='sql')
                                    if sr.get('was_fixed'):
                                        st.warning(f"⚠️ 经过 {sr.get('retry_count', 0)} 次自修复后成功")
                                        with st.expander("查看修复前的原始 SQL 和错误", expanded=False):
                                            st.markdown("**原始 SQL：**")
                                            st.code(sr.get('original_sql', ''), language='sql')
                                            st.markdown(f"**原始错误：** ")
                                            st.error(sr.get('original_error', ''))
                                    elif sr.get('error'):
                                        st.error(f"错误: {sr['error']}")
                                        if sr.get('retry_count', 0) > 0:
                                            st.caption(f"（已尝试 {sr.get('retry_count', 0)} 次自修复均失败）")
                                    if sr.get('status') == 'success':
                                        st.caption(f"返回 {sr.get('row_count', 0)} 行数据")
                                        # 嵌套 expander 展示查询到的数据（默认不展开）
                                        sr_data = sr.get('data')
                                        if sr_data:
                                            with st.expander(f"查看查询数据（{len(sr_data)} 行）", expanded=False):
                                                try:
                                                    st.dataframe(pd.DataFrame(sr_data), width='stretch')
                                                except Exception:
                                                    st.json(sr_data)
                        st.divider()

            # 清除状态提示
            status_placeholder.empty()

            # 从 session_state 读取收集的查询详情
            query_items = st.session_state.pop('_pending_query_items', [])

            # 折叠展示：每条 SQL 和对应结果/错误合并在一个 expander 中
            if query_items:
                with st.expander(f"查看查询详情（共 {len(query_items)} 条）"):
                    for idx, item in enumerate(query_items):
                        st.markdown(f"**第 {idx+1} 条查询：**" + (" (已自动修复)" if item.get('fixed') else ""))
                        if item['sql']:
                            st.code(item['sql'], language='sql')
                        if item['error']:
                            st.error(item['error'])
                        elif item['data'] is not None:
                            st.dataframe(item['data'], width='stretch')
                        # 对经过修复的 SQL，展示原始 SQL 和原始错误
                        if item.get('fixed') and item.get('original_sql'):
                            with st.expander("查看修复前的原始 SQL 和错误", expanded=False):
                                st.markdown("**原始 SQL：**")
                                st.code(item['original_sql'], language='sql')
                                st.markdown(f"**原始错误：** ")
                                st.error(item.get('original_error', ''))
                        st.divider()

        # 保存到 session_state 对话历史
        chat_history.append({
            'role': 'user',
            'content': user_input,
        })
        chat_history.append({
            'role': 'assistant',
            'content': reply_text,
            'query_items': query_items if query_items else None,
            'mcp_tool_calls': mcp_tool_calls if mcp_tool_calls else None,
        })
        st.session_state['assistant_chat_history'] = chat_history

        st.rerun()

    elif user_input and not is_data_ready:
        st.error("当前没有可用数据，请先加载数据或连接数据库。")


# ============================================================
# 智能报告生成页面
# ============================================================
def page_report_generator():
    st.title("智能报告生成")

    api_key = DEFAULT_LLM_CONFIG.get('api_key', '')
    endpoint = DEFAULT_LLM_CONFIG.get('endpoint', '')
    model = DEFAULT_LLM_CONFIG.get('model', '')

    if not api_key:
        st.warning("未配置 API Key，智能报告生成功能无法使用。请在项目根目录 `.env` 文件中配置 `LLM_API_KEY`。")
        with st.expander(".env 配置说明"):
            st.code(
                "# 在项目根目录创建 .env 文件，写入以下内容：\n"
                "LLM_API_KEY=your-api-key-here\n"
                "LLM_ENDPOINT=https://api.openai.com/v1\n"
                "LLM_MODEL=gpt-3.5-turbo\n\n"
                "# 支持任何 OpenAI 兼容接口，例如：\n"
                "# - 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                "# - DeepSeek: https://api.deepseek.com\n"
                "# - OpenAI:   https://api.openai.com/v1",
                language='bash',
            )
        return

    # ---------- Session state 初始化 ----------
    if 'report_generator' not in st.session_state:
        st.session_state['report_generator'] = ReportGenerator(api_key=api_key, endpoint=endpoint, model=model)
    if 'generated_report' not in st.session_state:
        st.session_state['generated_report'] = None

    # ---------- 侧边栏配置 ----------
    with st.sidebar:
        st.header("报告生成设置")

        # MCP 模式开关
        if "_persist_mcp_mode_report" not in st.session_state:
            st.session_state["_persist_mcp_mode_report"] = False

        def _sync_mcp_mode_report():
            st.session_state["_persist_mcp_mode_report"] = st.session_state["mcp_mode_toggle_report"]

        mcp_mode = st.toggle(
            "MCP 模式",
            key="mcp_mode_toggle_report",
            value=st.session_state["_persist_mcp_mode_report"],
            on_change=_sync_mcp_mode_report,
            help="开启后，LLM 通过 MCP 工具调用获取分析数据（并发查询，性能更优）；关闭则使用默认模式。",
        )

        # 初始化持久化设置值（使用独立 key，不与 widget key 冲突）
        if "_persist_report_type" not in st.session_state:
            st.session_state["_persist_report_type"] = list(REPORT_TYPES.keys())[0]
        if "_persist_report_detail" not in st.session_state:
            st.session_state["_persist_report_detail"] = list(REPORT_DETAIL_LEVELS.keys())[1]  # 默认"标准版"
        if "_persist_report_batch" not in st.session_state:
            st.session_state["_persist_report_batch"] = None

        def _sync_report_type():
            st.session_state["_persist_report_type"] = st.session_state["report_type_select"]

        def _sync_report_detail():
            st.session_state["_persist_report_detail"] = st.session_state["report_detail_select"]

        def _sync_report_batch():
            st.session_state["_persist_report_batch"] = st.session_state["report_batch_select"]

        # 批次选择
        batch_no = None
        db_manager = st.session_state.get('db_manager')
        if db_manager and db_manager.is_connected():
            try:
                # 获取RFM和聚类批次列表
                rfm_batches = db_manager.get_batch_list('rfm_analysis')
                cluster_batches = db_manager.get_batch_list('customer_clusters')
                all_batches = sorted(set(rfm_batches + cluster_batches), reverse=True)
                
                if all_batches:
                    # 确保保存的批次值在当前批次列表中
                    saved_batch = st.session_state["_persist_report_batch"]
                    if saved_batch not in all_batches:
                        saved_batch = all_batches[0]
                    
                    selected_batch = st.selectbox(
                        "选择分析批次",
                        all_batches,
                        index=all_batches.index(saved_batch) if saved_batch in all_batches else 0,
                        key="report_batch_select",
                        on_change=_sync_report_batch,
                        help="选择特定批次生成报告（不同批次数据不会合并）"
                    )
                    batch_no = selected_batch
                    st.caption(f"当前批次: {batch_no}")
                else:
                    st.warning("数据库中暂无批次数据")
            except Exception as e:
                st.error(f"获取批次列表失败: {str(e)}")
        else:
            st.warning("数据库未连接，请先连接数据库")


        # 报告类型选择
        report_types_list = list(REPORT_TYPES.keys())
        saved_type = st.session_state["_persist_report_type"]
        type_index = report_types_list.index(saved_type) if saved_type in report_types_list else 0
        
        report_type = st.selectbox(
            "报告类型",
            report_types_list,
            index=type_index,
            key="report_type_select",
            on_change=_sync_report_type,
            help="选择要生成的报告类型"
        )

        # 显示报告类型描述
        if report_type:
            st.caption(REPORT_TYPES[report_type]["description"])

        # 详细程度选择
        detail_levels_list = list(REPORT_DETAIL_LEVELS.keys())
        saved_detail = st.session_state["_persist_report_detail"]
        detail_index = detail_levels_list.index(saved_detail) if saved_detail in detail_levels_list else 1

        detail_level = st.selectbox(
            "详细程度",
            detail_levels_list,
            index=detail_index,
            key="report_detail_select",
            on_change=_sync_report_detail,
            help="选择报告的详细程度"
        )

        # 显示详细程度说明
        if detail_level:
            st.caption(REPORT_DETAIL_LEVELS[detail_level]["description"])

    # ---------- 主区域 ----------
    st.success(f"LLM 模型：{model}")

    # 数据库连接状态检查
    db_connected = st.session_state.get('db_connected', False)
    db_manager = st.session_state.get('db_manager')

    is_data_ready = False
    if not db_connected or db_manager is None:
        st.warning("请先进入一次 **数据库管理** 页面来连接 MySQL 数据库。")
    else:
        is_data_ready = True
        st.info(f"已连接数据库：{MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")

    # 报告生成按钮
    def _on_generate_click():
        st.session_state['generated_report'] = None
        st.session_state.pop('_report_sql_results', None)
        st.session_state.pop('_report_mcp_tool_calls', None)
        st.session_state["_pending_report_gen"] = True
        st.session_state["_pending_report_type"] = st.session_state.get("report_type_select", "")
        st.session_state["_pending_report_detail"] = st.session_state.get("report_detail_select", "")

    generate_clicked = st.button(
        "生成报告",
        width='stretch',
        type="secondary",
        disabled=not is_data_ready,
        on_click=_on_generate_click,
    )

    # 初始化 pending 标志
    if "_pending_report_gen" not in st.session_state:
        st.session_state["_pending_report_gen"] = False

    # 统一报告显示区域：只有一个容器，流式输出和最终报告都在此渲染
    report_area = st.empty()

    if st.session_state["_pending_report_gen"]:
        st.session_state["_pending_report_gen"] = False
        report_type = st.session_state.get("_pending_report_type", report_type)
        detail_level = st.session_state.get("_pending_report_detail", detail_level)

        report_generator = st.session_state['report_generator']

        status_placeholder = st.empty()
        full_report = ""
        sqls_list = []
        sql_results_list = []
        mcp_tool_calls = []  # 收集所有 MCP 工具调用轮次
        chunk_started = False
        # 跟踪当前工具调用状态，用于状态流转显示
        current_questions = []
        question_statuses = {}  # {question: {'icon': str, 'detail': str}}

        def _render_questions(header):
            """根据 question_statuses 构建多行状态文本并刷新 placeholder。"""
            lines = [header]
            for i, q in enumerate(current_questions, 1):
                st_info = question_statuses.get(q, {'icon': '⏳', 'detail': ''})
                lines.append(f"  {i}. {st_info['icon']} {q}{st_info['detail']}")
            status_placeholder.info('\n'.join(lines))

        mcp_enabled = st.session_state.get("_persist_mcp_mode_report", False)

        for event in report_generator.generate_report_stream(
            report_type=report_type,
            detail_level=detail_level,
            db_manager=db_manager,
            batch_no=batch_no,
            enable_mcp=mcp_enabled,
        ):
            if event['type'] == 'status':
                if current_questions:
                    _render_questions(f"🔄 {event['content']}")
                else:
                    status_placeholder.info(event['content'])
            elif event['type'] == 'sql':
                sqls_list = event['sqls']
            elif event['type'] == 'sql_result':
                sql_results_list.append({
                    'index': event['index'],
                    'total': event['total'],
                    'sql': event['sql'],
                    'result': event['result'],
                    'dataframe': event.get('dataframe'),
                    'error': event['error'],
                    'was_fixed': event.get('was_fixed', False),
                    'original_sql': event.get('original_sql'),
                    'original_error': event.get('original_error', ''),
                })
            elif event['type'] == 'tool_call':
                round_num = event.get('round', 1)
                tool_name = event.get('tool_name', '')
                args = event.get('arguments', {})
                questions = args.get('questions', [])
                current_questions = questions
                question_statuses = {q: {'icon': '⏳', 'detail': ''} for q in questions}
                _render_questions(f"工具调用（第 {round_num} 轮）：{tool_name} — 正在生成查询...")
                mcp_tool_calls.append({'call': event, 'result': None})
            elif event['type'] == 'tool_progress':
                info = event.get('info', {})
                q = info.get('question', '')
                phase = info.get('phase', '')
                if q in question_statuses:
                    if phase == 'generating_sql':
                        question_statuses[q] = {'icon': '🔢', 'detail': ' — 生成SQL中...'}
                    elif phase == 'executing':
                        retry = info.get('retry')
                        hint = f"（第{retry}次重试）" if retry else ''
                        question_statuses[q] = {'icon': '🔍', 'detail': f' — 执行查询{hint}...'}
                    elif phase == 'retrying':
                        attempt = info.get('attempt', 1)
                        max_retries = info.get('max_retries', 2)
                        question_statuses[q] = {'icon': '🔄', 'detail': f' — 自修复中（{attempt}/{max_retries}）...'}
                    elif phase == 'done':
                        r = info.get('result', {})
                        if r.get('status') == 'success':
                            row_info = f" — 返回 {r.get('row_count', 0)} 行"
                            fix_info = f"（自修复{r.get('retry_count', 0)}次）" if r.get('was_fixed') else ''
                            question_statuses[q] = {'icon': '✅', 'detail': f"{row_info}{fix_info}"}
                        else:
                            question_statuses[q] = {'icon': '❌', 'detail': f" — {r.get('error', '失败')[:50]}"}
                    _render_questions(f"🔄 正在执行查询（第 {event.get('round', 1)} 轮）...")
            elif event['type'] == 'tool_result':
                round_num = event.get('round', 1)
                summary = event.get('result_summary', '')
                result_status = event.get('status', '')
                sub_results = event.get('sub_results', [])
                icon = "✅" if result_status == 'success' else "❌"

                detail_lines = [f"{icon} 工具执行完成（第 {round_num} 轮）：{summary}"]
                for si, sr in enumerate(sub_results, 1):
                    s_icon = "✅" if sr.get('status') == 'success' else "❌"
                    row_info = f" — 返回 {sr.get('row_count', 0)} 行" if sr.get('status') == 'success' else ''
                    fix_info = f"（自修复 {sr.get('retry_count', 0)} 次后成功）" if sr.get('was_fixed') else ''
                    detail_lines.append(f"  {si}. {s_icon} {sr.get('question', '')}{row_info}{fix_info}")
                status_placeholder.info('\n'.join(detail_lines))

                # 清除当前工具跟踪
                current_questions = []
                question_statuses = {}

                if mcp_tool_calls:
                    mcp_tool_calls[-1]['result'] = event

                # 每轮处理完成后延迟 2 秒，方便用户查看最终结果
                time.sleep(2)
            elif event['type'] == 'chunk':
                # 开始生成答案时清空实时状态显示
                if not chunk_started:
                    status_placeholder.empty()
                    chunk_started = True
                full_report += event['content']
                report_area.markdown(full_report)
            elif event['type'] == 'done':
                status_placeholder.empty()

                if event['status'] == 'success':
                    st.session_state['generated_report'] = {
                        'content': event['report'],
                        'type': event.get('report_type', report_type),
                        'detail_level': event.get('detail_level', detail_level),
                        'generated_at': event.get('generated_at'),
                    }
                    # 持久化 SQL 执行结果，供 rerun 后展示
                    if sql_results_list:
                        st.session_state['_report_sql_results'] = sql_results_list
                    # 持久化 MCP 工具调用详情，供 rerun 后展示
                    # 保留 sub_results 中的 data 字段（经序列化处理），用于执行轨迹中展示查询数据
                    sanitized_mcp_calls = []
                    for tc in mcp_tool_calls:
                        sanitized_tc = {'call': tc['call'], 'result': None}
                        if tc['result']:
                            result_copy = dict(tc['result'])
                            sub_results = result_copy.get('sub_results', [])
                            if sub_results:
                                result_copy['sub_results'] = [
                                    {**sr, 'data': _sanitize_mcp_data(sr.get('data'))}
                                    for sr in sub_results
                                ]
                            sanitized_tc['result'] = result_copy
                        sanitized_mcp_calls.append(sanitized_tc)
                    st.session_state['_report_mcp_tool_calls'] = sanitized_mcp_calls
                    st.rerun()
                else:
                    report_area.error(full_report if full_report else "报告生成失败")

    elif st.session_state['generated_report']:
        report_data = st.session_state['generated_report']

        # --- 报告内容 ---
        with report_area.container():
            st.markdown("---")
            st.subheader(f"{report_data['type']} ({report_data['detail_level']})")
            st.caption(f"生成时间: {report_data.get('generated_at', 'N/A')}")

            st.markdown(report_data['content'])

            st.markdown("---")
            st.subheader("导出报告")

            report_generator = st.session_state['report_generator']
            export_content = report_generator.export_report(
                report_data['content'],
                report_data['type'],
                format='markdown'
            )
            export_bytes = export_content.encode('utf-8-sig')
            st.download_button(
                label="下载报告 (Markdown)",
                data=export_bytes,
                file_name=f"{report_data['type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                width='stretch',
            )

        # --- SQL 执行结果展示（放在报告内容下方） ---
        _report_sql_results = st.session_state.get('_report_sql_results', [])
        _report_mcp_tool_calls = st.session_state.get('_report_mcp_tool_calls', [])

        # MCP 模式：展示完整的工具执行路径
        if _report_mcp_tool_calls:
            with st.expander(f"MCP 执行路径（共 {len(_report_mcp_tool_calls)} 轮工具调用）", expanded=False):
                for idx, tc in enumerate(_report_mcp_tool_calls, 1):
                    call = tc['call']
                    result = tc['result']
                    round_num = call.get('round', idx)
                    st.markdown(f"#### 第 {idx} 轮（Round {round_num}）")
                    st.markdown(f"**工具名称：** `{call.get('tool_name', '')}`")
                    args = call.get('arguments', {})
                    if args:
                        questions = args.get('questions', [])
                        if questions:
                            st.markdown(f"**查询问题（{len(questions)} 条）：**")
                            for i, q in enumerate(questions, 1):
                                st.markdown(f"  {i}. {q}")
                        if args.get('batch_no'):
                            st.caption(f"批次: {args['batch_no']}")
                    if result:
                        status_icon = "✅" if result.get('status') == 'success' else "❌"
                        st.markdown(f"**执行结果：** {status_icon} {result.get('result_summary', '')}")
                        # 展示子请求详情
                        sub_results = result.get('sub_results', [])
                        if sub_results:
                            st.markdown("**子请求执行详情：**")
                            for si, sr in enumerate(sub_results, 1):
                                s_icon = "✅" if sr.get('status') == 'success' else "❌"
                                st.markdown(f"  **{si}. {s_icon} {sr.get('question', '')}**")
                                if sr.get('sql'):
                                    st.code(sr['sql'], language='sql')
                                if sr.get('was_fixed'):
                                    st.warning(f"⚠️ 经过 {sr.get('retry_count', 0)} 次自修复后成功")
                                    with st.expander("查看修复前的原始 SQL 和错误", expanded=False):
                                        st.markdown("**原始 SQL：**")
                                        st.code(sr['original_sql'], language='sql')
                                        st.markdown(f"**原始错误：** ")
                                        st.error(sr.get('original_error', ''))
                                elif sr.get('error'):
                                    st.error(f"错误: {sr['error']}")
                                    if sr.get('retry_count', 0) > 0:
                                        st.caption(f"（已尝试 {sr.get('retry_count', 0)} 次自修复均失败）")
                                if sr.get('status') == 'success':
                                    st.caption(f"返回 {sr.get('row_count', 0)} 行数据")
                                    # 嵌套 expander 展示查询到的数据（默认不展开）
                                    sr_data = sr.get('data')
                                    if sr_data:
                                        with st.expander(f"查看查询数据（{len(sr_data)} 行）", expanded=False):
                                            try:
                                                st.dataframe(pd.DataFrame(sr_data), width='stretch')
                                            except Exception:
                                                st.json(sr_data)
                    st.divider()

        # 非 MCP 模式：SQL 执行结果展示
        if _report_sql_results:
            with st.expander(f"查看 {len(_report_sql_results)} 条 SQL 执行结果", expanded=False):
                for r in _report_sql_results:
                    st.caption(f"SQL #{r['index']}/{r['total']}")
                    if r.get('was_fixed'):
                        st.success("此语句经过自动修复后成功")
                    st.code(r['sql'], language='sql')
                    if r['error']:
                        st.error(f"执行失败: {r['error']}")
                    else:
                        df = r.get('dataframe')
                        if df is not None and not df.empty:
                            st.dataframe(df, width='stretch')
                        else:
                            st.text(r['result'])
                    # 对经过修复的 SQL，展示原始 SQL 和原始错误
                    if r.get('was_fixed') and r.get('original_sql'):
                        with st.expander("查看修复前的原始 SQL 和错误", expanded=False):
                            st.markdown("**原始 SQL：**")
                            st.code(r['original_sql'], language='sql')
                            st.markdown(f"**原始错误：** ")
                            st.error(r.get('original_error', ''))
                    st.divider()


# ============================================================
# 路由
# ============================================================
PAGE_MAP = {
    "首页": page_home,
    "数据加载与清洗": page_cleaning,
    "数据分析": page_analysis,
    "数据可视化": page_visualization,
    "数据库管理": page_database,
    "智能查询": page_query,
    "智能客服": page_smart_assistant,
    "智能报告": page_report_generator,
}

PAGE_MAP[page]()
