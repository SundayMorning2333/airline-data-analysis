# 客户分析与智能查询系统

基于航空客户数据的智能分析平台，集成数据清洗、RFM 模型分析、K-Means 聚类、数据可视化、自然语言智能查询、智能客服与智能报告生成等功能，帮助深入理解客户行为，精准制定营销策略。

## 功能特性

### 数据分析流水线
- **数据加载与清洗**：支持 CSV 文件上传或使用默认数据集，提供缺失值处理、异常值检测（IQR / Z-Score）与处理策略，自动生成数据质量报告
- **RFM 模型分析**：基于最近消费（R）、消费频率（F）、消费金额（M）计算客户价值指标
- **K-Means 聚类分析**：肘部法自动推荐最佳聚类数，支持自定义 K 值、随机种子、初始化次数，自动映射客户分群标签（高价值客户、流失客户、潜力客户等）
- **数据可视化**：饼图、三维散点图、分组柱状图、相关性热力图、箱线图、雷达图、小提琴图、散点矩阵图共 8 种图表

### 数据库管理
- 基于 MySQL 存储分析结果，支持批次化管理
- 三张核心表：`rfm_analysis`（RFM 分析结果）、`customer_clusters`（客户分群）、`member_data`（原始客户数据）
- 使用 `dbutils.PooledDB` 实现连接池化，支持自动重连与健康检查
- 提供聚类结果入库、数据查询、批次删除、CSV 导出功能

### 智能查询
- **NL2SQL 自然语言查询**：输入自然语言问题，由 LLM 或规则引擎自动生成 SQL 并执行
- **三种查询模式**：LLM 智能模式、规则匹配模式（无需 API Key）、手动 SQL 模式
- **SQL 自修复**：执行失败时自动让 LLM 修复 SQL 语句（最多重试 2 次）
- 内置 60+ 常见查询示例，覆盖客户分群、人口统计、乘机行为、积分分析等场景

### 智能客服
- 基于大模型的对话式交互，支持流式输出
- 实时显示工具调用状态（生成 SQL 中、执行查询中、自修复中等）
- 预置 23 个快速提问模板（客户概况、高价值客户分析、流失预警等）
- 保留多轮对话历史，支持查询详情折叠查看

### 智能报告生成
- 四种报告类型：综合分析报告、客户画像报告、营销策略报告、运营优化报告
- 三种详细程度：摘要版、标准版、详细版
- 流式生成 Markdown 报告，支持下载导出
- 报告生成过程可视化，包含 SQL 执行结果展示

### MCP 工具调用机制
- 基于 MCP（Model Context Protocol）协议的工具调用服务
- 支持 1-5 条自然语言问题并发查询，显著提升性能
- 实时渲染执行轨迹：工具名称、查询问题列表、子请求 SQL、执行状态、行数、重试信息
- 执行轨迹持久化保存，历史消息中可折叠查看

## 技术栈

| 类别 | 技术 |
|------|------|
| 前端界面 | Streamlit 1.50.0 |
| 数据处理 | pandas、numpy、scikit-learn |
| 可视化 | matplotlib、seaborn、plotly |
| 数据库 | MySQL、pymysql、dbutils（连接池） |
| 大模型 | OpenAI 兼容 API（支持通义千问、DeepSeek、OpenAI 等） |
| 配置管理 | python-dotenv |

## 项目结构

```
.
├── app.py                      # Streamlit 主应用入口
├── requirements.txt            # Python 依赖
├── database_init.sql           # MySQL 数据库初始化脚本
├── .env                        # 环境变量配置（需自行创建）
├── .streamlit/
│   └── config.toml             # Streamlit 配置
├── config/
│   └── settings.py             # 全局配置（LLM、MySQL、清洗参数、报告类型等）
├── csv_data/
│   ├── air.csv                 # 默认航空客户数据集
│   └── generate_test_data.py   # 测试数据生成脚本
└── modules/
    ├── __init__.py
    ├── data_cleaner.py         # 数据清洗模块
    ├── data_analyzer.py        # RFM 分析与 K-Means 聚类
    ├── data_visualizer.py      # 数据可视化
    ├── database_manager.py     # MySQL 数据库管理（连接池）
    ├── nl2sql_query.py         # NL2SQL 查询引擎
    ├── mcp_tool_service.py     # MCP 工具调用服务
    ├── smart_assistant.py      # 智能客服
    └── report_generator.py     # 智能报告生成
```

## 快速开始

### 1. 环境准备

- Python 3.9+
- MySQL 8.0+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# LLM 配置（支持任何 OpenAI 兼容接口）
LLM_API_KEY=your-api-key-here
LLM_ENDPOINT=https://api.openai.com/v1
LLM_MODEL=gpt-3.5-turbo

# 通义千问示例
# LLM_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1

# DeepSeek 示例
# LLM_ENDPOINT=https://api.deepseek.com

# MySQL 配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=airline_analysis
MYSQL_CHARSET=utf8mb4
```

### 4. 初始化数据库

```bash
mysql -u root -p < database_init.sql
```

### 5. 启动应用

```bash
streamlit run app.py
```

## 使用流程

1. **数据加载与清洗**：进入「数据加载与清洗」页面，加载默认数据集或上传 CSV，配置清洗参数并执行
2. **数据分析**：进入「数据分析」页面，计算 RFM 指标，使用肘部法确定最佳 K 值，执行 K-Means 聚类
3. **数据可视化**：进入「数据可视化」页面，选择多种图表查看分析结果
4. **数据库管理**：进入「数据库管理」页面，自动连接 MySQL，将聚类结果入库（支持批次管理）
5. **智能查询**：通过自然语言或手动 SQL 查询已入库数据
6. **智能客服**：对话式交互分析数据，可开启 MCP 模式获得并发查询性能
7. **智能报告**：选择报告类型与详细程度，生成可下载的 Markdown 分析报告

## MCP 模式说明

智能客服与智能报告模块支持 MCP 模式开关：

- **关闭（默认）**：LLM 直接生成单条 SQL 串行查询
- **开启**：LLM 通过 MCP 工具调用机制，将复杂问题拆分为 1-5 条子查询并发执行，并支持 SQL 自修复

MCP 执行轨迹会实时展示，并持久化保存到对话历史中，便于回溯审查。

## 数据库表结构

- **rfm_analysis**：RFM 分析结果（会员编号、批次号、R/F/M 值、分析日期）
- **customer_clusters**：客户分群结果（会员编号、批次号、聚类标签、客户类型）
- **member_data**：原始客户数据（人口统计、乘机行为、积分消费等 40+ 字段）

所有表均以 `(member_no, batch_no)` 作为唯一键，支持多批次数据共存。

## 配置说明

核心配置位于 `config/settings.py`，包括：

- `DEFAULT_LLM_CONFIG`：LLM 模型、端点、API Key（优先在`.env`内配置该项，此处仅做回退）
- `MYSQL_CONFIG`：MySQL 连接参数
- `CLEANING_DEFAULTS` / `CLUSTERING_DEFAULTS`：清洗与聚类默认参数
- `REPORT_TYPES` / `REPORT_DETAIL_LEVELS`：报告类型与详细程度定义
- `MCP_CONFIG`：MCP 工具调用配置（并发数、重试次数、超时）
- `DB_POOL_CONFIG`：数据库连接池配置
- `CHAT_TEMPLATES`：智能客服快速提问模板
- `QUERY_DB_EXAMPLES`：智能查询示例问题
