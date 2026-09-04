"""
智能分析报告生成器模块
基于 LLM 将数据分析结果转化为结构化的商业洞察报告
支持：
1. 整合 RFM 分析和聚类分析结果
2. 数据库模式下通过 LLM 生成 SQL 进行全量数据综合分析
3. 生成多种类型的商业分析报告
4. 支持报告导出为多种格式
"""

import datetime
import json
import re
import queue
import threading
from typing import Dict, List, Optional, Any
import pandas as pd
from openai import OpenAI

from config.settings import (
    DEFAULT_LLM_CONFIG,
    REPORT_GENERATION_CONFIG,
    REPORT_TYPES,
    REPORT_DETAIL_LEVELS,
    REPORT_SQL_MAX_COUNT,
    REPORT_MCP_MAX_ROUNDS,
    REPORT_AUTO_FIX_SQL_ERRORS,
    REPORT_TOKEN_TO_CHAR_RATIO,
    MYSQL_VERSION,
)
from modules.nl2sql_query import NL2SQLQueryEngine


class ReportGenerator:
    """智能报告生成引擎，基于 LLM 生成结构化商业分析报告"""


    def __init__(self, api_key=None, endpoint=None, model=None):
        """
        初始化报告生成器

        Args:
            api_key: LLM API 密钥（可选，默认从配置读取）
            endpoint: API 端点 URL（可选，默认从配置读取）
            model: 模型名称（可选，默认从配置读取）
        """
        self.api_key = api_key or DEFAULT_LLM_CONFIG['api_key']
        self.endpoint = endpoint or DEFAULT_LLM_CONFIG['endpoint']
        self.model = model or DEFAULT_LLM_CONFIG['model']
        self.config = REPORT_GENERATION_CONFIG

        # 初始化 OpenAI 客户端
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
            except Exception:
                self.client = None

    def _generate_analysis_sqls(
        self,
        report_type: str,
        detail_level: str,
        batch_filter: str,
    ) -> List[str]:
        """
        让 LLM 根据报告类型生成多条 SQL 查询，对数据库全量数据进行综合分析。
        SQL 数量由 LLM 自行判断，不超过配置的最大值。

        Args:
            report_type: 报告类型
            detail_level: 详细程度
            batch_filter: 批次过滤条件（自然语言描述）

        Returns:
            SQL 查询语句列表
        """
        max_count = REPORT_SQL_MAX_COUNT.get(detail_level, 8)

        prompt = f"""你是一位资深的数据库分析专家。请根据以下数据库表结构，生成适量的 SQL 查询语句（最多 {max_count} 条），用于对客户数据进行全面分析，以支持生成一份"{report_type}"。

当前数据库版本: {MYSQL_VERSION}，请确保生成的 SQL 符合该版本的语法规则。

数据库表结构：
{NL2SQLQueryEngine.DB_TABLE_SCHEMA}

{'-' * 40}

所有查询必须应用以下批次过滤条件，确保同一批次的数据不会与其他批次合并：
{batch_filter}

请根据报告类型和详细程度（{detail_level}）自行判断需要生成多少条 SQL，但不得超过 {max_count} 条。

推荐的分析维度（根据报告类型选择最相关的）：
- 数据总体概览（总记录数、客户数、各表数量等）
- RFM 指标统计（R/F/M 的均值、标准差、中位数、四分位数、最大最小值）
- 客户分群分布（各客户类型的数量和占比）
- 各分群的 RFM 特征对比（按 customer_type 分组统计 R/F/M 均值）
- 高价值/低价值客户分析
- 分布特征分析（如 R 值分段统计、F 值分段统计等）
- RFM 各维度交叉分析（如高F高M客户数、高R低F客户数等）
- 客户价值分层、潜力客户识别、流失风险客户
- 原始数据概览（总客户数、性别分布、年龄分布、等级分布等）
- 乘机行为分析（乘机次数分布、飞行里程分布）
- 积分消费分析（积分分布、兑换行为、折扣率分析）

要求：
- 每条 SQL 只用 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP 等修改操作
- 每条 SQL 用独立的 ```sql ... ``` 代码块包裹
- SQL 之间用空行分隔
- {detail_level}：请自行判断合适的查询数量和粒度，宁少勿滥，确保每条 SQL 都有明确的分析目的

请开始生成 SQL 查询："""

        messages = [
            {"role": "system", "content": f"你是一位精于数据仓库和 SQL 的分析专家。当前数据库版本为 {MYSQL_VERSION}，请只输出 SQL 代码块，不要输出其他解释。"},
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(messages, temperature=0)

        # 提取所有 SQL 语句
        # 关键修复：单个代码块内可能包含多条以分号分隔的 SQL，
        # 必须按分号拆分，否则会被当作一条语句导致执行无限报错。
        sqls = self._extract_all_sql_from_response(response)
        return sqls

    def _extract_all_sql_from_response(self, response: str) -> List[str]:
        """
        从 LLM 响应中提取所有 SQL 语句。

        解析顺序：
        1. 优先提取所有 ```sql ... ``` / ``` ... ``` 代码块
        2. 若代码块不存在，回退到裸文本中提取 SELECT/WITH 语句

        关键修复：单个代码块内可能包含多条以分号分隔的 SQL 语句，
        必须按分号拆分，否则会被当作一条语句导致执行无限报错。

        Args:
            response: LLM 的响应文本

        Returns:
            SQL 语句列表（可能为空列表）
        """
        if not response:
            return []

        candidate_blocks = []

        # 1. 提取所有 markdown 代码块
        code_block_pattern = r'```(?:sql)?\s*\n?(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            candidate_blocks = [m.strip() for m in matches if m.strip()]

        # 2. 回退：从裸文本提取以 SELECT/WITH 开头的语句
        if not candidate_blocks:
            text = response
            text = re.sub(r'```sql\s*\n?', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text)
            parts = text.split(';')
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                cleaned = re.sub(r'^\s*--[^\n]*\n', '', part)
                cleaned = re.sub(r'^\s*/\*.*?\*/\s*', '', cleaned, flags=re.DOTALL)
                cleaned = cleaned.strip()
                m = re.search(r'\b(SELECT|WITH)\b', cleaned, re.IGNORECASE)
                if m:
                    candidate_blocks.append(cleaned[m.start():].strip())

        # 3. 对每个候选块按分号拆分为独立 SQL 语句
        sqls = []
        for block in candidate_blocks:
            block = re.sub(r'```sql\s*\n?', '', block, flags=re.IGNORECASE)
            block = block.replace('```', '').strip()
            parts = block.split(';')
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                cleaned = re.sub(r'^\s*--[^\n]*\n', '', part)
                cleaned = re.sub(r'^\s*/\*.*?\*/\s*', '', cleaned, flags=re.DOTALL)
                cleaned = cleaned.strip()
                if re.match(r'^(SELECT|WITH)\b', cleaned, re.IGNORECASE):
                    sqls.append(cleaned)

        return sqls

    def _execute_analysis_sql(self, db_manager, sql: str) -> tuple:
        """
        在数据库上执行一条分析 SQL 查询。

        Args:
            db_manager: DatabaseManager 实例
            sql: SQL 查询语句

        Returns:
            (result_df, error_message) 元组
        """
        if not db_manager.is_connected():
            return None, "数据库未连接"
        result_df, error_msg = db_manager.execute_query_safe(sql)
        if error_msg:
            return None, error_msg
        return result_df, ""

    def _format_sql_result(self, sql: str, df: Optional[pd.DataFrame], error: str) -> str:
        """
        将一条 SQL 查询结果格式化为报告可用的文本。

        Args:
            sql: 执行的 SQL 语句
            df: 查询结果 DataFrame
            error: 错误信息（如有）

        Returns:
            格式化的文本
        """
        if error:
            return f"SQL 执行失败 ({error}):\n```sql\n{sql}\n```"

        if df is None or df.empty:
            return f"查询结果为空:\n```sql\n{sql}\n```"

        max_rows = 30
        display_df = df.head(max_rows)
        text = f"查询结果（{len(df)} 行）：\n{display_df.to_string(index=False)}"
        if len(df) > max_rows:
            text += f"\n（共 {len(df)} 行，以上展示前 {max_rows} 行）"
        return text

    def _execute_sqls_and_collect(
        self, db_manager, sqls: List[str]
    ) -> str:
        """
        批量执行 SQL 查询并收集格式化的结果。

        Args:
            db_manager: DatabaseManager 实例
            sqls: SQL 查询列表

        Returns:
            所有查询结果的格式化文本
        """
        all_results = []
        for i, sql in enumerate(sqls, 1):
            df, error = self._execute_analysis_sql(db_manager, sql)
            result_text = self._format_sql_result(sql, df, error)
            all_results.append(f"### 分析查询 {i}\n{result_text}\n")
        return "\n".join(all_results)

    def _fix_sql_error(self, sql: str, error: str, batch_filter: str,
                       successful_sqls: Optional[List[str]] = None,
                       report_type: str = '',
                       fix_dialog: Optional[List[Dict]] = None) -> Optional[str]:
        """
        让 LLM 修复执行失败的 SQL 语句（带防串扰约束）。

        防串扰设计：
        1. 修复 prompt 中不携带 LLM 的完整多 SQL 响应，只提供当前出错 SQL
        2. 修复 prompt 明确指示只修复这一条 SQL，禁止返回其他 SQL
        3. 调用方应在获得修复 SQL 后自行校验是否与 successful_sqls 重复

        Args:
            sql: 原始 SQL 语句
            error: 错误信息
            batch_filter: 批次过滤条件（自然语言描述）
            successful_sqls: 已成功的 SQL 列表，用于 prompt 中告知 LLM 避免重复
            report_type: 报告类型（提供修复上下文）
            fix_dialog: 当前 SQL 的修复历史（多轮对话），由调用方维护

        Returns:
            修复后的 SQL 语句，如果 LLM 未能修复则返回 None
        """
        # 构建"已成功 SQL"的提示（只在列表非空时告知 LLM，不泄露完整 SQL 内容）
        avoid_hint = ""
        if successful_sqls:
            avoid_hint = (
                f"\n\n【防串扰约束】\n"
                f"此前已有 {len(successful_sqls)} 条 SQL 成功执行。"
                f"你输出的修复 SQL 必须针对当前出错的这条 SQL，"
                f"不要照搬或返回其他 SQL 语句。\n"
            )

        dialog_section = ""
        if fix_dialog:
            dialog_section = "\n\n【此前的修复尝试】\n"
            for msg in fix_dialog[-4:]:  # 最多保留最近 2 轮（4 条消息）
                role = "助手" if msg['role'] == 'assistant' else "用户"
                dialog_section += f"{role}: {msg['content'][:300]}\n"

        prompt = f"""以下 SQL 查询执行失败，请修复它。

数据库: {MYSQL_VERSION} (airline_analysis)
报告类型: {report_type}

批次过滤要求:
{batch_filter}

以下是数据库表结构定义，请据此分析错误原因（如列名拼写错误、类型不匹配、表关联条件缺失等）：
{NL2SQLQueryEngine.DB_TABLE_SCHEMA}

当前需要修复的 SQL（只修复这一条，不要返回其他 SQL）:
```sql
{sql}
```

错误信息:
{error}{avoid_hint}{dialog_section}

请分析错误原因并返回修复后的 SQL。
【重要约束】
1. 只返回修复后的这一条 SQL 语句，用 ```sql ... ``` 包裹
2. 修复后的 SQL 必须针对当前错误，不要照搬其他查询
3. 如果无法修复，请返回空内容"""

        messages = [
            {"role": "system", "content": f"你是一位 {MYSQL_VERSION} SQL 专家，擅长修复 SQL 语法错误和兼容性问题。只输出修复后的 SQL 代码块。严禁返回与问题无关的其他 SQL 语句。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._call_llm(messages, temperature=0)
        except Exception:
            return None

        # 提取修复后的 SQL（用统一的拆分逻辑，防止单代码块内多条 SQL）
        sqls = self._extract_all_sql_from_response(response)
        if sqls:
            return sqls[0]  # 只取第一条（修复应只返回一条）
        return None

    def _call_llm(self, messages, temperature=0):
        """
        调用 LLM API（非流式）
        """
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化，请检查 API Key 配置")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _call_llm_with_tools(self, messages, tools):
        """
        调用 LLM API（带工具定义，非流式）

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI function calling 格式）

        Returns:
            LLM 响应对象
        """
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化，请检查 API Key 配置")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice='auto',
            temperature=0,
        )
        return response

    def _collect_analysis_data(
        self,
        rfm_df: Optional[pd.DataFrame] = None,
        cluster_result: Optional[pd.DataFrame] = None,
        cluster_summary: Optional[pd.DataFrame] = None,
        clean_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        收集并整合分析数据

        Args:
            rfm_df: RFM 分析结果 DataFrame
            cluster_result: 聚类结果 DataFrame
            cluster_summary: 聚类汇总 DataFrame
            clean_data: 清洗后的原始数据 DataFrame

        Returns:
            整合后的分析数据字典
        """
        data = {
            'has_rfm': rfm_df is not None and not rfm_df.empty,
            'has_cluster': cluster_result is not None and not cluster_result.empty,
            'has_raw_data': clean_data is not None and not clean_data.empty,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 收集 RFM 数据统计
        if data['has_rfm']:
            rfm_stats = {
                'total_records': len(rfm_df),
                'r_mean': float(rfm_df['R'].mean()),
                'r_std': float(rfm_df['R'].std()),
                'r_min': float(rfm_df['R'].min()),
                'r_max': float(rfm_df['R'].max()),
                'f_mean': float(rfm_df['F'].mean()),
                'f_std': float(rfm_df['F'].std()),
                'f_min': float(rfm_df['F'].min()),
                'f_max': float(rfm_df['F'].max()),
                'm_mean': float(rfm_df['M'].mean()),
                'm_std': float(rfm_df['M'].std()),
                'm_min': float(rfm_df['M'].min()),
                'm_max': float(rfm_df['M'].max()),
            }
            data['rfm_stats'] = rfm_stats

        # 收集聚类数据统计
        if data['has_cluster']:
            # 客户分群统计
            if 'Customer_Label' in cluster_result.columns:
                label_counts = cluster_result['Customer_Label'].value_counts().to_dict()
                data['cluster_distribution'] = label_counts
                data['total_customers'] = len(cluster_result)
                data['num_clusters'] = len(label_counts)

            # 聚类汇总信息
            if cluster_summary is not None and not cluster_summary.empty:
                data['cluster_summary'] = cluster_summary.to_dict()

        # 收集原始数据统计
        if data['has_raw_data']:
            raw_stats = {
                'total_records': len(clean_data),
                'total_features': len(clean_data.columns),
                'date_range': {
                    'earliest_member': str(clean_data['FFP_DATE'].min()) if 'FFP_DATE' in clean_data.columns else 'N/A',
                    'latest_member': str(clean_data['FFP_DATE'].max()) if 'FFP_DATE' in clean_data.columns else 'N/A',
                },
            }

            # 性别分布
            if 'GENDER' in clean_data.columns:
                gender_dist = clean_data['GENDER'].value_counts().to_dict()
                raw_stats['gender_distribution'] = gender_dist

            # 年龄统计
            if 'AGE' in clean_data.columns:
                raw_stats['age_stats'] = {
                    'mean': float(clean_data['AGE'].mean()),
                    'min': float(clean_data['AGE'].min()),
                    'max': float(clean_data['AGE'].max()),
                }

            data['raw_stats'] = raw_stats

        return data

    def _build_report_context(
        self,
        analysis_data: Dict[str, Any],
        report_type: str,
        detail_level: str,
    ) -> str:
        """
        构建报告生成的上下文信息

        Args:
            analysis_data: 分析数据字典
            report_type: 报告类型
            detail_level: 详细程度

        Returns:
            格式化的上下文文本
        """
        context_parts = []

        # 添加报告元信息
        context_parts.append(f"报告类型: {report_type}")
        context_parts.append(f"详细程度: {detail_level}")
        context_parts.append(f"生成时间: {analysis_data.get('timestamp', 'N/A')}")
        context_parts.append("")

        # 添加数据概览
        context_parts.append("## 数据概览")
        if analysis_data.get('has_raw_data'):
            raw_stats = analysis_data.get('raw_stats', {})
            context_parts.append(f"- 总记录数: {raw_stats.get('total_records', 'N/A'):,}")
            context_parts.append(f"- 特征数量: {raw_stats.get('total_features', 'N/A')}")

            date_range = raw_stats.get('date_range', {})
            context_parts.append(f"- 入会时间范围: {date_range.get('earliest_member', 'N/A')} 至 {date_range.get('latest_member', 'N/A')}")

            gender_dist = raw_stats.get('gender_distribution', {})
            if gender_dist:
                gender_str = ", ".join([f"{k}: {v}" for k, v in gender_dist.items()])
                context_parts.append(f"- 性别分布: {gender_str}")

            age_stats = raw_stats.get('age_stats', {})
            if age_stats:
                context_parts.append(f"- 年龄范围: {age_stats.get('min', 'N/A'):.0f} - {age_stats.get('max', 'N/A'):.0f} 岁，平均: {age_stats.get('mean', 'N/A'):.1f} 岁")
        else:
            context_parts.append("- 原始数据: 未加载")
        context_parts.append("")

        # 添加 RFM 分析结果
        if analysis_data.get('has_rfm'):
            context_parts.append("## RFM 分析结果")
            rfm_stats = analysis_data.get('rfm_stats', {})
            context_parts.append(f"- 分析记录数: {rfm_stats.get('total_records', 'N/A'):,}")
            context_parts.append(f"- R 值（最近消费距今天数）: 均值={rfm_stats.get('r_mean', 0):.1f}, 标准差={rfm_stats.get('r_std', 0):.1f}, 范围=[{rfm_stats.get('r_min', 0):.0f}, {rfm_stats.get('r_max', 0):.0f}]")
            context_parts.append(f"- F 值（消费频率）: 均值={rfm_stats.get('f_mean', 0):.1f}, 标准差={rfm_stats.get('f_std', 0):.1f}, 范围=[{rfm_stats.get('f_min', 0):.0f}, {rfm_stats.get('f_max', 0):.0f}]")
            context_parts.append(f"- M 值（消费金额/里程）: 均值={rfm_stats.get('m_mean', 0):.1f}, 标准差={rfm_stats.get('m_std', 0):.1f}, 范围=[{rfm_stats.get('m_min', 0):.0f}, {rfm_stats.get('m_max', 0):.0f}]")
            context_parts.append("")

        # 添加聚类分析结果
        if analysis_data.get('has_cluster'):
            context_parts.append("## 客户分群分析结果")
            context_parts.append(f"- 总客户数: {analysis_data.get('total_customers', 'N/A'):,}")
            context_parts.append(f"- 分群数量: {analysis_data.get('num_clusters', 'N/A')}")

            cluster_dist = analysis_data.get('cluster_distribution', {})
            if cluster_dist:
                context_parts.append("- 各群体规模:")
                for label, count in sorted(cluster_dist.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / analysis_data.get('total_customers', 1)) * 100
                    context_parts.append(f"  - {label}: {count:,} 人 ({percentage:.1f}%)")
            context_parts.append("")

        return "\n".join(context_parts)

    def _apply_template(
        self,
        report_type: str,
        detail_level: str,
        context: str,
    ) -> str:
        """
        应用报告模板生成 prompt

        Args:
            report_type: 报告类型
            detail_level: 详细程度
            context: 数据上下文

        Returns:
            完整的 prompt 文本
        """
        report_config = REPORT_TYPES.get(report_type, REPORT_TYPES["综合分析报告"])
        detail_config = REPORT_DETAIL_LEVELS.get(detail_level, REPORT_DETAIL_LEVELS["标准版"])

        # 获取对应详细程度的sections
        sections_by_detail = report_config.get("sections", {})
        sections = sections_by_detail.get(detail_level, sections_by_detail.get("标准版", []))

        # 根据详细程度调整生成要求
        # 动态计算字数限制：token_limit * 转换比例
        token_limit = detail_config.get('token_limit', 4000)
        char_limit = int(token_limit * REPORT_TOKEN_TO_CHAR_RATIO)

        if detail_level == "摘要版":
            style_requirement = "简洁精炼，突出关键数据和核心发现，每个部分控制在2-3句话"
            data_requirement = "仅使用最关键的统计数据支撑观点"
        elif detail_level == "详细版":
            style_requirement = "全面深入，包含详细的数据分析和多角度解读"
            data_requirement = "充分利用所有可用数据，提供详细的数据表格和对比分析"
        else:  # 标准版
            style_requirement = "平衡详细度和可读性，重点突出"
            data_requirement = "使用关键数据支撑分析，提供必要的数据说明"

        length_requirement = f"报告总字数控制在 {char_limit} 字以内"

        prompt = f"""你是一位资深的航空业数据分析专家和商业顾问。请基于以下客户分析数据，生成一份专业的{report_type}。

【数据上下文】
{context}

【报告要求】
1. 报告类型: {report_type}
2. 报告详细程度: {detail_level} - {detail_config['description']}
3. 写作风格: {style_requirement}
4. 数据使用: {data_requirement}
5. 字数限制: {length_requirement}
6. 请使用 Markdown 格式输出报告
7. 报告应包含以下部分（按顺序）：
"""
        for i, section in enumerate(sections, 1):
            prompt += f"   {i}. {section}\n"

        prompt += f"""
8. 每个部分应有清晰的标题和结构
9. 使用专业但易懂的语言，适合业务决策者阅读
10. 重点关注业务价值和可操作性
11. 如果数据不足以支撑某个部分的分析，请如实说明

请开始生成报告："""

        return prompt

    def generate_report(
        self,
        report_type: str = "综合分析报告",
        detail_level: str = "标准版",
        rfm_df: Optional[pd.DataFrame] = None,
        cluster_result: Optional[pd.DataFrame] = None,
        cluster_summary: Optional[pd.DataFrame] = None,
        clean_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        生成智能分析报告

        Args:
            report_type: 报告类型（综合分析报告/客户画像报告/营销策略报告/运营优化报告）
            detail_level: 详细程度（摘要版/标准版/详细版）
            rfm_df: RFM 分析结果 DataFrame
            cluster_result: 聚类结果 DataFrame
            cluster_summary: 聚类汇总 DataFrame
            clean_data: 清洗后的原始数据 DataFrame

        Returns:
            包含报告内容和状态的字典
        """
        if self.client is None:
            return {
                'status': 'error',
                'report': None,
                'error': 'LLM 客户端未初始化，请检查 API Key 配置',
            }

        try:
            # 收集分析数据
            analysis_data = self._collect_analysis_data(
                rfm_df=rfm_df,
                cluster_result=cluster_result,
                cluster_summary=cluster_summary,
                clean_data=clean_data,
            )

            # 检查是否有足够的数据生成报告
            if not analysis_data.get('has_rfm') and not analysis_data.get('has_cluster'):
                return {
                    'status': 'error',
                    'report': None,
                    'error': '缺少分析数据，请先完成 RFM 分析和聚类分析',
                }

            # 构建上下文
            context = self._build_report_context(analysis_data, report_type, detail_level)

            # 应用模板生成 prompt
            prompt = self._apply_template(report_type, detail_level, context)

            # 调用 LLM 生成报告
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位资深的数据分析专家，擅长将数据转化为商业洞察。当前数据库版本为 {MYSQL_VERSION}。请用专业的语言生成分析报告。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config['temperature'],
                max_tokens=REPORT_DETAIL_LEVELS.get(detail_level, {}).get('token_limit', self.config.get('max_tokens', 4000)),
                timeout=self.config['timeout'],
            )

            report_content = response.choices[0].message.content

            return {
                'status': 'success',
                'report': report_content,
                'report_type': report_type,
                'detail_level': detail_level,
                'generated_at': analysis_data.get('timestamp'),
                'error': None,
            }

        except Exception as e:
            return {
                'status': 'error',
                'report': None,
                'error': f'报告生成失败: {str(e)}',
            }

    def generate_report_stream(
        self,
        report_type: str = "综合分析报告",
        detail_level: str = "标准版",
        db_manager=None,
        batch_no: Optional[str] = None,
        enable_mcp: bool = False,
    ):
        """
        以流式方式生成智能分析报告（基于 MySQL 数据库）

        Args:
            report_type: 报告类型
            detail_level: 详细程度
            db_manager: DatabaseManager 实例
            batch_no: 批次编号过滤条件
            enable_mcp: 是否启用 MCP 模式（LLM 通过 function calling 调用工具查询数据）

        Yields:
            dict: 包含以下类型的事件：
                - {'type': 'status', 'content': str} - 状态信息
                - {'type': 'sql', 'sqls': List[str]} - LLM 生成的分析 SQL 列表
                - {'type': 'sql_result', 'index': int, 'total': int, 'sql': str, 'result': str, 'dataframe': DataFrame, 'error': str} - 单条 SQL 执行结果
                - {'type': 'tool_call', 'tool_name': str, 'arguments': dict, 'round': int} - MCP 模式工具调用事件
                - {'type': 'tool_result', 'tool_name': str, 'result_summary': str, 'round': int, 'status': str, 'sub_results': list} - MCP 模式工具执行结果事件
                  sub_results 中每项含: question, sql, status, row_count, error, was_fixed, original_sql, original_error, retry_count
                - {'type': 'chunk', 'content': str} - 流式文本片段
                - {'type': 'done', 'report': str, 'status': str, ...} - 完成事件
        """
        if self.client is None:
            yield {'type': 'chunk', 'content': 'LLM 客户端未初始化，请检查 API Key 配置'}
            yield {'type': 'done', 'report': None, 'status': 'error'}
            return

        # ---------- MCP 模式 ----------
        if enable_mcp:
            yield from self._generate_report_mcp(report_type, detail_level, db_manager, batch_no)
            return

        try:
            yield {'type': 'status', 'content': '正在准备数据分析...'}

            # 通过 LLM 生成 SQL 进行全量数据综合分析
            if db_manager is None or batch_no is None:
                yield {'type': 'chunk', 'content': '请先连接数据库并选择分析批次'}
                yield {'type': 'done', 'report': None, 'status': 'error'}
                return

            yield {'type': 'status', 'content': '正在通过 LLM 生成数据库分析 SQL...'}

            # 构建批次过滤条件
            batch_filter = (
                f"必须仅分析 batch_no = '{batch_no}' 的数据，"
                f"不同批次的数据不得合并分析。"
                f"表 rfm_analysis、customer_clusters 和 member_data 都需要用 batch_no = '{batch_no}' 过滤。"
            )

            # 让 LLM 生成分析 SQL
            sqls = self._generate_analysis_sqls(report_type, detail_level, batch_filter)

            if not sqls:
                yield {'type': 'chunk', 'content': 'LLM 未能生成有效的分析 SQL，请重试'}
                yield {'type': 'done', 'report': None, 'status': 'error'}
                return

            # 将 SQL 列表发送给 UI 展示
            yield {'type': 'sql', 'sqls': sqls}
            yield {'type': 'status', 'content': f'LLM 已生成 {len(sqls)} 条分析 SQL，正在执行...'}

            # 逐条执行 SQL 并逐个发送结果给 UI（失败时自动让 LLM 修复）
            # 防串扰：严格串行执行，维护已成功 SQL 列表
            all_results_parts = []
            successful_sqls = []  # 已成功的 SQL 列表，用于防串扰校验

            for i, sql in enumerate(sqls, 1):
                yield {'type': 'status', 'content': f'正在执行第 {i}/{len(sqls)} 条 SQL 查询...'}

                # 第一次执行
                df, error = self._execute_analysis_sql(db_manager, sql)
                was_fixed = False
                display_sql = sql  # 用于界面展示的 SQL（修复成功时替换为修复后的语句）
                original_sql = sql  # 首次执行的原始 SQL（用于展示修复前的 SQL）
                original_error = ''  # 首次执行的原始错误（用于展示修复前的错误）

                # 失败时自动尝试修复（最多2次，严格串行 + 防串扰）
                if error and REPORT_AUTO_FIX_SQL_ERRORS:
                    original_error = error
                    current_sql = sql
                    # 当前 SQL 的修复对话历史（隔离，不跨 SQL）
                    fix_dialog = []

                    for attempt in range(1, 3):
                        yield {'type': 'status', 'content': f'正在执行第 {i}/{len(sqls)} 条 SQL 查询...（运行出现错误，正在尝试修复，第{attempt}/2次）'}
                        fixed_sql = self._fix_sql_error(
                            current_sql, error, batch_filter,
                            successful_sqls=successful_sqls,
                            report_type=report_type,
                            fix_dialog=fix_dialog,
                        )
                        if not fixed_sql:
                            break

                        # 防串扰校验：修复结果不能与已成功的 SQL 相同
                        if fixed_sql in successful_sqls:
                            fix_dialog.append({
                                'role': 'assistant',
                                'content': f"```sql\n{fixed_sql}\n```",
                            })
                            fix_dialog.append({
                                'role': 'user',
                                'content': (
                                    '修复后的 SQL 与此前已成功执行的某条 SQL 完全相同，'
                                    '这是不允许的。请生成一条不同的、针对当前错误的 SQL。'
                                ),
                            })
                            error = "修复结果与已成功 SQL 重复（串扰），已拒绝"
                            continue

                        # 防串扰校验：修复结果也不能与当前 SQL 完全相同（无意义）
                        if fixed_sql == current_sql:
                            fix_dialog.append({
                                'role': 'assistant',
                                'content': f"```sql\n{fixed_sql}\n```",
                            })
                            fix_dialog.append({
                                'role': 'user',
                                'content': '修复后的 SQL 与原 SQL 完全相同，请分析错误并生成不同的修复方案。',
                            })
                            continue

                        # 采纳修复后的 SQL，执行
                        df, error = self._execute_analysis_sql(db_manager, fixed_sql)
                        if not error:
                            was_fixed = True
                            display_sql = fixed_sql
                            break

                        # 修复后仍失败，记录对话历史并继续
                        current_sql = fixed_sql
                        fix_dialog.append({
                            'role': 'assistant',
                            'content': f"```sql\n{fixed_sql}\n```",
                        })
                        fix_dialog.append({
                            'role': 'user',
                            'content': f'SQL 执行出错：{error}',
                        })

                # 记录成功的 SQL（供后续 SQL 防串扰校验）
                if not error and display_sql:
                    successful_sqls.append(display_sql)

                result_text = self._format_sql_result(display_sql, df, error)
                if was_fixed:
                    result_text += f"\n(经过自动修复后成功)"

                all_results_parts.append(f"### 分析查询 {i}\n{result_text}\n")

                # 发送单条 SQL 执行结果，供 UI 实时展示
                yield {
                    'type': 'sql_result',
                    'index': i,
                    'total': len(sqls),
                    'sql': display_sql,
                    'result': result_text,
                    'dataframe': df,
                    'error': error,
                    'was_fixed': was_fixed,
                    'original_sql': original_sql if was_fixed else None,
                    'original_error': original_error if was_fixed else '',
                }

            sql_results = "\n".join(all_results_parts)
            yield {'type': 'status', 'content': f'已完成全部 {len(sqls)} 条 SQL 查询'}

            # 直接用 SQL 查询结果作为报告的数据上下文
            context = f"""【数据库全量分析结果】
数据来源: {MYSQL_VERSION} (airline_analysis)
分析批次: {batch_no}
报告类型: {report_type}

{sql_results}
"""
            yield {'type': 'status', 'content': '正在生成报告...'}

            # 应用模板生成 prompt
            prompt = self._apply_template(report_type, detail_level, context)

            # 调用 LLM 流式生成报告
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位资深的数据分析专家，擅长将 SQL 查询结果转化为商业洞察。当前数据库版本为 {MYSQL_VERSION}。请用专业的语言生成分析报告。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config['temperature'],
                max_tokens=REPORT_DETAIL_LEVELS.get(detail_level, {}).get('token_limit', self.config.get('max_tokens', 4000)),
                timeout=self.config['timeout'],
                stream=True,
            )

            report_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    report_content += content
                    yield {'type': 'chunk', 'content': content}

            yield {
                'type': 'done',
                'report': report_content,
                'report_type': report_type,
                'detail_level': detail_level,
                'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'success',
            }

        except Exception as e:
            yield {'type': 'chunk', 'content': f'报告生成失败: {str(e)}'}
            yield {'type': 'done', 'report': None, 'status': 'error'}

    def _generate_report_mcp(self, report_type, detail_level, db_manager, batch_no):
        """
        MCP 模式生成报告：LLM 通过 function calling 调用 database_query 工具获取分析数据。

        Args:
            report_type: 报告类型
            detail_level: 详细程度
            db_manager: DatabaseManager 实例
            batch_no: 批次编号

        Yields:
            dict: 事件类型（与 generate_report_stream 一致，额外增加 tool_call/tool_result）
        """
        from modules.mcp_tool_service import MCPToolService

        try:
            if db_manager is None or not db_manager.is_connected():
                yield {'type': 'chunk', 'content': '请先连接数据库'}
                yield {'type': 'done', 'report': None, 'status': 'error'}
                return

            yield {'type': 'status', 'content': 'MCP 模式：正在通过 LLM 规划数据查询...'}

            # 创建 MCP 工具服务
            tool_call_limit = REPORT_MCP_MAX_ROUNDS.get(detail_level, 3)
            mcp_service = MCPToolService(
                db_manager=db_manager,
                api_key=self.api_key,
                endpoint=self.endpoint,
                model=self.model,
                call_limit=tool_call_limit,
            )

            tool_defs = mcp_service.get_tool_definitions()

            # 构建批次过滤提示
            batch_filter_hint = ""
            if batch_no:
                batch_filter_hint = f"\n当前分析批次为 {batch_no}，所有查询应限定在该批次。调用工具时请传入 batch_no 参数。"

            # MCP 模式系统提示
            mcp_system_prompt = f"""你是一位资深的航空业数据分析专家和商业顾问。你需要生成一份“{report_type}”（{detail_level}）。

你可以调用 database_query 工具来查询数据库获取所需的分析数据。工具接受 1-5 条自然语言问题，会自动转换为 SQL 并执行查询。

数据库包含以下表：
- rfm_analysis: RFM 分析结果（R/F/M 值）
- customer_clusters: 客户分群结果（客户类型）
- member_data: 原始客户数据（人口统计、乘机行为、积分消费）

工具使用规则：
1. 请规划需要查询的数据维度，合理分批调用工具（每次最多 5 个问题）
2. 最多调用工具 {tool_call_limit} 次，请合理规划查询内容
3. 获取数据后，系统会基于查询结果生成报告
{batch_filter_hint}

请先调用工具获取生成报告所需的分析数据。"""

            messages = [
                {"role": "system", "content": mcp_system_prompt},
                {"role": "user", "content": f"请为“{report_type}”（{detail_level}）查询所需的分析数据。建议覆盖：数据总览、RFM 统计、客户分群分布、各分群特征对比等维度。"},
            ]

            # 多轮工具调用循环
            all_data_context_parts = []
            round_num = 0

            while round_num < tool_call_limit:
                round_num += 1

                response = self._call_llm_with_tools(messages, tool_defs)
                message = response.choices[0].message

                if not message.tool_calls:
                    # LLM 不再需要查询，退出循环
                    break

                # 记录 assistant 的 tool_calls
                messages.append({
                    'role': 'assistant',
                    'content': message.content or '',
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments,
                            }
                        } for tc in message.tool_calls
                    ]
                })

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}

                    if 'batch_no' not in arguments and batch_no:
                        arguments['batch_no'] = batch_no

                    yield {
                        'type': 'tool_call',
                        'tool_name': tool_name,
                        'arguments': arguments,
                        'round': round_num,
                    }
                    yield {'type': 'status', 'content': f'正在执行工具 {tool_name} 获取数据（第 {round_num} 轮）...'}

                    # 使用队列+线程执行工具，实时 yield 子请求进度
                    progress_queue = queue.Queue()

                    def _progress_cb(info):
                        progress_queue.put(info)

                    def _run_tool():
                        try:
                            res = mcp_service.call_tool(tool_name, arguments, progress_callback=_progress_cb)
                            progress_queue.put(('__done__', res))
                        except Exception as e:
                            progress_queue.put(('__error__', str(e)))

                    tool_thread = threading.Thread(target=_run_tool, daemon=True)
                    tool_thread.start()

                    tool_result = None
                    while True:
                        try:
                            item = progress_queue.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        if isinstance(item, tuple) and item[0] == '__done__':
                            tool_result = item[1]
                            break
                        elif isinstance(item, tuple) and item[0] == '__error__':
                            tool_result = {
                                'tool_name': tool_name,
                                'status': 'error',
                                'result': None,
                                'error': item[1],
                            }
                            break
                        else:
                            yield {'type': 'tool_progress', 'info': item, 'round': round_num}

                    tool_thread.join(timeout=1)

                    result_text = mcp_service.format_tool_result_for_llm(tool_result)

                    # 提取子请求详情（每条问题的 SQL、状态、重试信息等）
                    sub_results = []
                    if tool_result.get('status') == 'success':
                        sub_results = tool_result.get('result', {}).get('results', [])

                    yield {
                        'type': 'tool_result',
                        'tool_name': tool_name,
                        'result_summary': tool_result.get('result', {}).get('summary', '') if tool_result.get('status') == 'success' else tool_result.get('error', ''),
                        'round': round_num,
                        'status': tool_result['status'],
                        'sub_results': sub_results,
                    }

                    # 收集数据上下文
                    all_data_context_parts.append(result_text)

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': result_text,
                    })

            yield {'type': 'status', 'content': f'已完成数据查询（共 {round_num} 轮），正在生成报告...'}

            # 构建 SQL 结果展示（兼容原有 UI 的事件格式）
            sql_results_text = "\n\n".join(all_data_context_parts) if all_data_context_parts else "无查询数据"

            # 构建 context
            context = f"""【数据库全量分析结果（MCP 模式）】
数据来源: {MYSQL_VERSION} (airline_analysis)
分析批次: {batch_no or '全部'}
报告类型: {report_type}

{sql_results_text}
"""

            # 应用模板生成 prompt
            prompt = self._apply_template(report_type, detail_level, context)

            # 流式生成报告
            yield {'type': 'status', 'content': '正在生成报告...'}

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位资深的数据分析专家，擅长将查询结果转化为商业洞察。当前数据库版本为 {MYSQL_VERSION}。请用专业的语言生成分析报告。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config['temperature'],
                max_tokens=REPORT_DETAIL_LEVELS.get(detail_level, {}).get('token_limit', self.config.get('max_tokens', 4000)),
                timeout=self.config['timeout'],
                stream=True,
            )

            report_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    report_content += content
                    yield {'type': 'chunk', 'content': content}

            yield {
                'type': 'done',
                'report': report_content,
                'report_type': report_type,
                'detail_level': detail_level,
                'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'success',
            }

        except Exception as e:
            yield {'type': 'chunk', 'content': f'报告生成失败: {str(e)}'}
            yield {'type': 'done', 'report': None, 'status': 'error'}

    def export_report(
        self,
        report_content: str,
        report_type: str,
        format: str = 'markdown',
    ) -> str:
        """
        导出报告为指定格式

        Args:
            report_content: 报告内容（Markdown 格式）
            report_type: 报告类型
            format: 导出格式（markdown/text/html）

        Returns:
            导出的内容字符串
        """
        if format == 'markdown':
            # 添加报告头信息
            header = f"""# {report_type}

**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
            return header + report_content

        elif format == 'text':
            # 转换为纯文本（移除 Markdown 格式）
            import re
            # 移除标题标记
            text = re.sub(r'^#+\s+', '', report_content, flags=re.MULTILINE)
            # 移除加粗标记
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            # 移除斜体标记
            text = re.sub(r'\*(.*?)\*', r'\1', text)
            # 移除代码块
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            # 移除行内代码
            text = re.sub(r'`(.*?)`', r'\1', text)
            # 移除链接
            text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
            # 移除图片
            text = re.sub(r'!\[(.*?)\]\(.*?\)', r'[图片: \1]', text)
            # 移除列表标记
            text = re.sub(r'^\s*[-*+]\s+', '  • ', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*\d+\.\s+', '  ', text, flags=re.MULTILINE)
            # 移除水平线
            text = re.sub(r'^---+$', '─' * 40, text, flags=re.MULTILINE)
            # 清理多余空行
            text = re.sub(r'\n{3,}', '\n\n', text)

            header = f"{report_type}\n生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'═' * 40}\n\n"
            return header + text.strip()

        elif format == 'html':
            # 简单的 Markdown 转 HTML
            import re
            html = report_content

            # 转换标题
            html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
            html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
            html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

            # 转换加粗
            html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
            # 转换斜体
            html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
            # 转换代码块
            html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
            # 转换行内代码
            html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
            # 转换列表
            html = re.sub(r'^\s*[-*+]\s+(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
            # 转换段落
            html = re.sub(r'\n\n', '</p><p>', html)
            # 转换水平线
            html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)
            # 包装在 HTML 结构中
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{report_type}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        strong {{ color: #2c3e50; }}
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        code {{ background-color: #f8f9fa; padding: 2px 5px; border-radius: 3px; }}
        li {{ margin-bottom: 5px; }}
        hr {{ border: none; border-top: 1px solid #bdc3c7; margin: 20px 0; }}
        p {{ margin-bottom: 15px; }}
    </style>
</head>
<body>
    <p>{html}</p>
</body>
</html>"""
            return html

        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def is_llm_available(self) -> bool:
        """
        检查 LLM 是否可用

        Returns:
            bool: API Key 已配置且客户端初始化成功时返回 True
        """
        return self.client is not None and bool(self.api_key)
