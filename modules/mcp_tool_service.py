"""
MCP 工具调用服务模块
提供基于 MCP（Model Context Protocol）风格的工具调用机制，
让 LLM 通过 function calling 调用预定义工具来增强数据查询能力。

当前提供的工具：
- database_query: 自然语言转 SQL 并发查询工具（1-5 条问题，并发数 5）
"""

import json
import logging
from typing import Any, Dict, List, Optional

from config.settings import MCP_CONFIG, DEFAULT_LLM_CONFIG
from modules.nl2sql_query import NL2SQLQueryEngine

logger = logging.getLogger(__name__)


class MCPToolService:
    """
    MCP 工具调用服务。

    管理工具注册、调用计数、并发执行与异常处理。
    每次 SmartAssistant / ReportGenerator 会话创建一个新实例，
    以隔离工具调用计数。
    """

    # 工具名称常量
    TOOL_DATABASE_QUERY = "database_query"

    def __init__(self, db_manager=None, api_key=None, endpoint=None, model=None, call_limit=None):
        """
        初始化 MCP 工具服务。

        Args:
            db_manager: DatabaseManager 实例（已连接）
            api_key: LLM API 密钥（可选，默认从配置读取）
            endpoint: LLM API 端点（可选）
            model: LLM 模型名称（可选）
            call_limit: 工具调用次数上限（可选，默认从 MCP_CONFIG 读取）
        """
        self.db_manager = db_manager
        self.api_key = api_key or DEFAULT_LLM_CONFIG.get('api_key', '')
        self.endpoint = endpoint or DEFAULT_LLM_CONFIG.get('endpoint', '')
        self.model = model or DEFAULT_LLM_CONFIG.get('model', '')

        # 工具调用计数器（单次会话）
        self._call_count = 0
        self._call_limit = call_limit if call_limit is not None else MCP_CONFIG.get('tool_call_limit', 20)

        # NL2SQL 引擎实例（延迟初始化）
        self._nl2sql_engine: Optional[NL2SQLQueryEngine] = None

        # 工具注册表
        self._tools = {
            self.TOOL_DATABASE_QUERY: {
                'executor': self._execute_database_query,
                'definition': self._get_database_query_definition(),
            },
        }

    # ================================================================
    # NL2SQL 引擎管理
    # ================================================================
    def _get_engine(self) -> NL2SQLQueryEngine:
        """获取或创建 NL2SQLQueryEngine 实例（延迟初始化）。"""
        if self._nl2sql_engine is None:
            self._nl2sql_engine = NL2SQLQueryEngine(
                db_manager=self.db_manager,
                api_key=self.api_key,
                endpoint=self.endpoint,
                model=self.model,
            )
        return self._nl2sql_engine

    # ================================================================
    # 工具定义（OpenAI function calling 格式）
    # ================================================================
    def _get_database_query_definition(self) -> Dict[str, Any]:
        """返回 database_query 工具的 OpenAI function calling 定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.TOOL_DATABASE_QUERY,
                "description": (
                    "查询航空客户分析数据库。输入1到5条自然语言问题，"
                    "工具会自动将每条问题转换为SQL并执行查询，返回每条问题的SQL语句和查询结果。"
                    "适用于：客户分群统计、RFM指标分析、原始客户数据查询、跨表关联分析等。"
                    "一次调用可包含多个相关问题以提高效率。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 5,
                            "description": "1到5条自然语言问题，例如：[\"各客户类型有多少客户\", \"平均消费金额是多少\"]",
                        },
                        "batch_no": {
                            "type": "string",
                            "description": "数据批次编号（可选）。如不指定则查询所有批次数据。",
                        },
                    },
                    "required": ["questions"],
                },
            },
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册工具的定义（OpenAI function calling 格式）。

        Returns:
            工具定义列表，可直接传入 OpenAI API 的 tools 参数
        """
        return [tool['definition'] for tool in self._tools.values()]

    # ================================================================
    # 工具调用调度
    # ================================================================
    def call_tool(self, tool_name: str, arguments: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """
        调用指定工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数（dict）
            progress_callback: 可选的进度回调函数，接收 dict 参数

        Returns:
            dict: {
                'tool_name': str,
                'status': 'success' | 'error',
                'result': Any,     # 工具返回结果
                'error': str,      # 空字符串表示无错误
            }
        """
        # 检查工具是否存在
        if tool_name not in self._tools:
            return {
                'tool_name': tool_name,
                'status': 'error',
                'result': None,
                'error': f"未知工具: {tool_name}",
            }

        # 检查调用次数上限
        if self._call_count >= self._call_limit:
            return {
                'tool_name': tool_name,
                'status': 'error',
                'result': None,
                'error': (
                    f"已达到工具调用次数上限（{self._call_limit} 次），"
                    "请基于已获取的数据直接回答用户问题。"
                ),
            }

        # 递增计数
        self._call_count += 1

        # 执行工具
        try:
            executor = self._tools[tool_name]['executor']
            result = executor(arguments, progress_callback=progress_callback)
            return {
                'tool_name': tool_name,
                'status': 'success',
                'result': result,
                'error': '',
            }
        except Exception as e:
            logger.exception(f"MCP 工具 {tool_name} 执行异常")
            return {
                'tool_name': tool_name,
                'status': 'error',
                'result': None,
                'error': f"工具执行异常: {str(e)}",
            }

    def get_call_count(self) -> int:
        """获取当前会话工具调用次数。"""
        return self._call_count

    def reset_call_count(self):
        """重置工具调用计数器。"""
        self._call_count = 0

    # ================================================================
    # database_query 工具实现
    # ================================================================
    def _execute_database_query(self, arguments: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """
        执行 database_query 工具：自然语言转 SQL 并发查询。

        Args:
            arguments: {
                'questions': list[str],  # 1-5 条自然语言问题
                'batch_no': str,         # 可选批次编号
            }

        Returns:
            dict: {
                'questions_count': int,
                'results': list[dict],  # 每条问题的查询结果
                'summary': str,         # 结果摘要文本
            }
        """
        questions = arguments.get('questions', [])
        batch_no = arguments.get('batch_no')

        # 参数校验
        if not questions or not isinstance(questions, list):
            return {
                'questions_count': 0,
                'results': [],
                'summary': "错误：questions 参数必须是非空数组",
            }

        # 限制 1-5 条
        if len(questions) > 5:
            questions = questions[:5]
        if len(questions) < 1:
            return {
                'questions_count': 0,
                'results': [],
                'summary': "错误：至少需要 1 条问题",
            }

        # 检查数据库连接
        if self.db_manager is None or not self.db_manager.is_connected():
            try:
                if self.db_manager and self.db_manager.ensure_connection():
                    pass
                else:
                    return {
                        'questions_count': len(questions),
                        'results': [],
                        'summary': "错误：数据库未连接，无法执行查询",
                    }
            except Exception as e:
                return {
                    'questions_count': len(questions),
                    'results': [],
                    'summary': f"错误：数据库连接异常: {str(e)}",
                }

        # 调用 NL2SQLQueryEngine 批量查询（带自修复重试）
        engine = self._get_engine()
        max_workers = MCP_CONFIG.get('max_concurrency', 5)
        max_retries = MCP_CONFIG.get('sql_max_retries', 2)

        results = engine.query_batch(
            questions=questions,
            batch_no=batch_no,
            max_workers=max_workers,
            max_retries=max_retries,
            progress_callback=progress_callback,
        )

        # 生成摘要（含自修复统计）
        success_count = sum(1 for r in results if r.get('status') == 'success')
        total_rows = sum(r.get('row_count', 0) for r in results if r.get('status') == 'success')
        fixed_count = sum(1 for r in results if r.get('was_fixed'))
        failed_count = len(questions) - success_count
        retry_total = sum(r.get('retry_count', 0) for r in results)

        summary_parts = [
            f"共处理 {len(questions)} 条问题，成功 {success_count} 条，失败 {failed_count} 条，"
            f"总计返回 {total_rows} 行数据。"
        ]
        if fixed_count > 0:
            summary_parts.append(f"其中 {fixed_count} 条经过自修复后成功（共重试 {retry_total} 次）。")
        summary = " ".join(summary_parts)

        return {
            'questions_count': len(questions),
            'results': results,
            'summary': summary,
            'success_count': success_count,
            'failed_count': failed_count,
            'fixed_count': fixed_count,
        }

    # ================================================================
    # 辅助方法
    # ================================================================
    def format_tool_result_for_llm(self, tool_result: Dict[str, Any]) -> str:
        """
        将工具返回结果格式化为 LLM 可读的文本。

        Args:
            tool_result: call_tool 返回的 dict

        Returns:
            格式化的文本字符串
        """
        if tool_result.get('status') == 'error':
            return f"工具调用失败: {tool_result.get('error', '未知错误')}"

        result = tool_result.get('result', {})
        if not result:
            return "工具调用成功，但无返回结果。"

        # database_query 工具结果格式化
        results_list = result.get('results', [])
        if not results_list:
            return result.get('summary', '无查询结果')

        parts = [result.get('summary', '')]

        for i, r in enumerate(results_list, 1):
            parts.append(f"\n--- 问题 {i}: {r.get('question', '')} ---")
            if r.get('status') == 'success':
                parts.append(f"SQL: {r.get('sql', '')}")
                parts.append(f"行数: {r.get('row_count', 0)}")
                if r.get('was_fixed'):
                    parts.append(
                        f"（注意：此查询经过 {r.get('retry_count', 0)} 次自修复后成功，"
                        f"原始错误: {r.get('original_error', '')}）"
                    )
                # 包含数据预览（前 20 行）
                data = r.get('data', [])
                if data:
                    parts.append(f"数据预览（前 {min(20, len(data))} 行）:")
                    for row in data[:20]:
                        parts.append(f"  {row}")
            else:
                parts.append(f"错误: {r.get('error', '未知错误')}")
                if r.get('sql'):
                    parts.append(f"SQL: {r.get('sql', '')}")
                if r.get('retry_count', 0) > 0:
                    parts.append(
                        f"（已尝试 {r.get('retry_count', 0)} 次自修复均失败）"
                    )

        return "\n".join(parts)
