"""
智能客服对话引擎模块
基于 NL2SQL 能力构建多轮对话聊天机器人，支持：
1. 多轮对话上下文管理
2. 两阶段 LLM 调用：先生成 SQL，再用自然语言总结查询结果
3. 基于 MySQL 数据库进行查询
"""

import json
import re
import queue
import threading
import pandas as pd
from openai import OpenAI

from config.settings import DEFAULT_LLM_CONFIG, MAX_CHAT_HISTORY_TURNS, CHAT_MAX_SQL_COUNT, MYSQL_VERSION, REPORT_AUTO_FIX_SQL_ERRORS, ASSISTANT_MCP_MAX_ROUNDS
from modules.nl2sql_query import NL2SQLQueryEngine


class SmartAssistant:
    """智能客服对话引擎，支持多轮对话、SQL 透明执行、自然语言分析输出"""

    def __init__(self, api_key=None, endpoint=None, model=None, db_manager=None):
        """
        初始化智能客服助手

        Args:
            api_key: LLM API 密钥（可选，默认从配置读取）
            endpoint: API 端点 URL（可选，默认从配置读取）
            model: 模型名称（可选，默认从配置读取）
            db_manager: DatabaseManager 实例（可选，供 MCP 模式使用）
        """
        self.api_key = api_key or DEFAULT_LLM_CONFIG['api_key']
        self.endpoint = endpoint or DEFAULT_LLM_CONFIG['endpoint']
        self.model = model or DEFAULT_LLM_CONFIG['model']
        self.db_manager = db_manager
        self.chat_history = []

        # 初始化 OpenAI 客户端
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
            except Exception:
                self.client = None

    # ================================================================
    # 系统提示词构建
    # ================================================================
    def _build_system_prompt(self, schema_info):
        """
        构建系统提示词

        Args:
            schema_info: 数据表结构说明文本

        Returns:
            系统提示词字符串
        """
        system_prompt = f"""你是一位资深航空公司客户数据分析专家，擅长利用数据回答业务问题。
你当前可以访问以下数据表结构：

{schema_info}

【数据库版本】
当前数据库版本为 {MYSQL_VERSION}，请确保生成的 SQL 查询语句符合该版本的语法规则，避免使用不兼容的函数或语法。

【背景说明】
该数据集包含航空公司客户的出行记录与消费数据，其中 RFM 分析结果用于衡量客户价值。
RFM模型定义如下：
R=最近一次消费距今天数，值越小表示最近消费日期越近，
F=乘机次数（值越大表示消费越频繁），
M=消费金额/里程，值越大表示消费越高。


【意图判断与响应规则 — 务必严格遵守】
你的首要任务是判断用户问题是否需要对数据库中的实际数据执行查询才能准确回答：

**必须生成 SQL 查询的情况**（用户的问题依赖实际数据）：
- 任何涉及数值统计的问题：如"有多少"、"占比"、"排名"、"前N位"、"平均值"、"最大值"、"分布"
- 任何需要从数据中提取信息的问题：如"高价值客户有哪些特征"、"客户类型分布"、"乘机行为分析"
- 任何数据对比/筛选/排序：如"各类型对比"、"消费最多的客户"、"R值最小的"
- 任何可能用"是/否/有/无"回答的关于数据的问题

→ 每条 SQL 只用 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP 等修改操作
→ 每条 SQL 用 ```sql ... ``` 代码块包裹
→ SQL 之间用空行分隔
→ **极其重要**：此时只输出 SQL 代码块，不要附带任何分析、解释、总结、问候语或自然语言。系统会自动执行 SQL 并稍后将查询结果发回给你进行分析总结。

**不需要生成 SQL 的情况**（纯自然语言回复即可）：
- 问候寒暄：如"你好"、"你是谁"
- 纯概念/方法论解释（与当前数据无关）：如"什么是K-Means聚类原理"、"SQL中的JOIN有哪几种"
- 对工具使用方式的询问：如"你怎么用"、"这个页面是干什么的"
- 闲聊：如"谢谢"、"辛苦了"
- **注意**：涉及"建议"、"策略"、"怎么办"的问题，如果已在当前对话中提供了相关数据，可以直接基于数据给出建议，不需要重复查询。但如果是首次提问且没有数据上下文，必须先查询相关数据再给建议。

始终使用中文回复。对于涉及到业务逻辑的回答应保持专业性，适度控制回答的长度，便于业务人员理解且保证分析全面。"""

        return system_prompt

    def _build_mcp_system_prompt(self, schema_info, batch_no=None):
        """
        构建 MCP 模式的系统提示词。

        与默认模式不同，MCP 模式下系统提示词引导 LLM 使用 database_query 工具获取数据，
        而非直接生成 SQL。

        Args:
            schema_info: 数据表结构说明文本
            batch_no: 批次编号（可选）

        Returns:
            系统提示词字符串
        """
        batch_hint = ""
        if batch_no:
            batch_hint = f"\n\n【批次约束】\n当前分析批次为 {batch_no}，所有查询应限定在该批次范围内。调用工具时可传入 batch_no 参数。"

        system_prompt = f"""你是一位资深航空公司客户数据分析专家。你可以通过调用 database_query 工具来查询数据库获取所需数据。

你当前可以访问以下数据表结构：

{schema_info}

【数据库版本】
当前数据库版本为 MySQL 8.0.44。

【背景说明】
该数据集包含航空公司客户的出行记录与消费数据，其中 RFM 分析结果用于衡量客户价值。
RFM模型定义如下：
R=最近一次消费距今天数，值越小表示最近消费日期越近，
F=乘机次数（值越大表示消费越频繁），
M=消费金额/里程，值越大表示消费越高。

【工具使用规则】
1. 当用户问题需要基于实际数据回答时，请调用 database_query 工具查询数据
2. 工具接受 1-5 条自然语言问题，你可以一次提交多个相关问题以提高效率
3. 获取数据后，请用专业的自然语言总结分析结果，不要直接输出 SQL
4. 最多调用工具 {ASSISTANT_MCP_MAX_ROUNDS} 次，请合理规划查询
5. 对于问候、概念解释等不需要数据的问题，直接回复即可，无需调用工具
6. 始终使用中文回复，保持专业性和可读性
{batch_hint}"""

        return system_prompt

    # ================================================================
    # 对话历史管理
    # ================================================================
    def add_message(self, role, content):
        """
        向对话历史添加一条消息

        Args:
            role: 消息角色，'user' 或 'assistant'
            content: 消息内容
        """
        self.chat_history.append({
            'role': role,
            'content': content
        })
        self.trim_history()

    def get_messages(self):
        """
        获取当前对话历史

        Returns:
            对话历史列表（深拷贝的引用，供调用方读取）
        """
        return self.chat_history

    def clear_history(self):
        """清空全部对话历史"""
        self.chat_history = []

    def trim_history(self):
        """
        裁剪对话历史，保持不超过最大轮数上限

        当历史消息数超过 MAX_CHAT_HISTORY_TURNS * 2 时，
        从头部移除最早的一轮对话（user + assistant 各一条）。
        """
        max_messages = MAX_CHAT_HISTORY_TURNS * 2
        while len(self.chat_history) > max_messages:
            # 移除最旧的一轮（user 消息 + assistant 消息）
            if len(self.chat_history) >= 2:
                self.chat_history.pop(0)
                self.chat_history.pop(0)
            else:
                self.chat_history.pop(0)

    # ================================================================
    # 主对话入口
    # ================================================================
    def chat(self, user_message, db_manager=None, batch_no=None):
        """
        处理用户消息并返回对话结果（非流式版本，供兼容使用）

        实现两阶段 LLM 调用流程：
        第一阶段：将用户问题发送给 LLM，获取可能包含 SQL 的回复
        第二阶段：若提取到 SQL 则执行查询，并请 LLM 用自然语言总结结果

        Args:
            user_message: 用户输入的消息
            db_manager: DatabaseManager 实例
            batch_no: 批次编号过滤条件（可选）

        Returns:
            dict: {
                'reply': str,       # 助手回复文本
                'sql': str | None,  # 生成的 SQL（如有）
                'data': DataFrame | None,  # 查询结果（如有）
                'status': str       # 'success' 或 'error'
            }
        """
        sql = None
        result_df = None
        max_retries = 2 if REPORT_AUTO_FIX_SQL_ERRORS else 0

        try:
            schema_info = NL2SQLQueryEngine.DB_TABLE_SCHEMA

            system_prompt = self._build_system_prompt(schema_info)

            # 组装发送给 LLM 的消息列表
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(self.chat_history)
            messages.append({'role': 'user', 'content': user_message})

            # ---- 第一阶段：生成 SQL ----
            first_response = self._call_llm(messages, temperature=0)
            sql_list = self._extract_all_sql(first_response)

            result_df = None
            sql = None  # 用于返回值兼容（单条 SQL 场景）

            if sql_list:
                sql = sql_list[0]  # 兼容字段：返回首条 SQL
                # 校验并执行每条 SQL，收集结果（严格串行，防串扰）
                all_results = []      # [(sql, df_or_None, error_or_None, was_fixed)]
                any_success = False
                successful_sqls = []  # 已成功的 SQL 列表，用于防串扰校验

                for i, one_sql in enumerate(sql_list):
                    # 校验 SQL 安全性
                    is_valid, error_msg = self._validate_sql(one_sql)
                    if not is_valid:
                        all_results.append((one_sql, None, f"校验失败：{error_msg}", False))
                        continue

                    # 执行查询（带自动修复重试），复用统一的隔离修复方法
                    fix_result = self._fix_single_sql(
                        sql_index=i,
                        current_sql=one_sql,
                        exec_error="",  # 占位，方法内部会首次执行
                        user_message=user_message,
                        max_retries=max_retries,
                        db_manager=db_manager,
                        successful_sqls=successful_sqls,
                    )

                    if fix_result['exec_success']:
                        any_success = True
                        final_sql = fix_result['final_sql']
                        all_results.append((final_sql, fix_result['result_df'], None, fix_result['was_fixed']))
                        # 加入已成功列表，供后续 SQL 防串扰校验
                        successful_sqls.append(final_sql)
                        if result_df is None:
                            result_df = fix_result['result_df']
                    else:
                        all_results.append((fix_result['final_sql'], None, fix_result['last_error'], False))

                # ---- 第二阶段：请 LLM 总结查询结果 ----
                # 拼接所有成功结果为文本
                result_parts = []
                for idx, (s, df_i, err, fixed) in enumerate(all_results):
                    if df_i is not None:
                        part_text = self._format_query_result_for_prompt(df_i)
                        suffix = "（经过自动修复后成功）" if fixed else ""
                        result_parts.append(f"【查询 {idx+1}】{suffix}\n{part_text}")
                    elif err:
                        result_parts.append(f"【查询 {idx+1}】执行失败：{err}")
                result_text = "\n\n".join(result_parts)

                if not any_success:
                    first_err = all_results[0][2] if all_results else "未知错误"
                    final_reply = f"抱歉，SQL 查询执行失败（已尝试 {max_retries + 1} 次）：{first_err}"
                    self.add_message('user', user_message)
                    self.add_message('assistant', final_reply)
                    return {
                        'reply': final_reply,
                        'sql': sql,
                        'data': None,
                        'status': 'error'
                    }

                summary_messages = messages.copy()
                # 替换第一阶段系统提示，防止 LLM 再次输出 SQL
                summary_messages[0] = {
                    'role': 'system',
                    'content': (
                        "你是一位资深航空公司客户数据分析专家。现在已进入分析总结阶段，"
                        "你此前已生成了 SQL 查询并获得了真实数据结果。"
                        "你需要基于提供的查询结果，对用户最初的问题给出专业、完整的自然语言回复。"
                        "如果用户问的是经营建议或策略问题，请结合数据给出具体、可落地的建议。"
                        "不要输出任何 SQL 语句，不要提及'查询结果'等内部过程用语。"
                        "始终使用中文回复。"
                    )
                }
                summary_messages.append({
                    'role': 'assistant',
                    'content': '已根据用户问题生成并执行了相应的 SQL 查询语句。'
                })
                summary_messages.append({
                    'role': 'user',
                    'content': (
                        f"以下是对用户问题「{user_message}」的查询结果：\n\n{result_text}\n\n"
                        "请直接回答用户的问题，不要再次输出 SQL。"
                    )
                })
                final_reply = self._call_llm(summary_messages, temperature=0.3)
            else:
                # 无 SQL，LLM 直接回复即为最终答案
                final_reply = first_response

            # 保存对话历史
            self.add_message('user', user_message)
            self.add_message('assistant', final_reply)

            return {
                'reply': final_reply,
                'sql': sql,
                'data': result_df,
                'status': 'success'
            }

        except Exception as e:
            error_reply = f"对话处理异常：{str(e)}"
            self.add_message('user', user_message)
            self.add_message('assistant', error_reply)
            return {
                'reply': error_reply,
                'sql': sql if 'sql' in dir() else None,
                'data': None,
                'status': 'error'
            }

    # ================================================================
    # 辅助方法
    # ================================================================
    def _fix_single_sql(self, sql_index, current_sql, exec_error,
                        user_message, max_retries, db_manager,
                        successful_sqls=None, progress_callback=None):
        """
        对单条出错的 SQL 执行串行的自动修复重试。

        防串扰设计：
        1. 修复 prompt 中不携带 LLM 的完整多 SQL 响应（first_response），
           避免其他 SQL（尤其已成功的）被当作"修复参考"返回。
        2. 修复 prompt 明确指示只修复当前这一条 SQL，禁止返回其他 SQL。
        3. 修复后的 SQL 若与已成功的 SQL 完全相同，则视为串扰，拒绝采纳。
        4. 多次重试严格串行：前一次失败后才发起下一次，不并发。

        Args:
            sql_index: int, 当前 SQL 在 sql_list 中的序号（从 0 开始），用于提示
            current_sql: str, 当前执行失败的 SQL
            exec_error: str, 数据库返回的错误信息（首次执行前的占位，方法内部会首次执行）
            user_message: str, 用户原始问题（提供修复上下文）
            max_retries: int, 最大修复重试次数
            db_manager: DatabaseManager 实例
            successful_sqls: list[str], 已成功执行的 SQL 列表，用于防串扰校验
            progress_callback: callable, optional, 进度回调
                fn(event_type, payload) 其中 event_type 为：
                'sql_executed' / 'sql_error' / 'retrying' / 'sql_fixed' / 'sql_adopted'

        Returns:
            dict: {
                'exec_success': bool,
                'result_df': DataFrame or None,
                'final_sql': str,         # 最终使用的 SQL（可能是修复后的）
                'original_sql': str,      # 原始 SQL（首次执行的）
                'original_error': str,    # 首次错误信息
                'was_fixed': bool,        # 是否经过修复后成功
                'last_error': str or None # 最终错误（失败时）
            }
        """
        if successful_sqls is None:
            successful_sqls = []

        def _notify(event_type, payload=None):
            if progress_callback is not None:
                try:
                    progress_callback(event_type, payload or {})
                except Exception:
                    pass

        original_sql = current_sql
        original_error = exec_error
        result_df = None
        exec_success = False
        was_fixed = False
        last_error = exec_error

        # 重试上下文：仅包含当前 SQL 的修复历史，不包含其他 SQL
        # 用独立列表累积当前 SQL 的修复对话，避免跨 SQL 串扰
        fix_dialog = []  # [{'role':'user'/'assistant', 'content':...}]

        for attempt in range(max_retries + 1):
            result_df_i, err = self._execute_sql(current_sql, db_manager)

            if not err:
                # 执行成功
                # 防串扰校验：若修复后的 SQL 与某条已成功的 SQL 完全相同，视为串扰
                if current_sql in successful_sqls and current_sql != original_sql:
                    # 修复结果与已成功 SQL 重复，拒绝采纳，继续尝试修复
                    err = (
                        f"修复后的 SQL 与此前已成功执行的某条 SQL 完全相同，"
                        f"疑似上下文串扰。请生成一条不同的、针对当前错误的 SQL。"
                    )
                    last_error = err
                    _notify('sql_error', {'sql': current_sql, 'error': err, 'attempt': attempt})
                    if attempt < max_retries:
                        fix_dialog.append({
                            'role': 'user',
                            'content': f"上一次修复结果被拒绝：{err}"
                        })
                        # 继续进入下方的修复逻辑
                    else:
                        break
                else:
                    exec_success = True
                    result_df = result_df_i
                    if attempt > 0:
                        was_fixed = True
                    _notify('sql_executed', {'sql': current_sql, 'attempt': attempt, 'was_fixed': was_fixed})
                    break

            # 记录首次失败的原始错误
            if attempt == 0:
                original_error = err
                _notify('sql_error', {'sql': current_sql, 'error': err, 'attempt': attempt})
            else:
                _notify('sql_error', {'sql': current_sql, 'error': err, 'attempt': attempt})

            last_error = err
            if attempt >= max_retries:
                break

            # 构造修复 prompt —— 严格隔离，不携带 first_response
            # 只提供：表结构 + 用户问题 + 当前出错 SQL + 错误信息 + 修复历史
            _notify('retrying', {'attempt': attempt + 1, 'max_retries': max_retries, 'sql_index': sql_index})

            fix_prompt = (
                f"用户问题：{user_message}\n\n"
                f"以下是数据库表结构定义：\n"
                f"{NL2SQLQueryEngine.DB_TABLE_SCHEMA}\n\n"
                f"当前需要修复的是第 {sql_index + 1} 条 SQL，执行出错：\n"
                f"```sql\n{current_sql}\n```\n\n"
                f"错误信息：{err}\n\n"
                f"【重要约束】\n"
                f"1. 只修复上面这一条 SQL，不要返回其他 SQL 语句\n"
                f"2. 修复后的 SQL 必须针对当前错误，不要照搬其他查询\n"
                f"3. 只输出修复后的 SQL 语句（用 ```sql ... ``` 包裹），不要输出任何解释\n"
            )

            fix_messages = [
                {
                    'role': 'system',
                    'content': (
                        f"你是一位 {MYSQL_VERSION} SQL 修复专家。"
                        "用户会提供一条执行出错的 SQL 及其错误信息，"
                        "你需要分析错误原因并输出修复后的 SQL。"
                        "严禁返回与问题无关的其他 SQL 语句。"
                    ),
                },
                {'role': 'user', 'content': fix_prompt},
            ]
            # 追加当前 SQL 的修复历史（仅限本条 SQL 的多次重试）
            fix_messages.extend(fix_dialog)

            try:
                fix_response = self._call_llm(fix_messages, temperature=0)
            except Exception:
                # LLM 调用失败，跳出重试
                break

            new_sql = self._extract_sql(fix_response)
            if not new_sql or new_sql == current_sql:
                # LLM 未生成新 SQL 或与当前 SQL 相同，继续下一次重试
                fix_dialog.append({
                    'role': 'assistant',
                    'content': fix_response,
                })
                fix_dialog.append({
                    'role': 'user',
                    'content': '未能生成有效的修复 SQL，请重新分析错误并输出不同的修复方案。',
                })
                continue

            # 防串扰校验：修复结果不能与已成功的 SQL 相同
            if new_sql in successful_sqls:
                fix_dialog.append({
                    'role': 'assistant',
                    'content': fix_response,
                })
                fix_dialog.append({
                    'role': 'user',
                    'content': (
                        '修复后的 SQL 与此前已成功执行的某条 SQL 完全相同，'
                        '这是不允许的。请生成一条不同的、针对当前错误的 SQL。'
                    ),
                })
                # 不采纳，继续重试
                continue

            # 采纳修复后的 SQL
            current_sql = new_sql
            _notify('sql_adopted', {'sql': new_sql, 'attempt': attempt + 1})
            fix_dialog.append({
                'role': 'assistant',
                'content': fix_response,
            })
            fix_dialog.append({
                'role': 'user',
                'content': f'SQL 执行出错：{err}',
            })

        if exec_success and was_fixed:
            _notify('sql_fixed', {
                'original_sql': original_sql,
                'original_error': original_error,
            })

        return {
            'exec_success': exec_success,
            'result_df': result_df,
            'final_sql': current_sql,
            'original_sql': original_sql,
            'original_error': original_error,
            'was_fixed': was_fixed,
            'last_error': last_error if not exec_success else None,
        }

    def _execute_sql(self, sql, db_manager):
        """
        通过 MySQL 数据库执行 SQL 查询

        Args:
            sql: SQL 查询语句
            db_manager: DatabaseManager 实例

        Returns:
            (result_df, error_message) 元组
        """
        if db_manager is None or not db_manager.is_connected():
            return None, "数据库未连接，请先在数据库管理页面连接 MySQL 数据库"
        result_df, error_msg = db_manager.execute_query_safe(sql)
        if error_msg:
            return None, f"MySQL 执行失败: {error_msg}"
        return result_df, ""

    def _format_query_result_for_prompt(self, df):
        """
        将查询结果 DataFrame 转换为可读文本，用于发送给 LLM 总结

        Args:
            df: 查询结果 DataFrame

        Returns:
            格式化的文本字符串
        """
        if df is None or df.empty:
            return "查询结果为空，没有匹配的数据。"

        total_rows = len(df)
        max_rows = 50
        display_df = df.head(max_rows)
        text = display_df.to_string(index=False)

        if total_rows > max_rows:
            text += f"\n\n（共 {total_rows} 条结果，以上展示前 {max_rows} 条）"

        return text

    def _extract_sql(self, response):
        """
        从 LLM 响应中提取单条 SQL 语句（兼容旧逻辑）

        Args:
            response: LLM 的响应文本

        Returns:
            提取到的第一条 SQL 语句，未找到则返回 None
        """
        sqls = self._extract_all_sql(response)
        return sqls[0] if sqls else None

    def _extract_all_sql(self, response):
        """
        从 LLM 响应中提取所有 SQL 语句

        解析顺序：
        1. 优先提取所有 ```sql ... ``` / ``` ... ``` 代码块
        2. 若代码块不存在，回退到裸文本中提取 SELECT/WITH 语句

        关键修复：单个代码块内可能包含多条以分号分隔的 SQL 语句，
        必须按分号拆分，否则会被当作一条语句导致执行无限报错。
        最终按 CHAT_MAX_SQL_COUNT 截断。

        Args:
            response: LLM 的响应文本

        Returns:
            SQL 语句列表（可能为空列表）
        """
        if not response:
            return []

        # 候选 SQL 片段（可能含多条语句）
        candidate_blocks = []

        # 1. 提取所有 markdown 代码块
        code_block_pattern = r'```(?:sql)?\s*\n?(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            candidate_blocks = [m.strip() for m in matches if m.strip()]

        # 2. 回退：从裸文本提取以 SELECT/WITH 开头的语句（匹配到分号或代码块结束）
        if not candidate_blocks:
            # 先按分号切分裸文本，再过滤出 SELECT/WITH 开头的片段
            # 这种方式比贪婪正则更稳健，能正确处理 WITH ... SELECT ...; 形式
            text = response
            # 移除可能存在的 markdown 残留标记
            text = re.sub(r'```sql\s*\n?', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text)
            # 按分号切分
            parts = text.split(';')
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                # 剥离前导注释
                cleaned = re.sub(r'^\s*--[^\n]*\n', '', part)
                cleaned = re.sub(r'^\s*/\*.*?\*/\s*', '', cleaned, flags=re.DOTALL)
                cleaned = cleaned.strip()
                # 片段可能以非 SQL 文字开头（如"以下是查询：\nSELECT..."），
                # 定位第一个 SELECT/WITH 关键字作为 SQL 起点
                m = re.search(r'\b(SELECT|WITH)\b', cleaned, re.IGNORECASE)
                if m:
                    cleaned = cleaned[m.start():].strip()
                    candidate_blocks.append(cleaned)

        # 3. 对每个候选块按分号拆分为独立 SQL 语句
        sqls = []
        for block in candidate_blocks:
            # 移除块内的 markdown 残留
            block = re.sub(r'```sql\s*\n?', '', block, flags=re.IGNORECASE)
            block = block.replace('```', '').strip()
            # 按分号拆分（分号可能出现在字符串字面值内的情况极少，且 LLM 生成的分析 SQL 通常不包含分号在字符串内）
            parts = block.split(';')
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                # 确保是以 SELECT 或 WITH 开头（跳过块内的注释或说明文字）
                # 先剥离前导注释
                cleaned = re.sub(r'^\s*--[^\n]*\n', '', part)
                cleaned = re.sub(r'^\s*/\*.*?\*/\s*', '', cleaned, flags=re.DOTALL)
                cleaned = cleaned.strip()
                if re.match(r'^(SELECT|WITH)\b', cleaned, re.IGNORECASE):
                    sqls.append(cleaned)

        return sqls[:CHAT_MAX_SQL_COUNT]

    def _validate_sql(self, sql):
        """
        SQL 安全校验

        复用 NL2SQLQueryEngine 的危险关键字列表和校验逻辑。
        会先去除 SQL 注释再检查语句类型前缀，避免注释导致误判。

        Args:
            sql: 待校验的 SQL 语句

        Returns:
            (is_valid, error_message) 元组
        """
        if not sql or not sql.strip():
            return False, "SQL 语句为空"

        sql_upper = sql.upper().strip()

        for keyword in NL2SQLQueryEngine.DANGEROUS_KEYWORDS:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                return False, f"SQL 包含危险操作: {keyword}，只允许 SELECT 查询"

        # 去除行级注释（-- ...）和块级注释（/* ... */）后再检查前缀
        stripped = re.sub(r'--[^\n]*', '', sql_upper)
        stripped = re.sub(r'/\*.*?\*/', '', stripped, flags=re.DOTALL)
        stripped = stripped.strip()

        if not (stripped.startswith('SELECT') or stripped.startswith('WITH')):
            return False, "只允许 SELECT 查询语句"

        return True, ""

    def _call_llm(self, messages, temperature=0):
        """
        调用 LLM API（非流式）

        Args:
            messages: 消息列表（包含 role 和 content 的字典列表）
            temperature: 采样温度（SQL 生成用 0，自然语言总结用 0.3）

        Returns:
            LLM 响应文本
        """
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化，请检查 API Key 配置")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def _call_llm_stream(self, messages, temperature=0.6):
        """
        调用 LLM API（流式返回）

        Args:
            messages: 消息列表（包含 role 和 content 的字典列表）
            temperature: 采样温度（SQL 生成用 0，自然语言总结用 0.3）

        Yields:
            LLM 响应的文本片段（generator）
        """
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化，请检查 API Key 配置")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_stream(self, user_message, db_manager=None, batch_no=None):
        """
        处理用户消息并以流式方式返回对话结果

        实现两阶段 LLM 调用流程：
        第一阶段：将用户问题发送给 LLM，获取可能包含 SQL 的回复（非流式，需完整提取 SQL）
        第二阶段：若提取到 SQL 则执行查询，并请 LLM 用流式方式输出自然语言总结

        Args:
            user_message: 用户输入的消息
            db_manager: DatabaseManager 实例
            batch_no: 批次编号过滤条件（可选）

        Yields:
            dict: 包含以下类型的事件：
                - {'type': 'status', 'content': str} - 状态信息
                - {'type': 'sql', 'content': str} - 执行的单条 SQL
                - {'type': 'data', 'content': DataFrame} - 单条 SQL 的查询结果
                - {'type': 'sql_error', 'content': str} - 单条 SQL 执行失败的错误信息
                - {'type': 'sql_fixed', 'content': None, 'original_sql': str, 'original_error': str} - SQL 经修复后成功，附带原始 SQL 和原始错误
                - {'type': 'chunk', 'content': str} - 流式文本片段
                - {'type': 'done', 'reply': str, 'sql': list, 'data': DataFrame, 'status': str} - 完成事件
        """
        sql_list = []
        result_df = None
        max_retries = 2 if REPORT_AUTO_FIX_SQL_ERRORS else 0  # SQL 修复最大重试次数

        try:
            schema_info = NL2SQLQueryEngine.DB_TABLE_SCHEMA

            system_prompt = self._build_system_prompt(schema_info)

            # 组装发送给 LLM 的消息列表
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(self.chat_history)
            messages.append({'role': 'user', 'content': user_message})

            # ---- 第一阶段：生成 SQL（非流式，需完整响应提取 SQL）----
            yield {'type': 'status', 'content': '正在分析问题...'}
            first_response = self._call_llm(messages, temperature=0)
            sql_list = self._extract_all_sql(first_response)

            if sql_list:
                yield {'type': 'status', 'content': f'已生成 {len(sql_list)} 条 SQL，正在执行查询...'}

                # 校验并执行每条 SQL，收集结果（严格串行，防串扰）
                all_results = []
                successful_sqls = []  # 已成功的 SQL 列表，用于防串扰校验

                for i, sql in enumerate(sql_list):
                    yield {'type': 'sql', 'content': sql}

                    # 校验 SQL 安全性
                    is_valid, error_msg = self._validate_sql(sql)
                    if not is_valid:
                        yield {'type': 'sql_error', 'content': f"第 {i+1} 条 SQL 校验失败：{error_msg}"}
                        continue

                    # 第一次执行
                    yield {'type': 'status', 'content': f'正在执行第 {i+1}/{len(sql_list)} 条 SQL 查询...'}
                    df, exec_error = self._execute_sql(sql, db_manager)
                    was_fixed = False
                    original_sql = sql
                    original_error = ''
                    current_sql = sql  # 当前执行的 SQL（修复成功后会更新）

                    # 失败时自动尝试修复（严格串行 + 防串扰，实时 yield 进度）
                    if exec_error and max_retries > 0:
                        original_error = exec_error
                        current_sql = sql
                        # 当前 SQL 的修复对话历史（隔离，不跨 SQL）
                        fix_dialog = []

                        for attempt in range(1, max_retries + 1):
                            yield {'type': 'status', 'content': f'第 {i+1} 条 SQL 出错，正在自动修复（第 {attempt}/{max_retries} 次）...'}

                            # 构造修复 prompt —— 严格隔离，不携带 first_response
                            fix_prompt = (
                                f"用户问题：{user_message}\n\n"
                                f"以下是数据库表结构定义：\n"
                                f"{NL2SQLQueryEngine.DB_TABLE_SCHEMA}\n\n"
                                f"当前需要修复的是第 {i + 1} 条 SQL，执行出错：\n"
                                f"```sql\n{current_sql}\n```\n\n"
                                f"错误信息：{exec_error}\n\n"
                                f"【重要约束】\n"
                                f"1. 只修复上面这一条 SQL，不要返回其他 SQL 语句\n"
                                f"2. 修复后的 SQL 必须针对当前错误，不要照搬其他查询\n"
                                f"3. 只输出修复后的 SQL 语句（用 ```sql ... ``` 包裹），不要输出任何解释\n"
                            )
                            fix_messages = [
                                {
                                    'role': 'system',
                                    'content': (
                                        f"你是一位 {MYSQL_VERSION} SQL 修复专家。"
                                        "用户会提供一条执行出错的 SQL 及其错误信息，"
                                        "你需要分析错误原因并输出修复后的 SQL。"
                                        "严禁返回与问题无关的其他 SQL 语句。"
                                    ),
                                },
                                {'role': 'user', 'content': fix_prompt},
                            ]
                            fix_messages.extend(fix_dialog)

                            try:
                                fix_response = self._call_llm(fix_messages, temperature=0)
                            except Exception:
                                break

                            new_sql = self._extract_sql(fix_response)
                            if not new_sql or new_sql == current_sql:
                                fix_dialog.append({'role': 'assistant', 'content': fix_response})
                                fix_dialog.append({'role': 'user', 'content': '未能生成有效的修复 SQL，请重新分析错误并输出不同的修复方案。'})
                                continue

                            # 防串扰校验：修复结果不能与已成功的 SQL 相同
                            if new_sql in successful_sqls:
                                fix_dialog.append({'role': 'assistant', 'content': fix_response})
                                fix_dialog.append({
                                    'role': 'user',
                                    'content': (
                                        '修复后的 SQL 与此前已成功执行的某条 SQL 完全相同，'
                                        '这是不允许的。请生成一条不同的、针对当前错误的 SQL。'
                                    ),
                                })
                                exec_error = "修复结果与已成功 SQL 重复（串扰），已拒绝"
                                continue

                            # 采纳修复后的 SQL，实时 yield 给 UI
                            current_sql = new_sql
                            sql_list[i] = new_sql
                            yield {'type': 'sql', 'content': new_sql}

                            # 执行修复后的 SQL
                            df, exec_error = self._execute_sql(current_sql, db_manager)
                            if not exec_error:
                                was_fixed = True
                                yield {
                                    'type': 'sql_fixed',
                                    'content': None,
                                    'original_sql': original_sql,
                                    'original_error': original_error,
                                }
                                break

                            # 修复后仍失败，记录对话历史并继续
                            fix_dialog.append({'role': 'assistant', 'content': fix_response})
                            fix_dialog.append({'role': 'user', 'content': f'SQL 执行出错：{exec_error}'})

                    if exec_error:
                        yield {'type': 'sql_error', 'content': f"第 {i+1} 条 SQL 执行失败（已尝试 {max_retries + 1} 次）：{exec_error}"}
                        continue

                    # 执行成功
                    successful_sqls.append(current_sql)
                    if df is not None and not df.empty:
                        all_results.append(df)
                        yield {'type': 'data', 'content': df}

                # 合并所有查询结果
                if all_results:
                    result_df = pd.concat(all_results, ignore_index=True)

                yield {'type': 'status', 'content': '正在生成分析总结...'}

                # ---- 第二阶段：请 LLM 用流式方式总结查询结果 ----
                result_texts = []
                for idx, df in enumerate(all_results):
                    result_texts.append(f"--- 第 {idx+1} 条 SQL 查询结果 ---\n{self._format_query_result_for_prompt(df)}")

                combined_result = "\n\n".join(result_texts) if result_texts else "查询结果为空"

                summary_messages = messages.copy()
                # 替换第一阶段系统提示，防止 LLM 再次输出 SQL
                summary_messages[0] = {
                    'role': 'system',
                    'content': (
                        "你是一位资深航空公司客户数据分析专家。现在已进入分析总结阶段，"
                        "你此前已生成了 SQL 查询并获得了真实数据结果。"
                        "你需要基于提供的查询结果，对用户最初的问题给出专业、完整的自然语言回复。"
                        "如果用户问的是经营建议或策略问题，请结合数据给出具体、可落地的建议。"
                        "不要输出任何 SQL 语句，不要提及'查询结果'等内部过程用语。"
                        "始终使用中文回复。"
                    )
                }
                summary_messages.append({
                    'role': 'assistant',
                    'content': '已根据用户问题生成并执行了相应的 SQL 查询语句。'
                })
                summary_messages.append({
                    'role': 'user',
                    'content': (
                        f"以下是对用户问题「{user_message}」的查询结果：\n\n{combined_result}\n\n"
                        "请直接回答用户的问题，不要再次输出 SQL。"
                    )
                })

                final_reply = ""
                for chunk_text in self._call_llm_stream(summary_messages, temperature=0.3):
                    final_reply += chunk_text
                    yield {'type': 'chunk', 'content': chunk_text}
            else:
                # 无 SQL，LLM 直接回复即为最终答案（也用流式）
                final_reply = ""
                for chunk_text in self._call_llm_stream(messages, temperature=0.3):
                    final_reply += chunk_text
                    yield {'type': 'chunk', 'content': chunk_text}

            # 保存对话历史
            self.add_message('user', user_message)
            self.add_message('assistant', final_reply)

            yield {
                'type': 'done',
                'reply': final_reply,
                'sql': sql_list if sql_list else None,
                'data': result_df,
                'status': 'success'
            }

        except Exception as e:
            error_reply = f"对话处理异常：{str(e)}"
            self.add_message('user', user_message)
            self.add_message('assistant', error_reply)
            yield {'type': 'chunk', 'content': error_reply}
            yield {'type': 'done', 'reply': error_reply, 'sql': sql_list if sql_list else None, 'data': None, 'status': 'error'}

    def chat_stream_mcp(self, user_message, db_manager=None, batch_no=None):
        """
        MCP 模式流式对话：LLM 通过 function calling 调用 database_query 工具获取数据。

        流程：
        1. 将用户问题发送给 LLM，附带 database_query 工具定义
        2. LLM 决定是否调用工具，若调用则执行工具（并发查询）
        3. 将工具结果回传 LLM，LLM 可能再次调用工具或直接回复
        4. 最终 LLM 流式输出自然语言总结

        Args:
            user_message: 用户输入的消息
            db_manager: DatabaseManager 实例
            batch_no: 批次编号过滤条件（可选）

        Yields:
            dict: 事件类型：
                - {'type': 'status', 'content': str}
                - {'type': 'tool_call', 'tool_name': str, 'arguments': dict, 'round': int}
                - {'type': 'tool_result', 'tool_name': str, 'result_summary': str, 'round': int, 'status': str, 'sub_results': list}
                  sub_results 中每项含: question, sql, status, row_count, error, was_fixed, original_sql, original_error, retry_count
                - {'type': 'chunk', 'content': str}
                - {'type': 'done', 'reply': str, 'status': str}
        """
        from modules.mcp_tool_service import MCPToolService

        # 使用传入的 db_manager 或初始化时的
        dm = db_manager if db_manager is not None else self.db_manager

        # 创建 MCP 工具服务实例（每次对话独立计数）
        tool_call_limit = ASSISTANT_MCP_MAX_ROUNDS
        mcp_service = MCPToolService(
            db_manager=dm,
            api_key=self.api_key,
            endpoint=self.endpoint,
            model=self.model,
            call_limit=tool_call_limit,
        )

        try:
            schema_info = NL2SQLQueryEngine.DB_TABLE_SCHEMA
            system_prompt = self._build_mcp_system_prompt(schema_info, batch_no)

            # 组装消息
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(self.chat_history)
            messages.append({'role': 'user', 'content': user_message})

            tool_defs = mcp_service.get_tool_definitions()
            round_num = 0

            # 多轮工具调用循环
            while round_num < tool_call_limit:
                round_num += 1
                yield {'type': 'status', 'content': f'正在分析问题（第 {round_num} 轮）...'}

                # 调用 LLM（非流式，需完整响应以判断是否调用工具）
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_defs,
                    tool_choice='auto',
                    temperature=0,
                )

                message = response.choices[0].message

                # 如果 LLM 没有调用工具，说明已准备好直接回复
                if not message.tool_calls:
                    # 流式输出最终回复
                    yield {'type': 'status', 'content': '正在生成回答...'}
                    final_reply = ""
                    for chunk_text in self._call_llm_stream(messages, temperature=0.3):
                        final_reply += chunk_text
                        yield {'type': 'chunk', 'content': chunk_text}

                    self.add_message('user', user_message)
                    self.add_message('assistant', final_reply)
                    yield {'type': 'done', 'reply': final_reply, 'status': 'success'}
                    return

                # 处理工具调用
                # 先将 assistant 的 tool_calls 消息加入上下文
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

                    # 注入 batch_no（如果用户未在参数中指定）
                    if 'batch_no' not in arguments and batch_no:
                        arguments['batch_no'] = batch_no

                    yield {
                        'type': 'tool_call',
                        'tool_name': tool_name,
                        'arguments': arguments,
                        'round': round_num,
                    }
                    yield {'type': 'status', 'content': f'正在执行工具 {tool_name}（第 {round_num} 轮）...'}

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

                    # 格式化结果给 LLM
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

                    # 将工具结果加入消息上下文
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': result_text,
                    })

                # 继续循环，让 LLM 基于工具结果决定下一步

            # 达到工具调用上限，让 LLM 基于已有数据直接回复
            yield {'type': 'status', 'content': '已达到工具调用上限，正在基于已获取数据生成回答...'}

            # 添加提示让 LLM 直接回答
            messages.append({
                'role': 'user',
                'content': '请基于以上已获取的数据，直接回答我最初的问题。不要再调用工具。'
            })

            final_reply = ""
            for chunk_text in self._call_llm_stream(messages, temperature=0.3):
                final_reply += chunk_text
                yield {'type': 'chunk', 'content': chunk_text}

            self.add_message('user', user_message)
            self.add_message('assistant', final_reply)
            yield {'type': 'done', 'reply': final_reply, 'status': 'success'}

        except Exception as e:
            error_reply = f"对话处理异常：{str(e)}"
            self.add_message('user', user_message)
            self.add_message('assistant', error_reply)
            yield {'type': 'chunk', 'content': error_reply}
            yield {'type': 'done', 'reply': error_reply, 'status': 'error'}

    def is_llm_available(self):
        """
        检查 LLM 是否可用

        Returns:
            bool: API Key 已配置且客户端初始化成功时返回 True
        """
        return self.client is not None and bool(self.api_key)
