import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque

# 为了跨平台文件锁，简单实现一个上下文管理器
# 在生产环境中，建议使用 filelock 库: pip install filelock
try:
    #import fcntl
    def lock_file(f):
        pass #fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def unlock_file(f):
        pass #fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except ImportError:
    # Windows 或非 Unix 环境的简化 fallback (实际生产建议用 filelock 库)
    def lock_file(f):
        pass
    def unlock_file(f):
        pass


class AgentMemory:
    def __init__(self, file_path: str = "agent_history.jsonl"):
        self.file_path = file_path
        # 确保文件存在
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                pass  # 创建空文件

    def _parse_line(self, line: str) -> Optional[Dict]:
        """安全解析单行 JSON"""
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            print(f"[Warning] 跳过损坏的记忆行: {line[:50]}...")
            return None

    def add(self, role: str, content: str, meta: Optional[Dict] = None):
        """
        添加一条新记忆到文件末尾 (Append Only)
        :param role: 'user', 'assistant', 'system', 或 'tool'
        :param content: 对话内容
        :param meta: 额外元数据 (如 token 消耗, 工具名称等)
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "meta": meta or {}
        }

        json_line = json.dumps(record, ensure_ascii=False) + "\n"

        with open(self.file_path, 'a', encoding='utf-8') as f:
            lock_file(f)
            try:
                f.write(json_line)
                f.flush()  # 确保立即写入磁盘
            finally:
                unlock_file(f)

    def get_recent_context(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        读取最近 N 条记忆 (滑动窗口)
        这是 AgentLoop 中最常用的方法，用于构建 Prompt 上下文
        :param n: 需要读取的最近条数
        :return: 字典列表，按时间顺序排列
        """
        if n <= 0:
            return []

        records = []
        # 使用 deque 高效地保留最后 N 个元素
        buffer = deque(maxlen=n)

        with open(self.file_path, 'r', encoding='utf-8') as f:
            lock_file(f)
            try:
                # 方案 A: 如果文件不大，直接遍历 (最简单可靠)
                # 对于超大文件 (GB 级)，可以使用 seek 从文件末尾倒读，但实现较复杂
                for line in f:
                    parsed = self._parse_line(line)
                    if parsed:
                        buffer.append(parsed)
            finally:
                unlock_file(f)

        return list(buffer)

    def get_all_iterator(self):
        """
        生成器：逐条迭代所有记忆 (用于全量分析或 RAG 索引构建)
        不会一次性加载到内存
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = self._parse_line(line)
                if parsed:
                    yield parsed

    def clear(self):
        """清空所有记忆"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            pass  # 截断文件


# ==========================================
# 模拟 AgentLoop 的使用示例
# ==========================================

def mock_agent_loop():
    # 1. 初始化记忆系统
    memory = AgentMemory("demo_agent.jsonl")

    # 模拟系统提示词 (通常只存一次，或者每次启动时动态注入，这里演示存入)
    # memory.add("system", "你是一个乐于助人的 AI 助手，擅长编程。")

    print("--- 开始对话循环 ---")

    # 模拟几轮对话
    conversation_flow = [
        ("user", "你好，我想学习 Python。"),
        ("assistant", "你好！Python 是一门非常棒的语言。你想从哪方面开始学？"),
        ("user", "推荐一些适合初学者的项目吧。"),
        ("assistant", "你可以尝试写一个待办事项列表 (ToDo List) 或者一个简单的爬虫。"),
        ("user", "爬虫怎么写？"),
        ("assistant", "你可以使用 requests 库来获取网页，用 BeautifulSoup 来解析..."),
    ]

    for role, text in conversation_flow:
        # 1. Agent 接收到输入，先存入记忆
        memory.add(role, text)

        # 2. 构建当前上下文 (只取最近 4 条，模拟有限的 Context Window)
        # 在实际 LLM 调用中，这里会将 context 拼接成 Prompt
        context = memory.get_recent_context(n=4)

        print(f"\n[{role.upper()}]: {text}")
        print(f"-> 当前送入模型的上下文长度: {len(context)} 条")
        # print(f"   上下文预览: {[c['content'][:20] for c in context]}")

    print("\n--- 对话结束，验证持久化 ---")

    # 3. 模拟程序重启 (重新实例化 AgentMemory)
    print("重新加载记忆系统...")
    new_memory = AgentMemory("demo_agent.jsonl")

    # 读取所有历史进行验证
    all_history = list(new_memory.get_all_iterator())
    print(f"成功从磁盘加载了 {len(all_history)} 条历史记录。")
    print(f"最后一条记录是: '{all_history[-1]['content']}' (角色: {all_history[-1]['role']})")


if __name__ == "__main__":
    mock_agent_loop()