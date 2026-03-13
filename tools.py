import re
import sys
import subprocess
# ====================== 3. 代码工具管理器 ======================
class CodeToolManager:

    
    @staticmethod
    def generate_code(code_content: str) -> str:
        """生成代码并检查语法"""
        try:
           
            clean_code = code_content.strip()
            clean_code = re.sub(r'```python|```', '', clean_code)
            
            if not clean_code:
                return "错误：代码为空"
            
            compile(clean_code, '<string>', 'exec')
            return f"✅ 代码生成成功（语法检查通过）\n生成的代码：\n{clean_code}"
        except SyntaxError as e:
            # 增强错误信息：显示清理后的代码，方便定位
            clean_code = CodeToolManager.clean_code_func(code_content)
            return f"❌ 语法错误：{str(e)}\n清理后的代码：\n{clean_code}\n原始代码：\n{code_content}"


    

    @staticmethod
    def execute_code(code_content: str, context_code: str = "") -> str:
        """执行测试代码"""
        temp_file = "temp_code_run.py"
        try:
            # 清理代码
            # 清理代码格式，保留换行
            context_code = re.sub(r'```python|```', '', context_code).strip()
            code_content = re.sub(r'```python|```', '', code_content).strip()
            
            full_code = f"{context_code}\n\n{code_content}"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(full_code)
            
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=False,  # 不自动解码，返回bytes
                timeout=10
            )

            # 手动解码：先试GBK，再试UTF-8，失败则忽略错误
            stdout = result.stdout.decode("gbk", errors="ignore") if result.stdout else ""
            stderr = result.stderr.decode("gbk", errors="ignore") if result.stderr else ""

            if result.returncode == 0:
                return f"✅ 执行成功\n输出：\n{stdout}"
            else:
                return f"❌ 执行失败\n错误：\n{stderr}"
        except Exception as e:
            return f"❌ 执行异常：{str(e)}"
