
import colorama
# 初始化 colorama
colorama.init(autoreset=True)

import sys
from colorama import init, Fore, Style
from llm_client import LLMClient

# 初始化 colorama
init(autoreset=True)


def main():
    print(Fore.BLUE + "=== 模块化 Ollama 工具调用系统 ===")
    print(Fore.WHITE + "架构：Main -> LLMClient -> Tools Module")
    print(Fore.YELLOW + "输入 'bye' 退出\n")

    # 初始化客户端
    try:
        client = LLMClient()
    except Exception as e:
        print(Fore.RED + f"启动失败: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input(Fore.WHITE + "👤 请输入问题: ")

            if user_input.strip().lower() in ['bye', 'exit', 'quit']:
                print(Fore.BLUE + "再见！")
                break

            if not user_input.strip():
                continue

            # 调用解耦后的聊天方法
            response = client.chat(user_input)

            print(Fore.MAGENTA + f"\n🤖 AI: {response}")

            # 显示统计
            stats = client.stats
            print(
                Fore.WHITE + Style.DIM + f"   [Token] In: {stats['input_tokens']} | Out: {stats['output_tokens']} | Total: {stats['total_tokens']}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n程序中断退出")
            break
        except Exception as e:
            print(Fore.RED + f"发生未知错误: {e}")


if __name__ == "__main__":
    main()

