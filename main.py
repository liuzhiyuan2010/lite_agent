
import colorama
import sys
from colorama import init, Fore, Style
#from llm_client import LLMClient
from agent import  AgentLoop
# 初始化 colorama
init(autoreset=True)

CONFIG_PATH = "config/conf.json"

def main():
    print(Fore.YELLOW + "type 'bye' to break \n")

    # 初始化客户端
    try:
        client = AgentLoop(CONFIG_PATH)
    except Exception as e:
        print(Fore.RED + f"start failed: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input(Fore.WHITE + "👤 >>>: ")

            if user_input.strip().lower() in ['bye', 'exit', 'quit']:
                print(Fore.BLUE + "Bye！")
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
            print("\nabort")
            break
        except Exception as e:
            print(Fore.RED + f"error: {e}")


if __name__ == "__main__":
    main()