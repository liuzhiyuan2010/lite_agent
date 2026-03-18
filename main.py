
import sys
from colorama import init, Fore, Style
from agent import  AgentLoop

#from  audio import  TTS_KOKORO
# 初始化 colorama
init(autoreset=True)

CONFIG_PATH = "config/conf.json"


#     你叫小q  我现在想听音乐 打开E:\Program Files\NetEase\CloudMusic\cloudmusic.exe


if __name__ == "__main__":
    print(Fore.YELLOW + "type 'bye' to break \n")
    #tts = TTS_KOKORO()
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
            #tts.tts_run(response)
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



