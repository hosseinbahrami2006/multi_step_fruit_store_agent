from agent import FruitAgent


# Words that close the program.
EXIT_WORDS = {"exit", "quit"}


# print_welcome
# Shows the program name, the commands, and an example message.
def print_welcome():
    print("Fruit Store Agent")
    print("Powered by LangChain, LangGraph, and local Ollama (qwen2.5:3b)")
    print()
    print("Commands:")
    print("- confirm order")
    print("- cancel order")
    print("- reset")
    print("- exit")
    print()
    print("Example:")
    print(
        "I need 3 kg of apples and 2 kg of bananas. "
        "My budget is 5 USD. Reduce quantities if necessary."
    )


# main
# Starts the agent and runs the chat loop.
# 1. Read a message from the user.
# 2. Skip empty messages.
# 3. Stop on "exit" or "quit" (any letter case, extra spaces allowed).
# 4. Send every other message to the agent and print the answer.
# Ctrl+C or Ctrl+D also closes the program safely.
def main():
    agent = FruitAgent(verbose=True)
    print_welcome()

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not message:
            continue

        if message.casefold() in EXIT_WORDS:
            print("Goodbye.")
            break

        answer = agent.chat(message)
        print(f"\nAgent:\n{answer}")


# Run main() only when this file is started directly:
#   python main.py
if __name__ == "__main__":
    main()
