from agent import FruitAgent


# Commands that exit the program.
EXIT_WORDS = {"exit", "quit"}


# Display the welcome message and commands.
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


# Run the interactive chat loop.
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


# Run the chat loop when executed directly.
if __name__ == "__main__":
    main()
