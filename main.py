from agent import FruitAgent


def main():
    agent = FruitAgent(verbose=True)

    print("Advanced Fruit Store Agent")
    print("Powered by local Ollama and qwen2.5:3b")
    print()
    print("Commands:")
    print("- confirm order")
    print("- cancel order")
    print("- reset")
    print("- exit")
    print()
    print("Example:")
    print(
        "Prepare an order with 3 kg of apples, "
        "2 kg of bananas, and 3 kg of oranges. "
        "My budget is $25. Reduce quantities if necessary, "
        "but keep all three fruits."
    )

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not message:
            continue

        if message.lower() == "exit":
            print("Goodbye.")
            break

        answer = agent.chat(message)
        print(f"\nAgent:\n{answer}")


if __name__ == "__main__":
    main()