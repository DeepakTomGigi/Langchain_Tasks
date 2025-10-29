import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load env variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

# Initialize Mistral model
llm = ChatOpenAI(
    model="mistralai/devstral-small-2505:free",
    temperature=0.5,
    api_key=api_key,
    base_url=base_url,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tool: Word Counter
def count_words(text: str) -> str:
    count = len(text.strip().split())
    return f"Your sentence has {count} word{'s' if count != 1 else ''}."

# Tool: Reverse Text
def reverse_text(text: str) -> str:
    words = text.strip().split()
    reversed_words = " ".join(reversed(words))
    return reversed_words

# Tool: Vocabulary Helper (uses LLM)
def define_word(word: str) -> str:
    prompt = f"Define the word '{word}' in one short sentence."
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return "Sorry, I couldn't define that word."

# Tool: Case Converter
def convert_case(text: str, mode: str) -> str:
    if mode == "upper":
        return text.upper()
    elif mode == "lower":
        return text.lower()
    else:
        return "Invalid case mode. Use 'upper' or 'lower'."

# Tool: Word Repeater
def repeat_word(text: str) -> str:
    parts = text.strip().split()
    if len(parts) != 2 or not parts[1].isdigit():
        return "Use format: repeat <word> <count>"
    word, count = parts[0], int(parts[1])
    return " ".join([word] * count)

# Command router
def handle_command(user_input: str) -> str:
    lower = user_input.lower().strip()

    if lower == "history":
        messages = memory.load_memory_variables({}).get("chat_history", [])
        if not messages:
            return "No conversation history yet."
        history_lines = []
        for msg in messages:
            role = "You" if msg.type == "human" else "Bot"
            history_lines.append(f"{role}: {msg.content}")
        return "\n".join(history_lines)

    if lower.startswith("count "):
        return count_words(user_input[len("count "):])

    if lower.startswith("reverse "):
        return reverse_text(user_input[len("reverse "):])

    if lower.startswith("define "):
        return define_word(user_input[len("define "):])

    if lower.startswith("upper "):
        return convert_case(user_input[len("upper "):], "upper")

    if lower.startswith("lower "):
        return convert_case(user_input[len("lower "):], "lower")

    if lower.startswith("repeat "):
        return repeat_word(user_input[len("repeat "):])


    return None  # fallback to LLM

# Chat loop
print("=== Mini Language Utility Bot ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        tool_response = handle_command(user_input)
        if tool_response:
            print(f"Bot: {tool_response}\n")
            if user_input.lower().strip() != "history":
                memory.save_context({"input": user_input}, {"output": tool_response})
        else:
            response = llm.invoke(user_input)
            print(f"Bot: {response.content}\n")
            memory.save_context({"input": user_input}, {"output": response.content})
    except Exception as e:
        print("Bot: Sorry, something went wrong.\n")