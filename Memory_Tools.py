# ============================================================
# Memory-Tools.py — Conversational Mistral Agent (fully working)
# ============================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


# ------------------------------------------------------------
# 1. Load environment variables
# ------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


# ------------------------------------------------------------
# 2. Initialize the Mistral model via OpenRouter
# ------------------------------------------------------------
llm = ChatOpenAI(
    model="mistralai/devstral-small-2505:free",
    temperature=0.4,
    max_tokens=256,
    api_key=api_key,
    base_url=base_url,
)


# ------------------------------------------------------------
# 3. Define helper tools
# ------------------------------------------------------------
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def greet(name: str) -> str:
    """Return a friendly greeting."""
    name = name.strip().replace('"', "").replace("'", "")
    return f"Hello {name}, welcome to the AI Agent demo!"

def weather(city: str) -> str:
    # Simulated results (you can extend this dictionary)
    conditions = {
        "dubai": "sunny with 33°C",
        "riyadh": "hot and dry with 36°C",
        "bengaluru": "cloudy with 27°C",
        "singapore": "humid with 30°C",
    }
    info = conditions.get(city.lower(), "not available right now")
    return f"The current weather in {city.title()} is {info}."


# ------------------------------------------------------------
# 4. Initialize memory
# ------------------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# ------------------------------------------------------------
# 5. Conversational loop
# ------------------------------------------------------------
print("\n=== Start chatting with your Agent ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("\nConversation ended.")
        break

    # Handle Multiply command
    if user_input.lower().startswith("multiply"):
        try:
            parts = user_input.split()
            a, b = int(parts[1]), int(parts[2])
            result = multiply(a, b)
            print("Agent:", result)
            memory.save_context({"input": user_input}, {"output": str(result)})
            continue
        except Exception:
            print("Agent: Please use 'Multiply a b' format.")
            continue

    # Handle Greet command
    if user_input.lower().startswith("greet"):
        try:
            name = " ".join(user_input.split()[1:]).strip()
            if not name:
                print("Agent: Please specify a name. Example: greet Abdullah")
                continue
            greeting = greet(name)
            print("Agent:", greeting)
            memory.save_context({"input": user_input}, {"output": greeting})
            continue
        except Exception as e:
            print("Agent: Could not greet properly:", e)
            continue

    # Handle name introduction
    if "my name is" in user_input.lower():
        name = user_input.split("is")[-1].strip()
        memory.save_context({"input": user_input}, {"output": name})
        print("Agent:", greet(name))
        continue

    # Handle asking for name
    if "what" in user_input.lower() and "my name" in user_input.lower():
        messages = memory.load_memory_variables({}).get("chat_history", [])
        if messages:
            last_output = messages[-1].content
            print("Agent: You said your name is", last_output)
        else:
            print("Agent: I don't know your name yet.")
        continue

    # handle weather search
    if user_input.lower().startswith("weather"):
        try:
            city = user_input.split()[1]
            result = weather(city)
            print("Agent:", result)
            memory.save_context({"input": user_input}, {"output": str(result)})
            continue
        except Exception:
            print("Agent: Please use 'Weather City' format.")
            continue


    # Default: use LLM
    try:
        response = llm.invoke(user_input)
        print("Agent:", response.content)
        memory.save_context({"input": user_input}, {"output": response.content})
    except Exception as e:
        print("Error:", e)
