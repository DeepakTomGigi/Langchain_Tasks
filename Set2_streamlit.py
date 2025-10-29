import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Initialize model
llm = ChatOpenAI(
    model="mistralai/devstral-small-2505:free",
    temperature=0.5,
    api_key=api_key,
    base_url=base_url,
)

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

memory = st.session_state.memory

# Tool functions
def count_words(text: str) -> str:
    count = len(text.strip().split())
    return f"Your sentence has {count} word{'s' if count != 1 else ''}."

def reverse_text(text: str) -> str:
    return " ".join(reversed(text.strip().split()))

def define_word(word: str) -> str:
    prompt = f"Define the word '{word}' in one short sentence."
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return "Sorry, I couldn't define that word."

def convert_case(text: str, mode: str) -> str:
    return text.upper() if mode == "upper" else text.lower()

def repeat_word(text: str) -> str:
    parts = text.strip().split()
    if len(parts) != 2 or not parts[1].isdigit():
        return "Use format: repeat <word> <count>"
    word, count = parts[0], int(parts[1])
    return " ".join([word] * count)

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

# Streamlit UI
st.set_page_config(page_title="Mini Language Utility Bot", layout="centered")
st.title("🧠 Mini Language Utility Bot")
st.markdown("Type a command like `count`, `reverse`, `define`, `upper`, `lower`, `repeat`, or `history`.")

user_input = st.text_input("Your message", key="input")

if user_input:
    with st.spinner("Thinking..."):
        tool_response = handle_command(user_input)
        if tool_response:
            st.markdown(f"**Bot:** {tool_response}")
            if user_input.lower().strip() != "history":
                memory.save_context({"input": user_input}, {"output": tool_response})
        else:
            response = llm.invoke(user_input)
            st.markdown(f"**Bot:** {response.content}")
            memory.save_context({"input": user_input}, {"output": response.content})

# Optional: show full history below
with st.expander("Conversation History"):
    messages = memory.load_memory_variables({}).get("chat_history", [])
    for msg in messages:
        role = "You" if msg.type == "human" else "Bot"
        st.markdown(f"**{role}:** {msg.content}")
