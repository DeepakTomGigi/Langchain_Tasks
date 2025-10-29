import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Load environment
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise ValueError("OPENROUTER_KEY missing in .env")

# Initialize model
llm = ChatOpenAI(
    model="mistralai/devstral-small-2505:free",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
)

# Setup memory
memory = ChatMessageHistory(return_messages=True)

# Prompt template with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that remembers past messages."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Session-based memory retriever
def get_session_history(session_id: str):
    return memory

# Chain with memory
parser = StrOutputParser()
chain_with_memory = RunnableWithMessageHistory(
    prompt | llm | parser,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Persistent NoteKeeper
NOTES_FILE = "notes_store.json"

def load_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

# Tool functions
def summarize(text: str) -> str:
    prompt = f"Summarize this in one sentence:\n{text}"
    return llm.invoke(prompt).content.strip()

def analyze_sentiment(text: str) -> str:
    prompt = f"Classify the sentiment of this text as Positive, Neutral, or Negative:\n{text}"
    return llm.invoke(prompt).content.strip()

def add_note(text: str) -> str:
    notes = load_notes()
    notes.append(text.strip())
    save_notes(notes)
    return f'Noted: "{text.strip()}"'

def get_notes() -> str:
    notes = load_notes()
    if not notes:
        return "You currently have 0 notes."
    return f"You have {len(notes)} note(s): " + "; ".join(f'"{n}"' for n in notes)

def improve(text: str) -> str:
    prompt = f"Rewrite this to be clearer and more professional. Explain the change:\n{text}"
    return llm.invoke(prompt).content.strip()

def classify_priority(task: str) -> str:
    prompt = f"Classify the priority of this task as HIGH, MEDIUM, or LOW:\n{task}"
    return llm.invoke(prompt).content.strip()

# Command router
def handle_command(user_input: str) -> str:
    lower = user_input.lower().strip()

    if lower.startswith("summarize"):
        return summarize(user_input[len("summarize"):].strip())

    if lower.startswith("analyze") or lower.startswith("sentiment"):
        return analyze_sentiment(user_input.split(None, 1)[1])

    if lower.startswith("note "):
        return add_note(user_input[len("note"):].strip())

    if lower in {"get notes", "get_notes", "notes"}:
        return get_notes()

    if lower.startswith("improve") or lower.startswith("rewrite"):
        return improve(user_input.split(None, 1)[1])

    if lower.startswith("priority"):
        return classify_priority(user_input[len("priority"):].strip())

    return None  # fallback to LLM

# Chat loop
print("=== Chat with Memory and Tools Enabled ===")
print("Type 'exit' to quit.\n")

session_id = "user-session-1"
os.makedirs("logs", exist_ok=True)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        tool_response = handle_command(user_input)
        if tool_response:
            print(f"Assistant: {tool_response}\n")
        else:
            response = chain_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"Assistant: {response}\n")

        # Log interaction
        log_entry = {
            "session_id": session_id,
            "user_input": user_input,
            "assistant_response": tool_response or response,
            "timestamp": datetime.now().isoformat()
        }
        with open("logs/chat_memory_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print("Assistant: Sorry, something went wrong.\n")
