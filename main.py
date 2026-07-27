import os
import sys
import time
import warnings
from typing import Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress deprecation warning for RunnableWithMessageHistory
# (In production, LangGraph's built-in persistence is the modern alternative)
warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory.*")

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# =====================================================================
# ENVIRONMENT SETUP
# =====================================================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "API Key missing! Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
    )

# =====================================================================
# SHARED CHAT MODEL
# =====================================================================
llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=api_key,
    max_output_tokens=1024,
    temperature=0.7,
)

output_parser = StrOutputParser()

core_question = "What is LangChain and why should developers use it instead of calling LLM APIs directly?"


# =====================================================================
# DEMO 1: PROMPT TEMPLATES
# =====================================================================
# Asking the exact same core question using 3 completely different
# prompt templates to observe how phrasing and persona alter the response.

templates = {
    "1. Explain Like I'm 5 (ELI5)": """You are explaining concepts to a 5-year-old child.
Use fun analogies like Lego bricks or building blocks. Keep it simple and under 3 sentences.
Do NOT include any preliminary reasoning, scratchpad, or meta-commentary. Output ONLY the final explanation directly.

Question: {question}""",

    "2. Senior AI Systems Architect": """You are a Senior AI Systems Architect addressing a software engineering team.
Focus on architectural advantages: modularity, model-agnostic design, LCEL (LangChain Expression Language) pipelines, and structured output parsing.
Use professional technical terminology and concise bullet points.
Do NOT include any preliminary reasoning, scratchpad, or meta-commentary. Output ONLY the final explanation directly.

Question: {question}""",

    "3. 17th-Century Pirate Captain": """Ahoy! You are a seasoned pirate captain who is also an expert AI engineer.
Explain the answer using hearty pirate slang, seafaring metaphors, and treasure-hunting analogies!
Do NOT include any preliminary reasoning, scratchpad, or meta-commentary. Output ONLY the final speech directly.

Question: {question}"""
}


def demo_prompt_templates() -> None:
    """Demo 1: Execute LCEL chains across different prompt templates."""
    print("=" * 75)
    print("🎭 DEMO 1: PROMPT TEMPLATE EXPERIMENTATION")
    print("=" * 75)
    print(f"Core Question: '{core_question}'\n")
    print("Executing LCEL pipelines across 3 distinct Prompt Templates...\n")

    for persona, template_str in templates.items():
        prompt = PromptTemplate.from_template(template_str)
        chain = prompt | llm | output_parser

        response = chain.invoke({"question": core_question})

        print("-" * 75)
        print(f"🎭 PROMPT TEMPLATE: {persona}")
        print("-" * 75)
        print(response.strip())
        print("\n")

    print("✅ Prompt template demo complete!\n")


# =====================================================================
# DEMO 2: STRUCTURED OUTPUT PARSING (Pydantic)
# =====================================================================
# Instead of StrOutputParser() returning freeform text, we force the
# LLM to return a strict, validated JSON object using PydanticOutputParser.


class ExplanationSchema(BaseModel):
    """Structured schema for an AI concept explanation."""
    summary: str = Field(description="A concise 1-2 sentence summary of the concept")
    analogy: str = Field(description="A creative real-world analogy to explain the concept")
    key_benefits: list[str] = Field(description="Exactly 3 key benefits, each as a short sentence")
    difficulty_level: str = Field(description="One of: beginner, intermediate, advanced")


def demo_structured_output() -> None:
    """Demo 2: Force the LLM to return a validated Pydantic object."""
    print("=" * 75)
    print("🧱 DEMO 2: STRUCTURED OUTPUT PARSING (Pydantic)")
    print("=" * 75)

    # PydanticOutputParser generates format instructions from the schema
    parser = PydanticOutputParser(pydantic_object=ExplanationSchema)

    print("📋 Format instructions injected into prompt:")
    print("-" * 75)
    print(parser.get_format_instructions())
    print("-" * 75 + "\n")

    # Lower temperature for more deterministic structured responses
    structured_llm = ChatGoogleGenerativeAI(
        model="gemma-4-26b-a4b-it",
        google_api_key=api_key,
        max_output_tokens=1024,
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template="""You are a helpful AI educator. Explain the following topic clearly and concisely.

IMPORTANT: You MUST respond with ONLY a valid JSON object. No extra text, no markdown, no code fences.
Do NOT include any preliminary reasoning, scratchpad, or meta-commentary.

{format_instructions}

Topic: {question}""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # LCEL Pipeline: Prompt -> LLM -> Pydantic Parser
    chain = prompt | structured_llm | parser

    print(f"Question: '{core_question}'\n")
    print("Invoking chain with Pydantic schema enforcement...\n")

    result = chain.invoke({"question": core_question})

    print(f"📦 Return Type: {type(result).__name__}\n")
    print(f"📝 Summary:\n   {result.summary}\n")
    print(f"🎯 Analogy:\n   {result.analogy}\n")
    print(f"📊 Difficulty Level: {result.difficulty_level}\n")
    print("✅ Key Benefits:")
    for i, benefit in enumerate(result.key_benefits, 1):
        print(f"   {i}. {benefit}")

    print("\n" + "-" * 75)
    print("📋 Raw JSON Output:")
    print("-" * 75)
    print(result.model_dump_json(indent=2))
    print("\n✅ Structured output demo complete!\n")


# =====================================================================
# DEMO 3: REAL-TIME STREAMING
# =====================================================================
# .invoke() blocks until the full response is generated.
# .stream() yields tokens one-by-one for a live typewriter effect.


def demo_streaming() -> None:
    """Demo 3: Compare .invoke() blocking vs .stream() real-time output."""
    print("=" * 75)
    print("⚡ DEMO 3: REAL-TIME STREAMING COMPARISON")
    print("=" * 75)
    print(f"Question: '{core_question}'\n")

    prompt = PromptTemplate.from_template(
        """You are a knowledgeable AI educator. Give a clear, detailed explanation.
Do NOT include any preliminary reasoning, scratchpad, or meta-commentary. Output ONLY the final explanation directly.

Question: {question}"""
    )
    chain = prompt | llm | output_parser

    # --- METHOD A: .invoke() ---
    print("-" * 75)
    print("🐢 METHOD A: .invoke() — Blocking (full response at once)")
    print("-" * 75)

    invoke_start = time.perf_counter()
    full_response = chain.invoke({"question": core_question})
    invoke_duration = time.perf_counter() - invoke_start

    preview = full_response[:200] + "..." if len(full_response) > 200 else full_response
    print(preview)
    print(f"\n⏱️  Total wait time: {invoke_duration:.2f}s")
    print(f"   (User sees NOTHING until {invoke_duration:.2f}s have passed)\n")

    # --- METHOD B: .stream() ---
    print("-" * 75)
    print("🚀 METHOD B: .stream() — Real-time (token-by-token)")
    print("-" * 75)

    stream_start = time.perf_counter()
    time_to_first_token = None
    token_count = 0

    for chunk in chain.stream({"question": core_question}):
        if time_to_first_token is None:
            time_to_first_token = time.perf_counter() - stream_start
        print(chunk, end="", flush=True)
        token_count += 1

    stream_duration = time.perf_counter() - stream_start

    print(f"\n\n⏱️  Time to first token: {time_to_first_token:.2f}s")
    print(f"⏱️  Total stream time:   {stream_duration:.2f}s")
    print(f"📊 Chunks received:      {token_count}")

    # --- Summary ---
    print("\n" + "-" * 75)
    print("📊 PERFORMANCE SUMMARY")
    print("-" * 75)
    print(f"  .invoke() — User waits:          {invoke_duration:.2f}s before seeing anything")
    print(f"  .stream() — User sees first text: {time_to_first_token:.2f}s (then streams live)")
    if time_to_first_token and invoke_duration > 0:
        improvement = ((invoke_duration - time_to_first_token) / invoke_duration) * 100
        print(f"  🚀 Perceived speed improvement:   {improvement:.0f}% faster first response")
    print("\n✅ Streaming demo complete!\n")


# =====================================================================
# DEMO 4: CONVERSATIONAL MEMORY
# =====================================================================
# Each .invoke() is stateless by default. RunnableWithMessageHistory
# automatically injects conversation history so the model remembers context.

session_store: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieve or create a chat history for the given session."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def demo_conversational_memory() -> None:
    """Demo 4: Interactive multi-turn chatbot with memory."""
    SESSION_ID = "user-session-001"

    print("=" * 75)
    print("🧠 DEMO 4: CONVERSATIONAL MEMORY (Multi-Turn Chatbot)")
    print("=" * 75)
    print("Chat with an AI that REMEMBERS your conversation!")
    print("Try asking a question, then ask a follow-up that references")
    print("the previous answer (e.g., 'Can you elaborate on point 2?').")
    print("")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'history' to view the stored conversation memory.")
    print("=" * 75 + "\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and friendly AI assistant specializing in AI engineering and LangChain.
You remember everything the user has said in this conversation.
When referencing previous messages, be specific about what was discussed.
Do NOT include any preliminary reasoning, scratchpad, or meta-commentary. Respond directly."""),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | output_parser

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\n👋 Ending session...\n")
            break

        if user_input.lower() == "history":
            history = get_session_history(SESSION_ID)
            print("\n" + "-" * 75)
            print("📜 CONVERSATION HISTORY:")
            print("-" * 75)
            if not history.messages:
                print("   (empty — no messages yet)")
            for msg in history.messages:
                role = "👤 Human" if msg.type == "human" else "🤖 AI"
                content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                print(f"   {role}: {content}")
            print("-" * 75 + "\n")
            continue

        print("🤖 AI: ", end="", flush=True)
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        print(response)
        print()

    # Show full log after exit
    history = get_session_history(SESSION_ID)
    if history.messages:
        print("=" * 75)
        print("📜 FULL CONVERSATION LOG:")
        print("=" * 75)
        for i, msg in enumerate(history.messages, 1):
            role = "👤 HUMAN" if msg.type == "human" else "🤖 AI"
            print(f"\n--- Turn {(i + 1) // 2} ({role}) ---")
            print(msg.content)
        print("\n" + "=" * 75)
        print(f"✅ Total turns: {len(history.messages) // 2}")
        print("=" * 75)

    print("\n✅ Memory demo complete!\n")


# =====================================================================
# MAIN MENU
# =====================================================================
def show_menu() -> None:
    """Display the interactive demo menu."""
    print("\n" + "=" * 75)
    print("🦜🔗 AI ENGINEERING TRACK: INTRODUCTION TO LANGCHAIN")
    print("=" * 75)
    print("")
    print("  [1] 🎭 Prompt Templates     — Same question, 3 personas")
    print("  [2] 🧱 Structured Output    — Force LLM to return validated JSON")
    print("  [3] ⚡ Streaming            — Real-time vs blocking comparison")
    print("  [4] 🧠 Conversational Memory — Multi-turn chatbot with context")
    print("  [5] 🚀 Run All (1-3)        — Execute demos 1-3 sequentially")
    print("  [0] ❌ Exit")
    print("")
    print("=" * 75)


if __name__ == "__main__":
    while True:
        show_menu()

        try:
            choice = input("Select a demo [0-5]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if choice == "1":
            demo_prompt_templates()
        elif choice == "2":
            demo_structured_output()
        elif choice == "3":
            demo_streaming()
        elif choice == "4":
            demo_conversational_memory()
        elif choice == "5":
            demo_prompt_templates()
            demo_structured_output()
            demo_streaming()
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("⚠️  Invalid choice. Please enter a number between 0 and 5.")

        input("\nPress Enter to return to the menu...")