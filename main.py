import os
import sys
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Load environment configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "API Key missing! Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
    )

# =====================================================================
# 1. CREATE YOUR FIRST CHAT MODEL
# =====================================================================
# Initializing Google Generative AI (Gemini/Gemma) wrapper
llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=api_key,
    max_output_tokens=1024,
    temperature=0.7,
)

# Output Parser: Transforms raw LLM message output into a clean string
output_parser = StrOutputParser()

# =====================================================================
# 2. EXPERIMENT WITH PROMPT TEMPLATES
# =====================================================================
# Asking the exact same core question using 3 completely different prompt templates
# to observe how phrasing and persona alter the model's response.

core_question = "What is LangChain and why should developers use it instead of calling LLM APIs directly?"

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


def run_prompt_experiments() -> Dict[str, str]:
    """Execute the LCEL chains across different prompt templates."""
    results = {}
    for persona, template_str in templates.items():
        # Build Prompt Template
        prompt = PromptTemplate.from_template(template_str)
        
        # Build LCEL Pipeline (Prompt -> LLM -> Parser)
        chain = prompt | llm | output_parser
        
        # Invoke chain
        response = chain.invoke({"question": core_question})
        results[persona] = response.strip()
    return results


if __name__ == "__main__":
    print("=" * 75)
    print("🦜🔗 AI ENGINEERING TRACK: INTRODUCTION TO LANGCHAIN")
    print("=" * 75)
    print(f"Core Question: '{core_question}'\n")
    print("Executing LCEL pipelines across 3 distinct Prompt Templates...\n")

    experiment_results = run_prompt_experiments()

    for persona, answer in experiment_results.items():
        print("-" * 75)
        print(f"🎭 PROMPT TEMPLATE: {persona}")
        print("-" * 75)
        print(answer)
        print("\n")
    print("=" * 75)
    print("✅ Experimentation complete! Check README.md for architecture deep-dive.")
    print("=" * 75)