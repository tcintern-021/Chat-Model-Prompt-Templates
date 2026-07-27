"""
AI Restaurant Branding & Gourmet Menu Generator
===============================================
A LangChain-powered sequential pipeline using Google Gemini/Gemma models.
Generates fine-dining restaurant branding concepts and structured gourmet menus.
"""

import os
import sys
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

# Initialize Gemini / Gemma LLM
llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=api_key,
    max_output_tokens=1024,
    temperature=0.7,
)

# Output Parser
output_parser = StrOutputParser()

# Chain 1: Restaurant Name & Branding Generation
prompt_name = PromptTemplate.from_template(
    "Suggest 3 elegant fine-dining restaurant names for {cuisine} cuisine along with a brief atmosphere description for each.\n"
    "Output ONLY the 3 items in this exact format without any introductory notes or scratchpad:\n"
    "1. [Name] - [Description]\n"
    "2. [Name] - [Description]\n"
    "3. [Name] - [Description]"
)
name_chain = prompt_name | llm | output_parser

# Chain 2: Gourmet Menu Generation
prompt_menu = PromptTemplate.from_template(
    "Given this restaurant branding concept:\n{restaurant_name}\n\n"
    "Create a single curated gourmet menu featuring sections for Antipasti, Main Courses, and Desserts.\n"
    "Under each section, list 3 signature dishes with enticing descriptions.\n"
    "Start immediately with the heading 'Antipasti' without any introductory notes or scratchpad."
)
menu_chain = prompt_menu | llm | output_parser

# Sequential Pipeline (Modern LCEL)
restaurant_pipeline = RunnablePassthrough.assign(
    restaurant_name=name_chain
).assign(menu_items=menu_chain)


def generate_restaurant_concept(cuisine: str = "Italian") -> Dict[str, Any]:
    """Execute the restaurant branding and menu generation pipeline."""
    return restaurant_pipeline.invoke({"cuisine": cuisine})


if __name__ == "__main__":
    cuisine_type = "Italian"
    print(f"🚀 Running AI Restaurant Branding & Menu Generator for [{cuisine_type}] cuisine...\n")

    response = generate_restaurant_concept(cuisine_type)

    print("=" * 50)
    print("🍽️ RESTAURANT BRANDING & CONCEPT")
    print("=" * 50)
    print(response["restaurant_name"])

    print("\n" + "=" * 50)
    print("📜 GOURMET MENU CURATION")
    print("=" * 50)
    print(response["menu_items"])