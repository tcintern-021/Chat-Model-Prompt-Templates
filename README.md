# 🍽️ AI Restaurant Branding & Menu Generator

An AI-powered sequential chain application built with **modern LangChain (LCEL)** and **Google Generative AI (Gemini / Gemma)**. This project automatically creates cohesive, professional branding and gourmet menus for any cuisine type.

---

## 🌟 Overview

Designing a restaurant concept requires consistency between branding and culinary offerings. This project utilizes LangChain Expression Language (LCEL) to create a multi-stage AI pipeline:
1. **Branding Consultant (Chain 1):** Takes a cuisine type (e.g., *"Italian"*) and generates 3 elegant, professional restaurant names along with atmosphere descriptions.
2. **Executive Chef (Chain 2):** Takes the generated branding concepts and crafts a curated, multi-course gourmet menu tailored to the restaurant's vibe.
3. **Reasoning Filter (`clean_parser`):** A custom LCEL output parser that cleanly extracts final results by stripping out internal LLM chain-of-thought and scratchpad reasoning blocks.

---

## ✨ Key Features

- **Modern LCEL Architecture:** Fully upgraded from legacy `LLMChain` and `SequentialChain` to modern pipe syntax (`|`) and `RunnablePassthrough.assign()`, eliminating all LangChain deprecation warnings.
- **Sequential Context Passing:** Automatically passes intermediate outputs from one model prompt into the next for a unified workflow.
- **Clean Output Parsing:** Uses custom `RunnableLambda` parsers to ensure outputs look polished, professional, and ready for presentation without conversational filler or brainstorming noise.
- **Configurable Models:** Powered by `langchain_google_genai` (supporting models like `gemma-4-26b-a4b-it`, Gemini Pro, etc.).

---

## 🛠️ Prerequisites

- **Python:** 3.9 or higher
- **Google API Key:** A valid API key from Google AI Studio / Vertex AI

---

## 🚀 Installation & Setup

1. **Clone or Navigate to the Workspace:**
   ```bash
   cd "LangChain-hat model"
   ```

2. **Create and Activate a Virtual Environment:**
   - **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   Install the required modern LangChain packages and Google GenAI SDK:
   ```bash
   pip install langchain langchain-core langchain-google-genai python-dotenv
   ```

---

## 🔑 Configuration

Create a `.env` file in the root directory and add your Google API key:

```env
GOOGLE_API_KEY="your_api_key_here"
# Alternatively, you can use GEMINI_API_KEY="your_api_key_here"
```

---

## 💻 Usage

Run the main script using your Python environment:

```bash
python main.py
```

### 📋 Example Output

```text
Restaurant Name:
1. Velluto — An intimate, dimly lit sanctuary designed for romantic fine dining and rich culinary textures.
2. L'Antica Eredità — A refined, warm space celebrating timeless culinary traditions through an upscale, heritage-inspired lens.
3. Luce d'Oro — A bright, high-end destination featuring sun-drenched interiors and contemporary Italian luxury.

Menu Items:
Appetizers (Antipasti)
- Burrata Pugliese: Creamy burrata served with heirloom tomato confit, basil oil, and aged balsamic pearls.
- Carpaccio di Manzo: Thinly sliced prime beef tenderloin with wild arugula, shaved Parmigiano-Reggiano, and white truffle oil.

Main Courses (Primi / Secondi)
- Tagliolini al Tartufo: Hand-spun silk pasta tossed in a rich Parmigiano butter emulsion, topped with freshly shaved black truffle.
- Branzino al Forno: Mediterranean sea bass roasted with olive oil, capers, cherry tomatoes, and fresh thyme.
- Bistecca alla Fiorentina: Prime dry-aged T-bone steak grilled over wood fire, finished with rosemary and Tuscan olive oil.

Desserts (Dolci)
- Tiramisu Tradizionale: Layers of espresso-soaked savoiardi ladyfingers and whipped mascarpone cream, dusted with Valrhona cocoa.
- Panna Cotta al Limoncello: Silky vanilla bean cream infused with Amalfi lemon zest, served with macerated wild berries.
```

---

## 📂 Project Structure

```text
LangChain-hat model/
├── venv/             # Python virtual environment
├── .env              # Environment variables (API keys)
├── main.py           # Core LCEL sequential chain pipeline
└── README.md         # Project documentation
```

---

## 🧠 How It Works (Code Architecture)

```mermaid
graph TD
    A[Input: Cuisine Type] -->|{"cuisine": "Italian"}| B[Prompt 1: Branding Consultant]
    B --> C[LLM: Google GenAI]
    C --> D[clean_parser: Extracts Names]
    D -->|Adds "restaurant_name"| E[RunnablePassthrough.assign]
    E --> F[Prompt 2: Executive Chef]
    F --> G[LLM: Google GenAI]
    G --> H[clean_parser: Extracts Menu]
    H -->|Adds "menu_items"| I[Final Dictionary Output]
```

1. **`RunnablePassthrough.assign(restaurant_name=name_chain)`**: Evaluates `name_chain` using the initial input (`cuisine`), cleans the response, and attaches `"restaurant_name"` to the data dictionary.
2. **`RunnablePassthrough.assign(menu_items=menu_chain)`**: Uses the newly generated `"restaurant_name"` to evaluate `menu_chain`, generating a structured menu that matches the names and concept.
