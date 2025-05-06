# LifeLoom — AI-Powered Goal-to-Plan Assistant

LifeLoom helps users turn life goals into actionable weekly plans using FastAPI, LangGraph (LLMs), and a clean HTML + CSS interface. Enter a goal, select a timeframe, and receive a personalized roadmap for success.

🌟 **Features**
✨ Convert high-level goals into structured weekly tasks.
⚡ FastAPI-powered backend with Jinja2 templating.
🧠 LangGraph (via LangChain) LLM workflow for generating plans.
🎨 Clean, responsive UI with pure HTML + CSS.
📂 Lightweight, framework-free front end.

🛠️ **Tech Stack**
- **Backend**: FastAPI
- **LLM Workflow**: LangGraph + OpenAI (gpt-3.5-turbo by default)
- **Templating**: Jinja2
- **Frontend**: HTML + CSS


## How LifeLoom Works
1.  **User Input**:
    *   Goal (e.g., “Learn Python”)
    *   Timeframe (e.g., “2 months”)
2.  FastAPI receives the input and calls the `planner` module.
3.  LangGraph builds an LLM prompt to break the goal into weekly tasks.
4.  The result is parsed from Markdown into a structured list of weeks and tasks.
5.  The structured plan is rendered using a Jinja2 HTML template.
6.  **Output**: A beautiful weekly roadmap with tasks.

## Setup and Running

1.  **Clone the repository.**

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    Create a `.env` file in the `lifeloom` root directory and add your API key:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    (A `.env.example` file can be provided as a template).

5.  **Run the FastAPI application:**
    Navigate to the `lifeloom` root directory in your terminal.
    ```bash
    uvicorn app.main:app --reload
    ```
    The `--reload` flag enables auto-reloading when you make code changes.

6.  **Open your browser:**
    Go to `http://127.0.0.1:8000`

## To-Do / Potential Enhancements
- [ ] More robust Markdown parsing (e.g., using a dedicated library if LLM output varies too much).
- [ ] Allow users to customize the number of tasks per week or plan intensity.
- [ ] Add ability to save/load plans (e.g., to a simple database or local storage).
- [ ] Implement user authentication if saving plans online.
- [ ] Add a "regenerate week" or "edit task" feature.
- [ ] Explore different LLM models or fine-tuning for better plan quality.
- [ ] Unit and integration tests.
