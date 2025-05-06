import os
import re
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq # Import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- Define State for the Graph ---
class PlannerState(TypedDict):
    goal: str
    timeframe: str
    plan_markdown: str # Raw output from LLM
    parsed_plan: List[dict] # Structured plan for Jinja2
    error_message: str # To capture any errors

# --- Define Nodes for the Graph ---

# 1. Node to generate the plan using LLM
async def generate_plan_node(state: PlannerState):
    print("---GENERATING PLAN (USING GROQ)---")
    goal = state["goal"]
    timeframe = state["timeframe"]

    # Check for Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return {
            "error_message": "Groq API key not found. Please set GROQ_API_KEY in your .env file.",
            "parsed_plan": []
        }

    # Initialize ChatGroq
    # Common models on Groq: "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
    # Check Groq's documentation for the latest available models.
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192", # Or "mixtral-8x7b-32768"
            temperature=0.7
        )
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
        return {
            "error_message": f"Failed to initialize LLM client: {str(e)}",
            "parsed_plan": []
        }

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert life coach and project planner. "
                "Your task is to break down a user's high-level goal into a structured, actionable weekly plan. "
                "Ensure the plan is realistic for the given timeframe. "
                "Output the plan in Markdown format. Each week should start with '## Week X: [Week Title]' "
                "followed by a bulleted list of tasks for that week. Do not include any introductory or concluding text outside this format. "
                "Be concise and focus on the tasks for each week." # Added a bit more for clarity, model dependent
            ),
            (
                "human",
                "Goal: {goal}\nTimeframe: {timeframe}\n\nProvide a weekly plan.",
            ),
        ]
    )

    chain = prompt_template | llm

    try:
        response = await chain.ainvoke({"goal": goal, "timeframe": timeframe})
        return {"plan_markdown": response.content, "error_message": None}
    except Exception as e:
        print(f"Error during LLM call (Groq): {e}")
        return {
            "error_message": f"Failed to generate plan with Groq: {str(e)}",
            "plan_markdown": "",
            "parsed_plan": []
        }

# 2. Node to parse the Markdown plan into a structured format
# This node (parse_plan_node) remains largely the same, as it processes markdown.
# However, you MIGHT need to tweak the regex or logic if the chosen Groq model
# formats its markdown output slightly differently than OpenAI's GPT models.
def parse_plan_node(state: PlannerState):
    print("---PARSING PLAN---")
    markdown_text = state.get("plan_markdown")
    if not markdown_text or state.get("error_message"):
        return {"parsed_plan": state.get("parsed_plan", [])}

    parsed_plan = []
    # Regex to find week titles and their subsequent tasks
    # It captures "## Week X: Title" and then everything until the next "## Week" or end of string
    # Using re.DOTALL to make '.' match newlines as well, in case tasks span multiple lines before a new list item.
    week_sections = re.split(r'(?=^## Week \d+)', markdown_text, flags=re.MULTILINE)

    for section in week_sections:
        section = section.strip()
        if not section.startswith("## Week"):
            continue

        lines = section.split('\n')
        week_title_match = re.match(r'## (Week \d+[:\s]*(.*))', lines[0])
        if not week_title_match:
            # Try a more lenient match if the title is just "## Week X"
            week_title_match_simple = re.match(r'## (Week \d+)', lines[0])
            if week_title_match_simple:
                week_full_title = week_title_match_simple.group(1).strip()
            else:
                continue # Skip if no valid week header
        else:
            week_full_title = week_title_match.group(1).strip()
        
        tasks = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith(("* ", "- ", "+ ")) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                task_text = line[2:].strip()
                if task_text: # Only add non-empty tasks
                    tasks.append(task_text)
            elif line and not line.startswith("##"): # Add non-empty lines that are not headers as tasks
                tasks.append(line)
        
        if tasks:
            parsed_plan.append({"week_title": week_full_title, "tasks": tasks})
    
    if not parsed_plan and markdown_text:
        # Fallback: treat the whole markdown as a single "week" of general advice
        # Split by newline, filter empty lines and markdown headers if any accidentally got in
        fallback_tasks = [line.strip() for line in markdown_text.split('\n') if line.strip() and not line.startswith("#")]
        if fallback_tasks:
            parsed_plan.append({"week_title": "General Plan Outline", "tasks": fallback_tasks})

    return {"parsed_plan": parsed_plan}


# --- Define the Graph ---
workflow = StateGraph(PlannerState)

workflow.add_node("generate_plan", generate_plan_node)
workflow.add_node("parse_plan", parse_plan_node)

workflow.set_entry_point("generate_plan")
workflow.add_edge("generate_plan", "parse_plan")
workflow.add_edge("parse_plan", END)

app_graph = workflow.compile()


# --- Function to be called by FastAPI ---
async def create_plan(goal: str, timeframe: str) -> dict:
    inputs = {"goal": goal, "timeframe": timeframe, "plan_markdown": "", "parsed_plan": [], "error_message": None}
    try:
        result = await app_graph.ainvoke(inputs)
        # Log the raw markdown if parsing fails, for easier debugging
        if not result.get("parsed_plan") and result.get("plan_markdown") and not result.get("error_message"):
            print("---PARSING MIGHT HAVE FAILED OR PRODUCED NO STRUCTURED DATA---")
            print("Raw Markdown Output from LLM:")
            print(result.get("plan_markdown"))
            print("---END OF RAW MARKDOWN---")

        return {
            "plan": result.get("parsed_plan", []),
            "error_message": result.get("error_message")
        }
    except Exception as e:
        print(f"Error in LangGraph execution with Groq: {e}")
        return {
            "plan": [],
            "error_message": f"An unexpected error occurred while generating your plan: {str(e)}"
        }

if __name__ == '__main__':
    import asyncio

    async def test_planner():
        test_goal = "Learn intermediate Spanish"
        test_timeframe = "3 months"
        
        print(f"Generating plan for: '{test_goal}' in '{test_timeframe}' using Groq")
        result = await create_plan(test_goal, test_timeframe)
        
        if result["error_message"]:
            print(f"\nError: {result['error_message']}")
        elif result["plan"]:
            print("\nGenerated Plan:")
            for week_data in result["plan"]:
                print(f"\n{week_data['week_title']}")
                for task in week_data['tasks']:
                    print(f"  - {task}")
        else:
            print("\nNo plan generated or an issue occurred, and no specific error message was set in the result.")
            # This could happen if plan_markdown was generated but parsing resulted in an empty list
            # and no error_message was explicitly set in the parse_plan_node for this case.

    asyncio.run(test_planner())