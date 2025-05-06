import os
import re
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# Load environment variables (for OPENAI_API_KEY)
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
    print("---GENERATING PLAN---")
    goal = state["goal"]
    timeframe = state["timeframe"]

    # It's good practice to ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "error_message": "OpenAI API key not found. Please set it in your .env file.",
            "parsed_plan": [] # Ensure this key exists even on error
        }

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7) # Or "gpt-4o" or "gpt-4-turbo"

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert life coach and project planner. "
                "Your task is to break down a user's high-level goal into a structured, actionable weekly plan. "
                "Ensure the plan is realistic for the given timeframe. "
                "Output the plan in Markdown format. Each week should start with '## Week X: [Week Title]' "
                "followed by a bulleted list of tasks for that week. Do not include any introductory or concluding text outside this format."
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
        print(f"Error during LLM call: {e}")
        return {
            "error_message": f"Failed to generate plan: {str(e)}",
            "plan_markdown": "",
            "parsed_plan": [] # Ensure this key exists
        }

# 2. Node to parse the Markdown plan into a structured format
def parse_plan_node(state: PlannerState):
    print("---PARSING PLAN---")
    markdown_text = state.get("plan_markdown")
    if not markdown_text or state.get("error_message"): # If there was an error or no markdown, skip parsing
        return {"parsed_plan": state.get("parsed_plan", [])} # Return existing parsed_plan or empty list

    parsed_plan = []
    # Regex to find week titles and their subsequent tasks
    # It captures "Week X: Title" and then everything until the next "## Week" or end of string
    week_sections = re.split(r'(?=^## Week \d+)', markdown_text, flags=re.MULTILINE)

    for section in week_sections:
        section = section.strip()
        if not section.startswith("## Week"):
            continue

        lines = section.split('\n')
        week_title_match = re.match(r'## (Week \d+[:\s]*(.*))', lines[0])
        if not week_title_match:
            continue
        
        week_full_title = week_title_match.group(1).strip() # "Week X: Actual Title"
        
        tasks = []
        for line in lines[1:]:
            line = line.strip()
            # Match lines starting with typical markdown list markers
            if line.startswith(("* ", "- ", "+ ")) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                tasks.append(line[2:].strip()) # Remove marker and strip
            elif line: # Add non-empty lines that are not headers as tasks
                tasks.append(line)
        
        if tasks: # Only add week if it has tasks
            parsed_plan.append({"week_title": week_full_title, "tasks": tasks})
    
    if not parsed_plan and markdown_text: # If parsing failed but there was markdown
        # Fallback: treat the whole markdown as a single "week" of general advice
        parsed_plan.append({"week_title": "General Plan", "tasks": [line.strip() for line in markdown_text.split('\n') if line.strip()]})

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
        return {
            "plan": result.get("parsed_plan", []),
            "error_message": result.get("error_message")
        }
    except Exception as e:
        print(f"Error in LangGraph execution: {e}")
        return {
            "plan": [],
            "error_message": f"An unexpected error occurred: {str(e)}"
        }

if __name__ == '__main__':
    # Example usage (for testing planner.py directly)
    import asyncio

    async def test_planner():
        # test_goal = "Learn to play the guitar"
        # test_timeframe = "3 months"
        test_goal = "Write a fantasy novel"
        test_timeframe = "6 months"
        
        print(f"Generating plan for: '{test_goal}' in '{test_timeframe}'")
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
            print("\nNo plan generated and no error message.")

    asyncio.run(test_planner())
