from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os # For path joining

# Import the planner logic
from .planner import create_plan

# Initialize FastAPI app
app = FastAPI(title="LifeLoom API")

# Mount static files (for CSS)
# Use os.path.join for cross-platform compatibility
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configure Jinja2 templates
templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=templates_dir)

class PlanRequest(BaseModel):
    goal: str
    timeframe: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main page with the goal input form."""
    return templates.TemplateResponse("index.html", {"request": request, "plan": None, "goal_input": None, "timeframe_input": None})

@app.post("/plan", response_class=HTMLResponse)
async def get_plan(
    request: Request,
    goal: str = Form(...),
    timeframe: str = Form(...)
):
    """
    Receives goal and timeframe, generates a plan, and returns it.
    """
    print(f"Received goal: {goal}, timeframe: {timeframe}") # For debugging

    # Call the planner function from planner.py
    plan_result = await create_plan(goal, timeframe)
    
    # The plan_result will be a dict like: {"plan": [...], "error_message": "..."}
    # We pass these directly to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "goal_input": goal,
        "timeframe_input": timeframe,
        "plan": plan_result.get("plan"),
        "error_message": plan_result.get("error_message")
    })

if __name__ == "__main__":
    import uvicorn
    # Run from the 'lifeloom' directory: python -m app.main
    # This direct run is mostly for quick testing if you modify uvicorn call.
    # Recommended to run with: uvicorn app.main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
