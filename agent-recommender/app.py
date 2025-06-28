# from openai import OpenAI
# from agent_recommender_ui import recommender_ui

# client = OpenAI(api_key="sk-proj-1234567890")

# def get_response(prompt):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content



# def main():
#     recommender_ui(get_response)

# if __name__ == "__main__":
#     main()

# AI Coding Agent Recommendation System
# Complete implementation with FastAPI

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from datetime import datetime
from dataclasses import asdict
from typing import List

# Import from recommendation_engine
from recommendation_engine import (
    AgentKnowledgeBase, 
    TaskAnalyzer, 
    RecommendationEngine, 
    CodingAgent, 
    TaskAnalysis, 
    Recommendation
)

# Pydantic models for API
class TaskRequest(BaseModel):
    task_description: str = Field(..., min_length=1, description="Description of the coding task")

class TaskAnalysisResponse(BaseModel):
    task_type: str
    complexity: str
    languages: List[str]
    domain: str
    requirements: List[str]
    urgency: str

class AgentResponse(BaseModel):
    name: str
    description: str
    strengths: List[str]
    weaknesses: List[str]
    best_for: List[str]
    languages: List[str]
    platforms: List[str]
    pricing: str
    features: List[str]
    integration: List[str]
    learning_curve: str
    performance_score: float

class RecommendationResponse(BaseModel):
    agent: AgentResponse
    score: float
    reasoning: str
    fit_percentage: int

class RecommendationsResponse(BaseModel):
    task_analysis: TaskAnalysisResponse
    recommendations: List[RecommendationResponse]
    timestamp: str

# Initialize global instances
kb = AgentKnowledgeBase()
analyzer = TaskAnalyzer()
recommender = RecommendationEngine(kb)

app = FastAPI(
    title="AI Coding Agent Recommender",
    description="Find the perfect AI coding assistant for your development needs",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recommend", response_model=RecommendationsResponse)
async def recommend_agents(task_request: TaskRequest):
    try:
        task_description = task_request.task_description.strip()
        
        if not task_description:
            raise HTTPException(status_code=400, detail="Task description is required")
        
        # Analyze task and get recommendations
        task_analysis = analyzer.analyze_task(task_description)
        recommendations = recommender.recommend_agents(task_analysis)
        
        # Convert to response models
        task_analysis_response = TaskAnalysisResponse(**asdict(task_analysis))
        
        recommendation_responses = []
        for rec in recommendations:
            agent_response = AgentResponse(**asdict(rec.agent))
            recommendation_response = RecommendationResponse(
                agent=agent_response,
                score=rec.score,
                reasoning=rec.reasoning,
                fit_percentage=rec.fit_percentage
            )
            recommendation_responses.append(recommendation_response)
        
        return RecommendationsResponse(
            task_analysis=task_analysis_response,
            recommendations=recommendation_responses,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents", response_model=List[AgentResponse])
async def get_all_agents():
    agents = kb.get_all_agents()
    return [AgentResponse(**asdict(agent)) for agent in agents]

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
