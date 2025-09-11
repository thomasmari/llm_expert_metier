# -*- coding: utf-8 -*-
"""
FastAPI server for AI Expert Lawyer
Author: Claude.ai
Debugger: Jérome Delaunay et Thomas Mari 
Date: September 11, 2025
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from starlette.responses import FileResponse

# Import your AIExpertLawyer class
from aiexpertlawyer import AIExpertLawyer

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.8
    nb_chunk: Optional[int] = 4

class QuestionResponse(BaseModel):
    question: str
    answer: str
    status: str

class ConfigRequest(BaseModel):
    system_prompt: Optional[str] = None
    chroma_collection_name: str = "code_penal"
    chroma_db_path: str = "./chroma_langchain_db"
    llm_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.3
    top_p: float = 0.8
    nb_chunk: int = 4

# Initialize FastAPI app
app = FastAPI(
    title="AI Expert Lawyer API",
    description="API for interacting with an AI legal expert specialized in criminal law",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the AI expert instance
ai_expert: Optional[AIExpertLawyer] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI Expert Lawyer on startup"""
    global ai_expert
    try:
        ai_expert = AIExpertLawyer(
            temperature=0.3,
            nb_chunk=4,
            top_p=0.8
        )
        print("AI Expert Lawyer initialized successfully!")
    except Exception as e:
        print(f"Error initializing AI Expert Lawyer: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the web interface"""
    html_content = FileResponse("./interface/index.html")
    return html_content

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the AI expert"""
    global ai_expert
    
    if ai_expert is None:
        raise HTTPException(status_code=500, detail="AI Expert not initialized")
    
    try:
        # Update AI expert parameters if they differ from current settings
        if (request.temperature != 0.3 or 
            request.top_p != 0.8 or 
            request.nb_chunk != 4):
            
            ai_expert = AIExpertLawyer(
                temperature=request.temperature,
                top_p=request.top_p,
                nb_chunk=request.nb_chunk
            )
        
        # Get the answer
        answer = ai_expert.ask(request.question)
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/configure")
async def configure_expert(request: ConfigRequest):
    """Configure the AI expert with new settings"""
    global ai_expert
    
    try:
        ai_expert = AIExpertLawyer(
            system_prompt=request.system_prompt,
            chroma_collection_name=request.chroma_collection_name,
            chroma_db_path=request.chroma_db_path,
            llm_model=request.llm_model,
            temperature=request.temperature,
            top_p=request.top_p,
            nb_chunk=request.nb_chunk
        )
        
        return {"status": "success", "message": "AI Expert reconfigured successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring AI expert: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_expert_initialized": ai_expert is not None
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "interface:app",  # Change this to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
