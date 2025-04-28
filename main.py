from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os

# Configure for serverless first, before any imports that might use tiktoken
from app.utils.serverless_utils import configure_for_serverless
configure_for_serverless()

# Import from our application structure
from app.agent.agent_aiofficer import AgentAIOfficer
from app.models.request_models import QueryRequest, HealthResponse
from app.utils.response_utils import create_response
from app.config.env_config import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger for the FastAPI app
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI-Officer Chatbot API",
    description="API for interacting with the AI-Officer Chatbot",
    version="1.0.0"
)
executor = ThreadPoolExecutor(max_workers=4)

# Initialize agent - this will start background initialization
ai_officer_agent = AgentAIOfficer()
logger.info("AI-Officer agent instance created - initialization started in background")

# A helper function to process the query synchronously
def process_query(query: str) -> str:
    """
    Process a user query using the agent.
    
    Args:
        query: The user's question
    
    Returns:
        The agent's response
    """
    # The agent_query method now handles the case when the agent is not initialized
    response = ai_officer_agent.agent_query(query)
    return response
    
# Define a POST endpoint to receive user queries
@app.post("/ask")
async def ask_query(payload: QueryRequest, request: Request):
    """
    Process a user question and return the agent's response.
    
    Args:
        payload: The query request containing the user question
        request: The FastAPI request object
    
    Returns:
        The agent's response
    """
    query = payload.query
    logger.debug(f"Processing query: {query}")
    
    # Offload the blocking agent call to a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, process_query, query)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to process query")
    
    return {"response": result}

# Improved health check endpoint with detailed agent status
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with agent initialization status."""
    init_status = ai_officer_agent.initialization_status()
    
    # Add more details about the agent initialization status
    details = {
        "agent": init_status["status"],
        "message": init_status["message"]
    }
    
    # Add elapsed time if initializing
    if init_status["status"] == "initializing" and "elapsed_seconds" in init_status:
        details["elapsed_seconds"] = init_status["elapsed_seconds"]
    
    return HealthResponse(status="ok", version="1.0.0", details=details)

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
# if __name__ == "__main__":
#     # Run the FastAPI app using uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port) 