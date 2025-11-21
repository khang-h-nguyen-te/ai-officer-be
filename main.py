from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time

# Configure for serverless first, before any imports that might use tiktoken
from app.utils.serverless_utils import configure_for_serverless, is_serverless_environment, get_serverless_info
configure_for_serverless()

# Import from our application structure
from app.agent.agent_silklounge import AgentSilkLounge
from app.models.request_models import QueryRequest, HealthResponse
from app.utils.response_utils import create_response
from app.config.env_config import config
from app.utils.chat_history import store_chat_history

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger for the FastAPI app
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Silk Lounge Chatbot API",
    description="API for interacting with the Silk Lounge FAQ Chatbot",
    version="1.0.0"
)
executor = ThreadPoolExecutor(max_workers=4)

# Record startup time
startup_time = time.time()

# Initialize agent - this will start background initialization
silk_lounge_agent = AgentSilkLounge()
logger.info("Silk Lounge agent instance created - initialization started in background")

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
    response = silk_lounge_agent.agent_query(query)
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
    try:
        query = payload.query
        logger.debug(f"Processing query: {query}")
        
        # Offload the blocking agent call to a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_query, query)
        
        if not result:
            logger.warning("Empty result returned from agent")
            if is_serverless_environment():
                # In serverless, provide a direct answer if agent returned empty result
                return {"response": "I'm here to help with information about Silk Lounge. How can I assist you today?"}
            else:
                raise HTTPException(status_code=500, detail="Failed to process query")
        
        # Store the chat history
        history_id = store_chat_history(query, result)
        if not history_id:
            logger.warning("Failed to store chat history")
        
        return {"response": result}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        
        # Provide a friendly response even on errors
        if is_serverless_environment():
            return {"response": "I'm here to help with information about Silk Lounge. How can I assist you today?"}
        else:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Improved health check endpoint with detailed agent status
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with agent initialization status."""
    init_status = silk_lounge_agent.initialization_status()
    
    # Add more details about the agent initialization status
    details = {
        "agent": init_status["status"],
        "message": init_status["message"],
        "uptime_seconds": time.time() - startup_time,
        "is_serverless": is_serverless_environment()
    }
    
    # Add elapsed time if initializing
    if init_status["status"] == "initializing" and "elapsed_seconds" in init_status:
        details["elapsed_seconds"] = init_status["elapsed_seconds"]
    
    # Add serverless info if applicable
    if is_serverless_environment():
        details["serverless_info"] = get_serverless_info()
    
    return HealthResponse(status="ok", version="1.0.0", details=details)

# Get serverless environment info
@app.get("/serverless-info")
async def serverless_info():
    """Get information about the serverless environment."""
    if is_serverless_environment():
        return {
            "is_serverless": True,
            "info": get_serverless_info(),
            "agent_status": silk_lounge_agent.initialization_status(),
            "uptime_seconds": time.time() - startup_time
        }
    else:
        return {"is_serverless": False}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info(f"Application starting up. Serverless environment: {is_serverless_environment()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Application shutting down")
    executor.shutdown(wait=False)

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
# if __name__ == "__main__":
#     # Run the FastAPI app using uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port) 