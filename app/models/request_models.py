from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class QueryRequest(BaseModel):
    """Model for query requests."""
    query: str = Field(..., description="The user's question to the chatbot")
    

class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="The status of the API")
    version: str = Field(..., description="The version of the API")
    details: Optional[Dict[str, Any]] = None 