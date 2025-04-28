from typing import Dict, Any, List
from fastapi import HTTPException


def validate_params(params: Dict[str, Any], required_params: List[str]) -> bool:
    """
    Validate that all required parameters are present.
    
    Args:
        params: Dictionary of parameters.
        required_params: List of required parameter names.
        
    Returns:
        True if all required parameters are present, False otherwise.
    """
    return all(param in params and params[param] for param in required_params)


def create_response(data: Any, status_code: int = 200) -> Dict[str, Any]:
    """
    Create a standardized API response.
    
    Args:
        data: The data to include in the response.
        status_code: The HTTP status code.
        
    Returns:
        A dictionary with the standardized response structure.
    """
    return {
        "status": "success" if 200 <= status_code < 300 else "error",
        "statusCode": status_code,
        "data": data
    } 