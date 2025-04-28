from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: Name of the tool.
            description: Description of what the tool does.
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> str:
        """
        Execute the tool's functionality.
        
        Returns:
            String result of the tool execution.
        """
        pass 