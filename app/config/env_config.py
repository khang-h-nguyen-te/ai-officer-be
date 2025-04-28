"""Environment configuration module."""

import os
import logging
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file - only once at module import time
load_dotenv()

# Config dictionary
config = {
    # OpenAI Configuration
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    
    # LlamaParse Configuration
    "llama_parse_api_key": os.getenv("LLAMA_PARSE_API_KEY"),
    
    # Supabase Configuration
    "supabase_url": os.getenv("SUPABASE_URL"),
    "supabase_key": os.getenv("SUPABASE_KEY"),
    
    # Vector Store Configuration
    "vector_store_table": os.getenv("VECTOR_STORE_TABLE", "aiofficer"),
    
    # Logging Configuration
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
}

class Config:
    """
    Configuration class for the application.
    This centralized config ensures environment variables are loaded consistently
    across the entire application.
    """
    
    def __init__(self):
        # OpenAI API keys and models
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.llm_model = os.environ.get('LLM_MODEL', 'gpt-4o')
        self.embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
        
        # Supabase credentials - handle multiple possible env var names
        self.supabase_url = os.environ.get('SUPABASE_URL') or os.environ.get('VITE_PUBLIC_BASE_URL', '')
        self.supabase_key = os.environ.get('SUPABASE_KEY') or os.environ.get('VITE_VITE_APP_SUPABASE_ANON_KEY', '')
        
        # LlamaIndex/LlamaParse configuration
        self.llama_parse_api_key = os.environ.get('LLAMA_CLOUD_API_KEY', '')
        
        # Application settings
        self.debug = os.environ.get('DEBUG', '0') == '1'
        
        # Memory token limit
        try:
            self.memory_token_limit = int(os.environ.get('MEMORY_TOKEN_LIMIT', '10000'))
        except ValueError:
            self.memory_token_limit = 10000
        
        # Validate critical configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate critical configuration settings and log warnings for missing values."""
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set in environment variables")
            
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not properly configured in environment variables")


# Create a global config instance - to be imported by other modules
config = Config() 