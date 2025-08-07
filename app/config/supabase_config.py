import logging
from supabase import create_client, Client
from app.config.env_config import config

logger = logging.getLogger(__name__)

def get_supabase_client(auth_header: str = None) -> Client:
    """
    Create a Supabase client using the centralized configuration.
    
    Args:
        auth_header: Optional auth header for authenticated requests.
        
    Returns:
        Supabase client instance.
    """
    url = config.supabase_url
    key = config.supabase_key
    
    if not url or not key:
        logger.warning("Supabase URL or key not set. Using empty values.")
    
    try:
        client = create_client(url, key)
        
        # Add auth header if provided
        if auth_header:
            # Remove 'Bearer ' prefix if present
            if auth_header.startswith("Bearer "):
                auth_token = auth_header[7:]
            else:
                auth_token = auth_header
            
            # Set auth header for the client session
            client.auth.set_session(auth_token)
        
        return client
    except Exception as e:
        logger.error(f"Error creating Supabase client: {e}")
        # For TypeError related to proxy argument in older versions
        try:
            from supabase.client import ClientOptions
            client = create_client(url, key, options=ClientOptions())
            return client
        except Exception as sub_e:
            logger.error(f"Failed to create Supabase client with alternative method: {sub_e}")
            raise