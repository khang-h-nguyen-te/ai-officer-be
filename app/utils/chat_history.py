import logging
from typing import Optional
from app.config.supabase_config import get_supabase_client

logger = logging.getLogger(__name__)

def store_chat_history(user_query: str, chatbot_reply: str) -> Optional[int]:
    """
    Store a chat interaction in the Supabase database.
    
    Args:
        user_query: The question asked by the user
        chatbot_reply: The response provided by the chatbot
        
    Returns:
        Optional[int]: The ID of the inserted record if successful, None if failed
    """
    try:
        supabase = get_supabase_client()
        
        # Insert the chat history record
        response = supabase.table('aio_chat_history').insert({
            'user_query': user_query,
            'chatbot_reply': chatbot_reply
        }).execute()
        
        # Check if insertion was successful
        if response and hasattr(response, 'data') and len(response.data) > 0:
            record_id = response.data[0].get('id')
            logger.info(f"Successfully stored chat history with ID: {record_id}")
            return record_id
        else:
            logger.error("Failed to store chat history - no data returned")
            return None
            
    except Exception as e:
        logger.error(f"Error storing chat history: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None