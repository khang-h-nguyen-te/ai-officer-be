import logging
from typing import List, Dict, Any
from app.tools.base_tool import BaseTool
from app.services.embeddings import EmbeddingService
from app.config.env_config import config
from app.config.supabase_config import get_supabase_client

logger = logging.getLogger(__name__)

class AIOfficerSemanticSearchTool(BaseTool):
    """Tool for semantically searching information using Supabase vector search."""
    
    def __init__(self):
        super().__init__(
            name="AIOfficerSemanticSearch",
            description="Search for information using semantic search. This tool helps retrieve relevant information to answer user queries with accurate and helpful responses."
        )
        self.embedding_service = EmbeddingService()
        self.supabase = get_supabase_client()
    
    def __call__(self, query: str, match_count: int = 3) -> str:
        """
        Perform semantic search for information.
        
        Args:
            query: The user's question
            match_count: Number of results to return (default: 5)
            
        Returns:
            Relevant information from the knowledge base
        """
        try:
            # Generate embedding for the query
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = self.embedding_service.get_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return "I'm sorry, I'm having trouble processing your question. Please try asking in a different way."
            
            # Call the Supabase RPC function for semantic search
            logger.info(f"Calling semantic_search_aiofficer with match_count={match_count}")
            response = self.supabase.rpc(
                'semantic_search_aiofficer',  # Note: You'll need to create this function in Supabase
                {
                    'query_embedding': query_embedding,
                    'match_count': match_count
                }
            ).execute()
            
            # Process and format the results
            if not hasattr(response, 'data') or not response.data:
                logger.warning("No results found from semantic search")
                return "I don't have specific information about that. Is there something else I can help you with?"
            
            results = response.data
            logger.info(f"Found {len(results)} matching documents")
            
            # Format the results into a readable response
            if len(results) == 1:
                # If only one result, return its text directly
                return results[0]['text']
            else:
                # If multiple results, combine them into a comprehensive answer
                combined_text = "\n\n".join([result['text'] for result in results])
                return combined_text
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "I apologize, but I'm having trouble accessing the information at the moment. Please try again later." 