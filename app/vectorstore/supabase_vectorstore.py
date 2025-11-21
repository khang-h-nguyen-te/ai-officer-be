import logging
from typing import Dict, List, Any, Optional

from app.config.supabase_config import get_supabase_client
from app.config.env_config import config

logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    """Interface to Supabase vector store for FAQ embeddings."""
    
    def __init__(self, auth: str = None):
        """
        Initialize the vector store.
        
        Args:
            auth: Optional auth token for authenticated requests.
        """
        self.auth = auth
        self.client = get_supabase_client(self.auth)
        
    def create_table_if_not_exists(self, table_name: str = "silklounge_faq_embeddings"):
        """
        Create the embeddings table if it doesn't exist.
        This is a simplified version and should be adapted to your Supabase setup.
        
        Args:
            table_name: The name of the table to create.
        """
        # Note: In practice, you would typically create the table through migrations or the Supabase UI
        # This is a simplified example to demonstrate the concept
        logger.info(f"Table creation would typically be handled through migrations: {table_name}")
        logger.info("Please ensure your Supabase database has the appropriate tables set up.")
        
    def upsert_documents(self, documents: List[Dict[str, Any]], table_name: str = "silklounge_faq_embeddings") -> List[str]:
        """
        Insert or update documents with embeddings into the vector store.
        
        Args:
            documents: Documents with embeddings to store.
            table_name: The name of the table to store the documents in.
            
        Returns:
            List of document IDs.
        """
        document_ids = []
        
        try:
            for doc in documents:
                # Prepare the data for insertion
                data = {
                    "text": doc.get("text", ""),
                    # "metadata": doc.get("metadata", {}),
                    "embedding": doc.get("embedding", [])
                }
                
                # Insert the data into the database
                response = self.client.table(table_name).upsert(data).execute()
                
                # Check if insertion was successful
                if response.data:
                    document_ids.append(response.data[0].get("id"))
                    logger.info(f"Successfully stored document with ID: {response.data[0].get('id')}")
                else:
                    logger.error(f"Failed to store document: {doc.get('id', 'unknown')}")
                    
        except Exception as e:
            logger.error(f"Error storing documents in Supabase: {e}")
            
        return document_ids
        
    def similarity_search(self, query_embedding: List[float], limit: int = 5, table_name: str = "silklounge_faq_embeddings"):
        """
        Perform a similarity search using the query embedding.
        
        Args:
            query_embedding: The embedding for the query.
            limit: Maximum number of results to return.
            table_name: The name of the table to search in.
            
        Returns:
            List of matching documents.
        """
        try:
            # This assumes you have a function setup in your Supabase database
            # You'll need to create this RPC function in your Supabase instance
            response = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.5,
                    "match_count": limit
                }
            ).execute()
            
            return response.data
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return [] 