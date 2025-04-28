from typing import List, Dict, Any
import openai
from app.config.env_config import config

class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self):
        """Initialize the embedding service with OpenAI API key."""
        openai.api_key = config.openai_api_key
        self.model = config.embedding_model
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to generate an embedding for.
            
        Returns:
            A list of floats representing the embedding.
            If an error occurs, returns an empty list.
        """
        try:
            # Use the OpenAI API to generate an embedding
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return an empty list if an error occurs
            return []
            
    def get_document_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: A list of document dictionaries with at least a 'text' field.
            
        Returns:
            A list of document dictionaries with embeddings added.
        """
        enriched_documents = []
        
        for doc in documents:
            try:
                # Generate embedding for the document's text
                text = doc.get('text', '')
                if not text:
                    print(f"Warning: Document has no text field: {doc}")
                    continue
                    
                embedding = self.get_embedding(text)
                
                if embedding:
                    # Add the embedding to the document
                    doc_with_embedding = doc.copy()
                    doc_with_embedding['embedding'] = embedding
                    enriched_documents.append(doc_with_embedding)
            except Exception as e:
                print(f"Error processing document for embedding: {e}")
                
        return enriched_documents 