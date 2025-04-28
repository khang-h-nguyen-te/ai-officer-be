import logging
from typing import List, Dict, Any
import os
from openai import OpenAI
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFToVectorStore:
    """Utility to convert PDF content to embeddings and store in Supabase."""
    
    def __init__(self, openai_api_key: str = None, table_name: str = "faq_embeddings"):
        """
        Initialize the utility.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            table_name: Name of the Supabase table to store embeddings
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.vectorstore = SupabaseVectorStore()
        self.table_name = table_name
        
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text using OpenAI.
        
        Args:
            text: The text to generate embedding for
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def process_pdf_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process PDF pages and convert to documents with embeddings.
        
        Args:
            pages: List of page objects from PDF parser
            
        Returns:
            List of documents with embeddings
        """
        documents = []
        
        for page in pages:
            try:
                # Extract page number and content
                page_num = page.page
                text_content = page.text
                
                # Skip empty pages
                if not text_content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "page_number": page_num,
                    "source": "FAQ"
                }
                
                # Generate embedding
                embedding = self.generate_embedding(text_content)
                
                # Create document
                document = {
                    "text": text_content,
                    "metadata": metadata,
                    "embedding": embedding
                }
                
                documents.append(document)
                logger.info(f"Processed page {page_num}")
                
            except Exception as e:
                logger.error(f"Error processing page {page.page}: {e}")
        
        return documents
    
    def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Store documents in Supabase vector store.
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            List of document IDs
        """
        return self.vectorstore.upsert_documents(documents, table_name=self.table_name)
    
    def process_and_store(self, pages: List[Dict[str, Any]]) -> List[str]:
        """
        Process pages and store in vector store.
        
        Args:
            pages: List of page objects from PDF parser
            
        Returns:
            List of document IDs
        """
        documents = self.process_pdf_pages(pages)
        return self.store_documents(documents) 