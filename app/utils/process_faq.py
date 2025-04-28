import os
import sys
import logging
from dotenv import load_dotenv
import nest_asyncio
from llama_cloud_services import LlamaParse
from openai import OpenAI
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore

# Add parent directory to path to ensure imports work regardless of where script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import from app package

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio for async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()

class FAQProcessor:
    """Process FAQ PDF and store in Supabase vector store."""
    
    def __init__(self):
        # Get API keys from environment
        self.llama_parse_api_key = os.environ.get("LLAMA_PARSE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.llama_parse_api_key:
            raise ValueError("LLAMA_PARSE_API_KEY environment variable is required")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.supabase_client = SupabaseVectorStore()
        
    def parse_pdf(self, pdf_path):
        """Parse PDF using LlamaParse."""
        logger.info(f"Parsing PDF: {pdf_path}")
        
        parser = LlamaParse(
            api_key=self.llama_parse_api_key,
            verbose=True,
            language="en",
            markdown_split_by_page=True,
        )
        
        return parser.parse(pdf_path)
    
    def generate_embedding(self, text):
        """Generate embedding using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def process_page(self, page):
        """Process a single page and return document with embedding."""
        # Extract page number and content
        page_num = page.page
        text_content = page.text
        
        # Skip empty pages
        if not text_content.strip():
            logger.info(f"Skipping empty page {page_num}")
            return None
        
        # Create metadata
        metadata = {
            "page_number": page_num,
            "source": "FAQ",
            "title": f"FAQ Page {page_num}"
        }
        
        # Generate embedding
        logger.info(f"Generating embedding for page {page_num}")
        embedding = self.generate_embedding(text_content)
        
        # Create document with text, metadata, and embedding. For metadata, flatten the metadata dictionary and append directly to the document.
        document = {}
        document["text"] = text_content
        document["embedding"] = embedding


        
        return document
    
    def store_in_supabase(self, documents, table_name="aiofficer"):
        """Store documents in Supabase."""
        logger.info(f"Storing {len(documents)} documents in Supabase table: {table_name}")
        return self.supabase_client.upsert_documents(documents, table_name=table_name)
    
    def process_and_store(self, pdf_path, table_name="aiofficer"):
        """Process PDF and store in Supabase."""
        # Parse PDF
        result = self.parse_pdf(pdf_path)
        pages = result.pages
        logger.info(f"Parsed {len(pages)} pages from PDF")
        
        # Process pages
        documents = []
        for page in pages:
            try:
                document = self.process_page(page)
                if document:
                    documents.append(document)
            except Exception as e:
                logger.error(f"Error processing page {page.page}: {e}")
        
        logger.info(f"Processed {len(documents)} documents with embeddings")
        
        # Store in Supabase
        doc_ids = self.store_in_supabase(documents, table_name)
        logger.info(f"Stored {len(doc_ids)} documents in Supabase")
        
        return doc_ids

def main():
    """Main function to run the FAQ processor."""
    # Path to FAQ PDF file
    pdf_path = "./FAQ.pdf"
    
    try:
        processor = FAQProcessor()
        doc_ids = processor.process_and_store(pdf_path)
        
        print(f"Successfully stored {len(doc_ids)} documents in Supabase")
        print(f"Document IDs: {doc_ids}")
        
    except Exception as e:
        logger.error(f"Error processing FAQ: {e}")
        raise

if __name__ == "__main__":
    main() 