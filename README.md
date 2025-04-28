# AI Officer PDF Embedding and Vector Storage

A toolkit for processing PDF documents, generating embeddings using OpenAI, and storing them in a Supabase vector database for semantic search.

## Features

- Parse PDF documents using LlamaParse
- Generate embeddings with OpenAI's text-embedding-3-small model
- Store documents and embeddings in Supabase for vector search
- Perform semantic searches on your document collection

## Setup

### Prerequisites

- Python 3.8+
- Supabase account with vector extension enabled
- LlamaParse API key
- OpenAI API key

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ai-officer.git
   cd ai-officer
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and Supabase credentials
   ```

### Database Setup

Create a table in Supabase with the following structure:

```sql
CREATE TABLE aiofficer (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  content TEXT,
  metadata JSONB,
  embedding VECTOR(1536)  -- Adjust dimension based on your embedding model
);

-- Create a function for similarity search
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(1536),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) AS similarity
  FROM aiofficer
  WHERE 1 - (embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
```

## Usage

### Processing a PDF

Run the PDF processing script:

```bash
# From the project root directory
python -m app.utils.process_faq
```

### Code Example

```python
from app.utils.process_faq import FAQProcessor

# Initialize the processor
processor = FAQProcessor()

# Process a PDF file and store in Supabase
doc_ids = processor.process_and_store("path/to/your/file.pdf", table_name="aiofficer")
```

### Querying the Vector Store

```python
from openai import OpenAI
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore

# Initialize clients
openai_client = OpenAI(api_key="your_openai_api_key")
supabase_client = SupabaseVectorStore()

# Generate query embedding
def generate_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Search for similar documents
query = "What is an AI Officer?"
query_embedding = generate_embedding(query)
results = supabase_client.similarity_search(
    query_embedding, 
    limit=3,
    table_name="aiofficer"
)

# Process results
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result.get('content', '')[:100]}...")
    print(f"Similarity: {result.get('similarity', 0)}")
```

## Project Structure

```
ai-officer/
├── app/
│   ├── config/
│   │   ├── env_config.py        # Environment configuration
│   │   └── supabase_config.py   # Supabase client configuration
│   ├── utils/
│   │   ├── pdf_to_vectorstore.py # PDF processing utility
│   │   └── process_faq.py       # Main processing script
│   └── vectorstore/
│       └── supabase_vectorstore.py # Supabase vector store client
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```


## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 