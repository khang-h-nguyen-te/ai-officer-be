# Silk Lounge FAQ Chatbot

A sophisticated chatbot for Silk Lounge that processes FAQ documents, generates embeddings using OpenAI, and stores them in a Supabase vector database for semantic search to provide accurate answers about Silk Lounge services and amenities.

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

1. Navigate to the project directory
   ```bash
   cd silk_be
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
CREATE TABLE silklounge (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  content TEXT,
  metadata JSONB,
  embedding VECTOR(1536)  -- Adjust dimension based on your embedding model
);

-- Create a function for similarity search
CREATE OR REPLACE FUNCTION semantic_search_silklounge(
  query_embedding VECTOR(1536),
  match_threshold FLOAT DEFAULT 0.5,
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  text TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    silklounge.id,
    silklounge.content as text,
    silklounge.metadata,
    1 - (silklounge.embedding <=> query_embedding) AS similarity
  FROM silklounge
  WHERE 1 - (silklounge.embedding <=> query_embedding) > match_threshold
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
doc_ids = processor.process_and_store("path/to/your/file.pdf", table_name="silklounge")
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
query = "What are the operating hours of Silk Lounge?"
query_embedding = generate_embedding(query)
results = supabase_client.similarity_search(
    query_embedding, 
    limit=3,
    table_name="silklounge"
)

# Process results
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result.get('content', '')[:100]}...")
    print(f"Similarity: {result.get('similarity', 0)}")
```

## Project Structure

```
silk_be/
├── app/
│   ├── agent/
│   │   └── agent_silklounge.py  # Main Silk Lounge agent
│   ├── config/
│   │   ├── env_config.py        # Environment configuration
│   │   └── supabase_config.py   # Supabase client configuration
│   ├── models/
│   │   └── request_models.py    # API request/response models
│   ├── services/
│   │   └── embeddings.py        # Embedding service
│   ├── templates/
│   │   └── prompt_templates.py  # System prompts for Silk Lounge
│   ├── tools/
│   │   └── search/
│   │       └── silklounge_semantic_search_tool.py # Semantic search tool
│   ├── utils/
│   │   ├── pdf_to_vectorstore.py # PDF processing utility
│   │   └── process_faq.py       # Main processing script
│   └── vectorstore/
│       └── supabase_vectorstore.py # Supabase vector store client
├── main.py                      # FastAPI application
├── requirements.txt             # Python dependencies
├── vercel.json                  # Vercel deployment config
└── README.md                    # This file
```


## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 