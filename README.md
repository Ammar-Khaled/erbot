# RAG Chatbot with Pinecone and Different LLMs for an ERP

A production-ready Retrieval-Augmented Generation (RAG) chatbot that combines Pinecone vector database for document retrieval with Different LLM models for intelligent response generation about ERP capabilities. This project is designed to provide learning, training, and support for users interested in our ERP system, leveraging the power of AI to answer questions based on a knowledge base and technical documents.

## Features

- **Free Embeddings**: Uses open-source BAAI/bge-small-en-v1.5 model
- **Smart Retrieval**: Uses Pinecone for fast, scalable vector search
- **Multi-Format Support**: Handles PDF, TXT, MD, DOCX, CSV, JSON, HTML files
- **AI-Powered Generation**: Leverages Different LLM models via Azure AI Inference
- **Production Ready**: Robust error handling with retry logic
- **Security First**: Environment-based credential management
- **Rich Metadata**: Detailed source attribution and performance metrics

## Quick Start

### 1. Setup Environment

```powershell
# Clone or download the project
cd erbot

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```powershell
# Copy environment template
copy .env.template .env
```

Edit `.env` with your credentials:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
GITHUB_TOKEN=your_github_token_here
PINECONE_INDEX_NAME=erbot-index1-bge-small-en-v1.5
```

### 3. Prepare Your Documents

```
data/
├── document1.pdf
├── notes.txt
├── manual.docx
└── data.csv
```

### 4. Index Your Documents

```powershell
python pinecone_ingestion.py
```

### 5. Start Chatting

```powershell
# Interactive mode
python rag_chatbot_final.py

# Or use the original version
python rag_chatbot.py
```

## Project Structure

```
erbot/
├── rag_chatbot_final.py      # Main RAG chatbot (refactored)
├── rag_chatbot.py            # Original implementation
├── pinecone_ingestion.py     # Document indexing script
├── example_usage.py          # Usage examples
├── requirements.txt          # Dependencies
├── .env.template            # Configuration template
├── .env                     # Your credentials (create this)
├── README.md                # This file
└── data/                    # Your documents folder
```

## Configuration

### Core Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PINECONE_INDEX_NAME` | `erbot-index1-bge-small-en-v1.5` | Your Pinecone index name |
| `top_k` | 3 | Number of documents to retrieve |
| `similarity_threshold` | 0.7 | Minimum relevance score |
| `max_tokens` | 1000 | Maximum response length |
| `temperature` | 0.7 | Response creativity (0-1) |

### Embedding Model

- **Model**: `BAAI/bge-small-en-v1.5`
- **Dimensions**: 384
- **Language**: English
- **License**: MIT (free for commercial use)

## Usage Examples

### Basic Usage

```python
from rag_chatbot_final import create_chatbot

# Initialize chatbot
chatbot = create_chatbot()

# Ask a question
response = chatbot.chat("What is the main topic of the documents?")

print(f"Answer: {response['answer']}")
print(f"Sources: {response['num_sources']}")
print(f"Time: {response['processing_time']:.2f}s")
```

### Custom Configuration

```python
from rag_chatbot_final import RAGConfig, RAGChatbot

config = RAGConfig(
   pinecone_api_key="your_key",
   github_token="your_token",
   pinecone_index_name="custom_index",
   top_k=5,
   similarity_threshold=0.8,
   max_tokens=1500
)

chatbot = RAGChatbot(config)
```

## Architecture

### Components

1. **DocumentRetriever**: Handles Pinecone vector search
2. **ResponseGenerator**: Manages DeepSeek AI interaction
3. **RAGChatbot**: Orchestrates the complete pipeline

### Processing Flow

```
Question → Embedding → Vector Search → Context Assembly → AI Generation → Response
```

1. **Query Processing**: Convert question to 384-dimensional vector
2. **Document Retrieval**: Search Pinecone for relevant content
3. **Context Assembly**: Format retrieved documents
4. **Response Generation**: Use DeepSeek to generate answer
5. **Metadata Collection**: Gather sources and performance metrics

## Performance

### Metrics Tracked

- **Processing Time**: End-to-end response time
- **Source Count**: Number of relevant documents found
- **Relevance Scores**: Document similarity scores
- **Context Quality**: Whether relevant context was found

### Expected Performance

- **Query Response**: < 3 seconds typical
- **Retrieval Accuracy**: ~85% with good document coverage
- **Memory Usage**: ~2GB RAM for embedding model

## Troubleshooting

### Common Issues

1. **"Dimension mismatch" error**
   ```bash
   # Solution: Ensure your Pinecone index has 384 dimensions
   # Delete and recreate index if needed
   ```

2. **"No relevant documents found"**
   ```python
   # Lower similarity threshold
   config.similarity_threshold = 0.5
   
   # Increase retrieval count
   config.top_k = 10
   ```

3. **"Embeddings model not found"**
   ```bash
   # Install required packages
   pip install sentence-transformers torch
   ```

4. **"Pinecone index not found"**
   ```bash
   # Run ingestion script first
   python pinecone_ingestion.py
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Best Practices

- No hardcoded credentials
- Environment variable configuration
- Secure Azure AI authentication
- Input validation and sanitization
- Error handling without data leaks

## Scaling Considerations

### For Production

1. **Caching**: Implement Redis for frequently asked questions
2. **Load Balancing**: Use multiple DeepSeek instances
3. **Monitoring**: Add application performance monitoring
4. **Rate Limiting**: Implement API rate limiting
5. **Database**: Consider PostgreSQL for chat history

### Cost Optimization

- **Embeddings**: Free (local processing)
- **Pinecone**: Pay per query/storage
- **DeepSeek**: Pay per token via GitHub
- **Hosting**: Can run on modest hardware (4GB+ RAM)

## What's New in v2.0

### Improvements

- **Simplified Architecture**: Cleaner, more maintainable code
- **Better Error Handling**: Graceful failure recovery
- **Enhanced Logging**: Structured logging with timestamps
- **Security Hardening**: Removed hardcoded credentials
- **Performance Optimization**: Faster initialization
- **Documentation**: Comprehensive setup guide

### Breaking Changes

- Configuration class restructured
- Some method names changed for clarity
- Environment variable validation added

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please ensure you comply with the licenses of:
- BAAI/bge-small-en-v1.5 (MIT)
- LangChain (MIT)
- Your DeepSeek API terms

## Support

- **Issues**: Create GitHub issue for bugs
- **Questions**: Use GitHub discussions
- **Documentation**: Check this README first

---

**Happy chatting!**
