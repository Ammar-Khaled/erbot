import os
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from pinecone import Pinecone
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress Azure SDK logging
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.ai.inference').setLevel(logging.WARNING)
logging.getLogger('azure').setLevel(logging.WARNING)

@dataclass
class RAGConfig:
    """Configuration class for RAG chatbot"""
    # Pinecone settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "erbot-index")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    
    # GitHub Models via Azure AI Inference settings
    github_token: str | None = os.getenv("GITHUB_TOKEN", None)
    azure_ai_endpoint: str = "https://models.github.ai/inference"
    model: str = os.getenv("MODEL_NAME", "openai/gpt-4.1")

    # RAG settings
    top_k: int = int(os.getenv("TOP_K", 3))  # Number of top documents to retrieve
    similarity_threshold: float =float(os.getenv("SIMILARITY_THRESHOLD", 0.7))  # Similarity threshold for relevance
    max_tokens: int = int(os.getenv("MAX_TOKENS", 1000))  # Max tokens for response generation
    embedding_model: str = "BAAI/bge-small-en-v1.5"

class PineconeRetriever:
    """Handles document retrieval from Pinecone vector store"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.pc = None
        self.index = None
        self.embeddings = None
        self._initialize_pinecone()
        self._initialize_embeddings()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            self.pc = Pinecone(api_key=self.config.pinecone_api_key)
            self.index = self.pc.Index(self.config.pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {self.config.pinecone_index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model for query encoding"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"Initialized embedding model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from Pinecone based on query
        
        Args:
            query: User's question/query
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            if self.embeddings is None:
                logger.error("Embeddings are not initialized")
                return []
            
            if self.index is None:
                logger.error("Pinecone index is not initialized")
                return []
                
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone index
            results = self.index.query(
                vector=query_embedding,
                top_k=self.config.top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Filter results by similarity threshold
            relevant_docs = []
            for match in results.matches:
                if match.score >= self.config.similarity_threshold:
                    relevant_docs.append({
                        'content': match.metadata.get('text', ''),
                        'source': match.metadata.get('source', 'Unknown'),
                        'score': match.score,
                        'id': match.id
                    })
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

class LLM:
    """Handles interaction with AI models via Azure AI Inference"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure AI Inference client following the sample pattern"""
        try:
            if self.config.github_token is None:
                raise ValueError("GitHub token cannot be None")
                
            # Create client following the provided sample
            client = ChatCompletionsClient(
                endpoint=self.config.azure_ai_endpoint,
                credential=AzureKeyCredential(self.config.github_token)
            )
            logger.info("Initialized Azure AI Inference client")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI client: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, messages: List) -> str:
        """
        Generate response using Azure AI Inference following the sample pattern
        
        Args:
            messages: List of SystemMessage and UserMessage objects
            
        Returns:
            Generated response text
        """
        try:
            # Make the request following the sample pattern with required parameters
            response = self.client.complete(
                messages=messages,
                temperature=0.7,
                top_p=1.0,
                max_tokens=self.config.max_tokens,
                model=self.config.model
            )
            
            # Extract response content following the sample
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    raise ValueError("Empty response content from model")
            else:
                raise ValueError("No response choices returned from model")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

class RAGChatbot:
    """Main RAG chatbot class that orchestrates retrieval and generation"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = PineconeRetriever(config)
        self.llm = LLM(config)
        
        # System prompt template
        self.system_prompt = """You are a friendly and helpful assistant. Answer questions based only on the provided context.

Guidelines:
- Use only the information from the context to answer questions
- If the context doesn't contain relevant information, respond with "I don't have enough context to answer this question accurately."
- Be concise but comprehensive in your responses
- If you're uncertain about something, acknowledge the uncertainty
- Maintain a helpful and professional tone"""

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Source {i} (Score: {doc['score']:.3f}):\n{doc['content']}")
        
        return "\n\n".join(context_parts)
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        """Create formatted user prompt with question and context"""
        return f"""Question: {question}

Context:
{context}

Please provide a comprehensive answer based on the context above."""

    def chat(self, question: str) -> Dict[str, Any]:
        """
        Process a user question and return an answer with metadata
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Processing question: {question}")
            documents = self.retriever.retrieve_documents(question)
            
            # Step 2: Format context
            context = self._format_context(documents)
            
            # Step 3: Create messages following the sample pattern
            messages = [
                SystemMessage(content=self.system_prompt),
                UserMessage(content=self._create_user_prompt(question, context))
            ]
            
            # Step 4: Generate response
            answer = self.llm.generate_response(messages)
            
            # Step 5: Prepare response metadata
            processing_time = time.time() - start_time
            
            response = {
                'answer': answer,
                'sources': [
                    {
                        'source': doc['source'],
                        'score': doc['score']
                    } for doc in documents
                ],
                'context_found': len(documents) > 0,
                'processing_time': processing_time,
                'num_sources': len(documents)
            }
            
            logger.info(f"Response generated in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            processing_time = time.time() - start_time
            return {
                'answer': "I'm sorry, I encountered an error while processing your question. Please try again.",
                'sources': [],
                'context_found': False,
                'processing_time': processing_time,
                'num_sources': 0,
                'error': str(e)
            }

def create_rag_chatbot() -> RAGChatbot:
    """
    Factory function to create a RAG chatbot instance
    
    Returns:
        Configured RAGChatbot instance
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get environment variables
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "erbot-index")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    
    # Validate required environment variables
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable is required")
        
    # Load configuration with validated values
    config = RAGConfig(
        pinecone_api_key=pinecone_api_key,
        github_token=github_token,
        pinecone_index_name=pinecone_index_name,
        model=model_name
    )
    
    return RAGChatbot(config)

# Example usage
if __name__ == "__main__":
    try:
        # Create chatbot instance
        chatbot = create_rag_chatbot()
        
        # Example interaction
        question = "What is the Purchase workflow?"
        response = chatbot.chat(question)
        
        print(f"Question: {question}")
        print(f"Answer: {response['answer']}")
        print(f"Sources found: {response['num_sources']}")
        print(f"Processing time: {response['processing_time']:.2f} seconds")
        
        if response['sources']:
            print("\nSources:")
            for source in response['sources']:
                print(f"- {source['source']} (Score: {source['score']:.3f})")
                
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")