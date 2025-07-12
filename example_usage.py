"""
Example usage of the RAG chatbot with Pinecone and DeepSeek
"""
from dotenv import load_dotenv
from rag_chatbot import create_rag_chatbot

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to demonstrate RAG chatbot usage"""
    
    try:
        # Create the RAG chatbot
        print("Initializing RAG chatbot...")
        chatbot = create_rag_chatbot()
        print("✅ RAG chatbot initialized successfully!")
        
        # Interactive chat loop
        print("\n🤖 RAG Chatbot is ready! Type 'quit' to exit.\n")
        
        while True:
            # Get user input
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🔍 Searching knowledge base and generating response...")
            
            # Get response from chatbot
            response = chatbot.chat(question)
            
            # Display the response
            print(f"\n🤖 Bot: {response['answer']}")
            
            # Show metadata if sources were found
            if response['context_found']:
                print(f"\n📊 Found {response['num_sources']} relevant sources")
                print(f"⏱️  Processing time: {response['processing_time']:.2f} seconds")
                
                if response['sources']:
                    print("\n📚 Sources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"   {i}. {source['source']} (Relevance: {source['score']:.3f})")
            else:
                print("\n❌ No relevant context found in knowledge base")
                
            print("\n" + "="*50 + "\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. PINECONE_API_KEY is set in your environment or .env file")
        print("2. GITHUB_TOKEN is set in your environment or .env file")
        print("3. Your Pinecone index 'erbot-index' exists and has data")


if __name__ == "__main__":
    main()

