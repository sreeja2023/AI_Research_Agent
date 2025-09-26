"""
RAG Chatbot Application with Sentence-BERT, FAISS, and Groq
===========================================================

A modular chatbot system that implements Retrieval-Augmented Generation (RAG)
using Sentence-BERT for embeddings, FAISS for vector search, and Groq for LLM inference.
Enhanced with comprehensive conversation history management.

Installation: pip install sentence-transformers faiss-cpu groq langchain langchain-core numpy
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime

# Core dependencies
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import groq
import requests
from urllib.parse import urlparse
import time

# LangChain components
try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    from langchain.schema import Document
    from langchain.schema.retriever import BaseRetriever
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.schema import HumanMessage, AIMessage, BaseMessage
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchFallback:
    """Web search fallback using Serper API and basic web scraping."""
    
    def __init__(self, serper_api_key: str = None, firecrawl_api_key: str = None):
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.search_enabled = bool(self.serper_api_key)
        
        if self.search_enabled:
            print("🌐 Web search fallback enabled")
        else:
            print("⚠️  Web search fallback disabled - SERPER_API_KEY not found")
    
    def is_valid_url(self, url):
        """Check if URL is valid."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except:
            return False
    
    def search_serper(self, query: str, top_k: int = 3):
        """Search using Serper API."""
        if not self.search_enabled:
            return []
        
        try:
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'q': query,
                'num': top_k
            }
            
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result_json = response.json()
                urls = []
                
                if 'organic' in result_json:
                    for result in result_json['organic']:
                        if 'link' in result and self.is_valid_url(result['link']):
                            urls.append({
                                'url': result['link'],
                                'title': result.get('title', ''),
                                'snippet': result.get('snippet', '')
                            })
                            if len(urls) >= top_k:
                                break
                
                return urls
            else:
                print(f"Serper API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error in Serper search: {e}")
            return []
    
    def simple_scrape(self, url: str, max_chars: int = 1000):
        """Simple web scraping fallback."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Basic text extraction - remove HTML tags
                import re
                text = re.sub(r'<[^>]+>', '', response.text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:max_chars]
            return ""
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
    
    def search_and_get_context(self, query: str, max_results: int = 2):
        """Search web and get context for the question."""
        if not self.search_enabled:
            return "Web search is not available. Please check your SERPER_API_KEY configuration."
        
        print(f"🔍 Searching web for: {query}")
        
        # Get search results
        search_results = self.search_serper(query, top_k=max_results)
        
        if not search_results:
            return "No relevant web search results found for your question."
        
        context_parts = []
        context_parts.append(f"Based on web search for '{query}':\n")
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"\n{i}. **{result['title']}**")
            context_parts.append(f"   Source: {result['url']}")
            
            if result['snippet']:
                context_parts.append(f"   Summary: {result['snippet']}")
            
            # Try to get more content
            content = self.simple_scrape(result['url'], max_chars=500)
            if content:
                context_parts.append(f"   Content: {content}...")
            
            time.sleep(0.5)  # Be respectful to servers
        
        return "\n".join(context_parts)


class ConversationHistoryManager:
    """Enhanced conversation history manager with multiple storage options."""
    
    def __init__(self, history_file: str = "conversation_history.json", max_memory_length: int = 20):
        # Use absolute path to ensure consistent location
        if not os.path.isabs(history_file):
            self.history_file = os.path.join(os.getcwd(), history_file)
        else:
            self.history_file = history_file
        self.max_memory_length = max_memory_length
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.load_history()
    
    def add_exchange(self, question: str, answer: str, sources: List[Dict] = None):
        """Add a question-answer exchange to history."""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "question": question,
            "answer": answer,
            "sources": sources or []
        }
        self.conversation_history.append(exchange)
        
        # Keep only recent exchanges to prevent memory overflow
        if len(self.conversation_history) > self.max_memory_length:
            self.conversation_history = self.conversation_history[-self.max_memory_length:]
        
        self.save_history()
    
    def get_recent_context(self, num_exchanges: int = 5) -> str:
        """Get recent conversation context as formatted string."""
        recent_history = self.conversation_history[-num_exchanges:]
        context_parts = []
        
        for exchange in recent_history:
            context_parts.append(f"Human: {exchange['question']}")
            context_parts.append(f"Assistant: {exchange['answer']}")
        
        return "\n".join(context_parts)
    
    def get_chat_history_messages(self, num_exchanges: int = 10) -> List[BaseMessage]:
        """Get recent chat history as LangChain message objects."""
        recent_history = self.conversation_history[-num_exchanges:]
        messages = []
        
        for exchange in recent_history:
            messages.append(HumanMessage(content=exchange['question']))
            messages.append(AIMessage(content=exchange['answer']))
        
        return messages
    
    def save_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            print(f"💾 History saved to: {os.path.abspath(self.history_file)}")
        except Exception as e:
            print(f"❌ Error saving history: {e}")
            logger.error(f"Error saving history: {e}")
    
    def load_history(self):
        """Load conversation history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                print(f"✅ Loaded {len(self.conversation_history)} conversation exchanges from {os.path.abspath(self.history_file)}")
                logger.info(f"Loaded {len(self.conversation_history)} conversation exchanges")
            else:
                print(f"📝 No existing history file found at {os.path.abspath(self.history_file)}. Starting fresh.")
                self.conversation_history = []
        except Exception as e:
            print(f"❌ Error loading history: {e}")
            logger.error(f"Error loading history: {e}")
            self.conversation_history = []
    
    def clear_history(self):
        """Clear all conversation history."""
        self.conversation_history = []
        self.save_history()
    
    def get_history_summary(self, num_recent: int = 5) -> str:
        """Get a summary of recent conversation topics."""
        if not self.conversation_history:
            return "No conversation history available."
        
        recent_exchanges = self.conversation_history[-num_recent:]
        summary_parts = []
        
        for i, exchange in enumerate(recent_exchanges, 1):
            timestamp = exchange['timestamp'][:19].replace('T', ' ')  # Format timestamp
            question_preview = exchange['question'][:80] + "..." if len(exchange['question']) > 80 else exchange['question']
            summary_parts.append(f"{i}. [{timestamp}] {question_preview}")
        
        return "\n".join(summary_parts)

class GroqLLM(LLM):
    """Custom LangChain LLM wrapper for Groq API using Llama 4 Scout."""
    
    api_key: str
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    client: Optional[Any] = None
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", **kwargs):
        # Initialize fields before calling super()._init_
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            client=None,
            **kwargs
        )
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Groq client."""
        try:
            self.client = groq.Groq(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
        
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    @property 
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "api_key_set": bool(self.api_key)
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Groq API."""
        if self.client is None:
            return "Error: Groq client not initialized. Please check your API key."
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"Error: Unable to generate response. {str(e)}"

class DocumentLoader:
    """Handles loading and processing of different document types."""
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Load content from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_json_file(file_path: str) -> str:
        """Load and stringify JSON file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Convert JSON to readable text format
                if isinstance(data, dict):
                    return DocumentLoader._dict_to_text(data)
                elif isinstance(data, list):
                    return "\n\n".join([DocumentLoader._dict_to_text(item) if isinstance(item, dict) else str(item) for item in data])
                else:
                    return str(data)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_markdown_file(file_path: str) -> str:
        """Load content from a Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading Markdown file {file_path}: {e}")
            return ""
    
    @staticmethod
    def _dict_to_text(data: Dict) -> str:
        """Convert dictionary to readable text format."""
        text_parts = []
        for key, value in data.items():
            if isinstance(value, dict):
                text_parts.append(f"{key}:\n{DocumentLoader._dict_to_text(value)}")
            elif isinstance(value, list):
                text_parts.append(f"{key}: {', '.join(map(str, value))}")
            else:
                text_parts.append(f"{key}: {value}")
        return "\n".join(text_parts)
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load multiple documents of different types."""
        documents = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            content = ""
            file_extension = path.suffix.lower()
            
            if file_extension == '.txt':
                content = self.load_text_file(file_path)
            elif file_extension == '.json':
                content = self.load_json_file(file_path)
            elif file_extension in ['.md', '.markdown']:
                content = self.load_markdown_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                continue
            
            if content:
                # Create LangChain document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "file_type": file_extension,
                        "file_name": path.name
                    }
                )
                documents.append(doc)
        
        return documents

class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def build_index(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
        """Build FAISS index from documents."""
        logger.info(f"Building FAISS index from {len(documents)} documents...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        self.documents = chunked_docs
        logger.info(f"Created {len(chunked_docs)} document chunks")
        
        # Generate embeddings
        texts = [doc.page_content for doc in chunked_docs]
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.embeddings = embeddings
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                doc = self.documents[idx]
                results.append((doc, float(score)))
        
        return results
    
    def save_index(self, index_path: str, documents_path: str):
        """Save FAISS index and documents to disk."""
        if self.index is None:
            raise ValueError("No index to save")
            
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Index saved to {index_path}, documents to {documents_path}")
    
    def load_index(self, index_path: str, documents_path: str):
        """Load FAISS index and documents from disk."""
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        logger.info(f"Index loaded from {index_path}, documents from {documents_path}")

class CustomRetriever(BaseRetriever):
    """Custom retriever that wraps FAISSVectorStore."""
    
    vector_store: FAISSVectorStore
    k: int = 4
    
    def __init__(self, vector_store: FAISSVectorStore, k: int = 4, **kwargs):
        super().__init__(
            vector_store=vector_store,
            k=k,
            **kwargs
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""
        results = self.vector_store.similarity_search(query, k=self.k)
        return [doc for doc, score in results]

class RAGChatbot:
    """Main RAG Chatbot class with enhanced conversation history management and web search fallback."""
    
    def __init__(self, groq_api_key: str, embedding_model_name: str = "all-MiniLM-L6-v2", 
                 memory_type: str = "buffer", max_memory_length: int = 20,
                 enable_web_search: bool = True, serper_api_key: str = None):
        self.groq_api_key = groq_api_key
        self.embedding_model_name = embedding_model_name
        self.max_memory_length = max_memory_length
        self.enable_web_search = enable_web_search
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.vector_store = FAISSVectorStore(embedding_model_name)
        self.llm = GroqLLM(groq_api_key)
        
        # Initialize web search fallback
        self.web_search = WebSearchFallback(serper_api_key) if enable_web_search else None
        
        # Initialize conversation history manager
        self.history_manager = ConversationHistoryManager(max_memory_length=max_memory_length)
        
        # Initialize memory based on type
        self.memory_type = memory_type
        self.memory = self._create_memory(memory_type)
        
        # LangChain components
        self.retriever = None
        self.qa_chain = None
        
    def _create_memory(self, memory_type: str):
        """Create appropriate memory type."""
        if memory_type == "buffer":
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif memory_type == "buffer_window":
            return ConversationBufferWindowMemory(
                k=self.max_memory_length,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif memory_type == "summary_buffer":
            return ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=2000,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        else:
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
    def load_and_index_documents(self, file_paths: List[str], chunk_size: int = 1000):
        """Load documents and build search index."""
        logger.info("Loading documents...")
        documents = self.document_loader.load_documents(file_paths)
        
        if not documents:
            raise ValueError("No documents loaded successfully")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Build index
        self.vector_store.build_index(documents, chunk_size=chunk_size)
        
        # Create retriever
        self.retriever = CustomRetriever(self.vector_store, k=4)
        
        # Create QA chain
        self._create_qa_chain()
        
    def _create_qa_chain(self):
        """Create the conversational retrieval chain with enhanced prompting."""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={
                "prompt": self._get_qa_prompt()
            }
        )
    
    def _get_qa_prompt(self):
        """Get custom prompt template for QA chain."""
        template = """You are a helpful AI assistant that answers questions based on the provided context and conversation history. 
        Use the following pieces of retrieved context to answer the question. Consider the conversation history for better context understanding.
        If you don't know the answer based on the context, just say that you don't know.
        Be conversational and helpful while staying grounded in the provided information.

        Context: {context}

        Question: {question}
        
        Answer:"""
        
        try:
            from langchain.prompts import PromptTemplate
            return PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        except:
            return template
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources, history context, and web search fallback."""
        if self.qa_chain is None:
            raise ValueError("No documents indexed. Call load_and_index_documents() first.")
        
        try:
            # Enhance question with recent context if available
            enhanced_question = self._enhance_question_with_context(question)
            
            # First, try to get response from RAG system
            response = self.qa_chain({"question": enhanced_question})
            
            # Check if the answer indicates lack of knowledge
            answer_text = response["answer"].lower()
            no_knowledge_indicators = [
                "i don't know", "don't know", "i do not know", "no information",
                "not mentioned", "doesn't mention", "not available", "can't find",
                "unable to find", "no relevant", "not in the context", "context doesn't",
                "provided context doesn't", "not sufficient information"
            ]
            
            needs_web_search = any(indicator in answer_text for indicator in no_knowledge_indicators)
            
            # Format initial response
            result = {
                "answer": response["answer"],
                "sources": [],
                "original_question": question,
                "enhanced_question": enhanced_question if enhanced_question != question else None,
                "used_web_search": False,
                "web_search_context": None
            }
            
            # Add source information from RAG
            source_info = []
            for doc in response.get("source_documents", []):
                source_data = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                source_info.append(source_data)
                result["sources"].append(source_data)
            
            # If RAG couldn't answer and web search is enabled, try web search
            if needs_web_search and self.web_search and self.web_search.search_enabled:
                print("\n🌐 RAG couldn't find relevant information. Searching the web...")
                
                # Get web search context
                web_context = self.web_search.search_and_get_context(question, max_results=2)
                
                if web_context and "No relevant web search results found" not in web_context:
                    # Create a new prompt with web search context
                    web_search_prompt = f"""
                    Based on the following web search results, please answer the user's question: "{question}"
                    
                    Web Search Results:
                    {web_context}
                    
                    Please provide a comprehensive answer based on this information. Be sure to mention that this information comes from web search results.
                    """
                    
                    # Get response from LLM with web context
                    try:
                        web_response = self.llm._call(web_search_prompt)
                        result["answer"] = web_response
                        result["used_web_search"] = True
                        result["web_search_context"] = web_context
                        result["sources"] = []  # Clear RAG sources since we used web search
                        
                        # Add web search info as sources
                        result["sources"].append({
                            "content": "Information gathered from web search results",
                            "metadata": {"source": "web_search", "search_query": question}
                        })
                        
                        print("✅ Web search provided additional context!")
                        
                    except Exception as web_error:
                        print(f"❌ Error processing web search results: {web_error}")
                        # Keep the original RAG response
                
            # Add to conversation history
            final_answer = result["answer"]
            if result["used_web_search"]:
                final_answer = f"[Web Search] {final_answer}"
            
            self.history_manager.add_exchange(question, final_answer, source_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            error_response = {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": [],
                "original_question": question,
                "enhanced_question": None,
                "used_web_search": False,
                "web_search_context": None
            }
            # Still add to history even if there was an error
            self.history_manager.add_exchange(question, error_response["answer"])
            return error_response
    
    def _enhance_question_with_context(self, question: str) -> str:
        """Enhance question with relevant conversation context."""
        recent_context = self.history_manager.get_recent_context(num_exchanges=3)
        
        if recent_context:
            # Check if question might be referential (contains pronouns or context-dependent terms)
            referential_terms = ["it", "this", "that", "they", "them", "what about", "how about", "also"]
            is_referential = any(term in question.lower() for term in referential_terms)
            
            if is_referential or len(question.split()) < 5:
                enhanced = f"Given this recent conversation context:\n{recent_context}\n\nCurrent question: {question}"
                return enhanced
        
        return question
    
    def get_conversation_history(self, num_exchanges: int = 10) -> List[Dict]:
        """Get recent conversation history."""
        return self.history_manager.conversation_history[-num_exchanges:]
    
    def get_history_summary(self, num_recent: int = 5) -> str:
        """Get a formatted summary of recent conversations."""
        return self.history_manager.get_history_summary(num_recent)
    
    def clear_history(self):
        """Clear all conversation history."""
        self.history_manager.clear_history()
        self.memory.clear()
        logger.info("Conversation history cleared")
    
    def save_index(self, index_dir: str = "rag_index"):
        """Save the current index for later use."""
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, "faiss.index")
        docs_path = os.path.join(index_dir, "documents.pkl")
        self.vector_store.save_index(index_path, docs_path)
    
    def load_index(self, index_dir: str = "rag_index"):
        """Load a previously saved index."""
        index_path = os.path.join(index_dir, "faiss.index")
        docs_path = os.path.join(index_dir, "documents.pkl")
        self.vector_store.load_index(index_path, docs_path)
        self.retriever = CustomRetriever(self.vector_store, k=4)
        self._create_qa_chain()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "memory_type": self.memory_type,
            "total_exchanges": len(self.history_manager.conversation_history),
            "current_session": self.history_manager.session_id,
            "langchain_memory_size": len(self.memory.chat_memory.messages) if hasattr(self.memory, 'chat_memory') else 0
        }

def create_sample_documents():
    """Create sample documents for demonstration."""
    
    # Sample text document
    sample_text = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is a broad field of computer science that aims to create 
    systems capable of performing tasks that typically require human intelligence. These 
    tasks include visual perception, speech recognition, decision-making, and language translation.
    
    Machine Learning (ML) is a subset of AI that focuses on the development of algorithms 
    that can learn and improve from experience without being explicitly programmed. ML 
    algorithms build mathematical models based on training data to make predictions or 
    decisions without being specifically programmed to perform the task.
    
    Deep Learning is a subset of machine learning that uses artificial neural networks 
    with multiple layers (hence "deep") to model and understand complex patterns in data.
    """
    
    # Sample JSON document
    sample_json = {
        "company": "TechCorp AI",
        "products": [
            {
                "name": "SmartBot",
                "type": "Chatbot",
                "features": ["Natural Language Processing", "Multi-language Support", "24/7 Availability"],
                "description": "An advanced AI chatbot for customer service automation"
            },
            {
                "name": "VisionAI",
                "type": "Computer Vision",
                "features": ["Object Detection", "Facial Recognition", "Real-time Processing"],
                "description": "AI-powered image and video analysis platform"
            }
        ],
        "about": "TechCorp AI specializes in developing cutting-edge artificial intelligence solutions for businesses across various industries."
    }
    
    # Sample Markdown document
    sample_markdown = """
    # RAG Systems Guide
    
    ## What is RAG?
    
    Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate more accurate, contextual responses.
    
    ## Key Components
    
    ### 1. Document Retrieval
    - Vector embeddings for semantic search
    - Similarity matching algorithms
    - Efficient indexing systems like FAISS
    
    ### 2. Text Generation
    - Large Language Models (LLMs)
    - Context-aware response generation
    - Prompt engineering techniques
    
    ## Benefits
    
    - *Accuracy*: Responses grounded in factual information
    - *Scalability*: Can work with large document collections
    - *Flexibility*: Easy to update knowledge base
    - *Transparency*: Provides source attribution
    """
    
    # Write sample files
    os.makedirs("sample_docs", exist_ok=True)
    
    with open("sample_docs/ai_basics.txt", "w") as f:
        f.write(sample_text)
    
    with open("sample_docs/company_info.json", "w") as f:
        json.dump(sample_json, f, indent=2)
    
    with open("sample_docs/rag_guide.md", "w") as f:
        f.write(sample_markdown)
    
    logger.info("Sample documents created in 'sample_docs' directory")

def main():
    """Enhanced RAG Chatbot with comprehensive history management."""

    # Set up Groq API key
    GROQ_API_KEY = "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"  # <-- replace with your key

    if GROQ_API_KEY == "your-groq-api-key-here":
        print("Please set your Groq API key in the GROQ_API_KEY variable")
        return

    try:
        # Initialize chatbot with enhanced memory
        print("Initializing RAG Chatbot with conversation history...")
        chatbot = RAGChatbot(
            groq_api_key=GROQ_API_KEY, 
            memory_type="buffer_window",  # Options: "buffer", "buffer_window", "summary_buffer"
            max_memory_length=15
        )

        # Load and index documents
        document_files = ["scout_results.txt"]

        print("Loading and indexing documents...")
        chatbot.load_and_index_documents(document_files)

        # Save index for later use
        print("Saving index...")
        chatbot.save_index()

        # Interactive mode with enhanced features
        print("\n" + "=" * 60)
        print("ENHANCED RAG CHATBOT WITH CONVERSATION HISTORY")
        print("=" * 60)
        print("Commands:")
        print("- Type your question normally")
        print("- 'history' - Show recent conversation summary")
        print("- 'clear' - Clear conversation history")
        print("- 'stats' - Show memory statistics")
        print("- 'debug' - Show debug information")
        print("- 'quit' - Exit the chatbot")
        print("\n" + "=" * 60)

        while True:
            user_input = input("\n🤖 Your question: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            elif user_input.lower() == "history":
                print("\n📜 Recent Conversation History:")
                print(chatbot.get_history_summary(10))
                continue
            elif user_input.lower() == "clear":
                chatbot.clear_history()
                print("\n🗑️  Conversation history cleared!")
                continue
            elif user_input.lower() == "stats":
                stats = chatbot.get_memory_stats()
                print(f"\n📊 Memory Statistics:")
                print(f"   Memory Type: {stats['memory_type']}")
                print(f"   Total Exchanges: {stats['total_exchanges']}")
                print(f"   Session ID: {stats['current_session']}")
                print(f"   LangChain Memory Size: {stats['langchain_memory_size']}")
                print(f"   History File: {chatbot.history_manager.history_file}")
                print(f"   File Exists: {os.path.exists(chatbot.history_manager.history_file)}")
                continue
            elif user_input.lower() == "debug":
                print(f"\n🔧 Debug Information:")
                print(f"   History File Path: {os.path.abspath(chatbot.history_manager.history_file)}")
                print(f"   Current Working Directory: {os.getcwd()}")
                print(f"   History Manager Length: {len(chatbot.history_manager.conversation_history)}")
                if chatbot.history_manager.conversation_history:
                    print(f"   Last Exchange: {chatbot.history_manager.conversation_history[-1]['question'][:50]}...")
                continue

            if user_input:
                print("\n🔍 Processing your question...")
                response = chatbot.ask(user_input)
                
                print(f"\n💬 Answer: {response['answer']}")
                
                # Show enhanced question if different
                if response.get('enhanced_question'):
                    print(f"\n🔄 Context-Enhanced Question: {response['enhanced_question']}")

                if response["sources"]:
                    print(f"\n📚 Sources ({len(response['sources'])} found):")
                    for i, source in enumerate(response["sources"], 1):
                        print(f"   {i}. {source['metadata']['file_name']}: {source['content']}")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Your conversation history has been saved.")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
