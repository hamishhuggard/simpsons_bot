#!/usr/bin/env python3
"""
RAG Chatbot for The Simpsons episodes using FastAPI, LangChain, and ChromaDB.
Supports both local TinyLlama and API-based models.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings

# LangChain imports
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Local LLM support
from ctransformers import AutoModelForCausalLM
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMOutput

# API LLM support
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class ChatMessage(BaseModel):
    """Model for chat messages"""
    message: str = Field(..., description="User's message")
    model_type: str = Field(default="local", description="Model type: 'local' or 'api'")
    api_key: Optional[str] = Field(default=None, description="API key for external models")
    model_name: Optional[str] = Field(default=None, description="Model name for API calls")


class ChatResponse(BaseModel):
    """Model for chat responses"""
    response: str
    relevant_episode: Optional[Dict[str, Any]] = None
    model_used: str
    timestamp: str


class SimpsonsRAGChatbot:
    """Main RAG chatbot class"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Simpsons RAG Chatbot",
            description="A RAG chatbot that can answer questions about The Simpsons episodes",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.episodes_data = []
        
        # Setup templates
        self.templates = Jinja2Templates(directory="templates")
        
        # Setup routes
        self.setup_routes()
        
        # Load episodes data
        self.load_episodes_data()
        
        # Initialize vector store
        self.initialize_vectorstore()
    
    def load_episodes_data(self):
        """Load episodes data from CSV or JSON file"""
        data_dir = Path("data")
        
        # Try to load from JSON first
        json_file = data_dir / "simpsons_episodes.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                self.episodes_data = json.load(f)
            print(f"Loaded {len(self.episodes_data)} episodes from JSON")
            return
        
        # Try to load from CSV
        csv_file = data_dir / "simpsons_episodes.csv"
        if csv_file.exists():
            self.episodes_data = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.episodes_data.append(row)
            print(f"Loaded {len(self.episodes_data)} episodes from CSV")
            return
        
        print("No episodes data found. Please run the scraper first.")
        self.episodes_data = []
    
    def initialize_vectorstore(self):
        """Initialize the vector store with episode data"""
        if not self.episodes_data:
            print("No episodes data available for vector store initialization")
            return
        
        # Create documents from episodes
        documents = []
        for episode in self.episodes_data:
            # Create a comprehensive text representation of the episode
            episode_text = f"""
            Season {episode.get('season', 'N/A')}, Episode {episode.get('episode_number_in_season', 'N/A')}: {episode.get('episode_title', 'N/A')}
            
            Air Date: {episode.get('air_date', 'N/A')}
            Description: {episode.get('description', 'N/A')}
            IMDb Rating: {episode.get('imdb_rating', 'N/A')}
            Vote Count: {episode.get('vote_count', 'N/A')}
            Episode URL: {episode.get('episode_url', 'N/A')}
            """
            
            # Create metadata
            metadata = {
                'season': episode.get('season', 'N/A'),
                'episode_number': episode.get('episode_number_in_season', 'N/A'),
                'title': episode.get('episode_title', 'N/A'),
                'air_date': episode.get('air_date', 'N/A'),
                'imdb_rating': episode.get('imdb_rating', 'N/A'),
                'url': episode.get('episode_url', 'N/A')
            }
            
            documents.append(Document(page_content=episode_text, metadata=metadata))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Initialize embeddings
        try:
            # Try to use Google Gemini embeddings
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=google_api_key
                )
                print("Using Google Gemini embeddings")
            else:
                # Fallback to sentence transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode
                print("Using sentence-transformers embeddings")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            return
        
        # Create vector store
        try:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            print(f"Vector store initialized with {len(split_docs)} document chunks")
        except Exception as e:
            print(f"Error creating vector store: {e}")
    
    def get_llm(self, model_type: str = "local", api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Get LLM based on model type"""
        if model_type == "local":
            if self.llm is None or not isinstance(self.llm, LocalTinyLlama):
                try:
                    # Download and initialize TinyLlama
                    model_path = "models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
                    if not os.path.exists(model_path):
                        print("Downloading TinyLlama model...")
                        self.download_tinyllama()
                    
                    self.llm = LocalTinyLlama(
                        model_path=model_path,
                        model_type="tinyllama",
                        max_new_tokens=512,
                        temperature=0.7
                    )
                    print("Local TinyLlama model loaded")
                except Exception as e:
                    print(f"Error loading local model: {e}")
                    raise HTTPException(status_code=500, detail="Failed to load local model")
        else:
            # API-based model
            if api_key is None:
                raise HTTPException(status_code=400, detail="API key required for external models")
            
            if model_name and "gpt" in model_name.lower():
                # OpenAI model
                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model_name=model_name or "gpt-3.5-turbo",
                    temperature=0.7
                )
            else:
                # Google Gemini model
                self.llm = ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=model_name or "gemini-pro",
                    temperature=0.7
                )
        
        return self.llm
    
    def download_tinyllama(self):
        """Download TinyLlama model"""
        import requests
        
        model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
        model_path = "models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        print(f"Downloading TinyLlama from {model_url}")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded to {model_path}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "episodes_loaded": len(self.episodes_data),
                "vectorstore_ready": self.vectorstore is not None,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/episodes")
        async def get_episodes(season: Optional[int] = None, limit: int = 10):
            """Get episodes with optional filtering"""
            episodes = self.episodes_data
            
            if season:
                episodes = [ep for ep in episodes if ep.get('season') == str(season)]
            
            return {
                "episodes": episodes[:limit],
                "total": len(episodes),
                "returned": min(limit, len(episodes))
            }
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(message: ChatMessage):
            """Chat endpoint"""
            try:
                # Get LLM
                llm = self.get_llm(
                    model_type=message.model_type,
                    api_key=message.api_key,
                    model_name=message.model_name
                )
                
                if self.vectorstore is None:
                    raise HTTPException(status_code=500, detail="Vector store not initialized")
                
                # Create conversation chain
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=memory,
                    return_source_documents=True
                )
                
                # Get response
                result = qa_chain({"question": message.message})
                
                # Extract relevant episode information
                relevant_episode = None
                if result.get("source_documents"):
                    # Get the most relevant episode from source documents
                    source_doc = result["source_documents"][0]
                    relevant_episode = source_doc.metadata
                
                return ChatResponse(
                    response=result["answer"],
                    relevant_episode=relevant_episode,
                    model_used=f"{message.model_type}_{message.model_name or 'tinyllama'}",
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


class LocalTinyLlama(LLM):
    """Custom LLM class for TinyLlama"""
    
    model_path: str
    model_type: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            model_type=self.model_type,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
    
    @property
    def _llm_type(self) -> str:
        return "tinyllama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMOutput] = None) -> str:
        # Format prompt for TinyLlama
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response = self.model(formatted_prompt)
        return response.strip()


def main():
    """Main function to run the FastAPI server"""
    chatbot = SimpsonsRAGChatbot()
    
    print("Starting Simpsons RAG Chatbot...")
    print("API will be available at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    
    uvicorn.run(
        chatbot.app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main() 