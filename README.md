# Simpsons RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions about The Simpsons episodes using FastAPI, LangChain, and ChromaDB. The system scrapes episode data from IMDb using Scrapy and provides both local TinyLlama and API-based model options.

## Features

- **Parallel Scraping**: Uses Scrapy to scrape Simpsons episodes from IMDb in parallel
- **Incremental Data Saving**: Saves scraped data incrementally to CSV and JSON files
- **RAG System**: Uses ChromaDB for vector storage and semantic search
- **Multiple LLM Options**: 
  - Local TinyLlama 1.1B model (CPU-based)
  - Google Gemini Pro API
  - OpenAI GPT models
- **Web Interface**: Simple HTML interface for chatting
- **FastAPI Backend**: RESTful API with automatic documentation

## Project Structure

```
movie-bot/
├── simpsons_scraper/          # Scrapy project
│   ├── simpsons_scraper/
│   │   ├── spiders/
│   │   │   └── episodes.py    # Main scraper spider
│   │   ├── pipelines.py       # Data processing pipeline
│   │   └── settings.py        # Scrapy settings
├── templates/
│   └── index.html             # Web interface
├── data/                      # Scraped data (created after scraping)
├── models/                    # Local models (created after download)
├── chroma_db/                 # Vector database (created after initialization)
├── scrape_episode_data.py     # Original sequential scraper
├── run_scraper.py            # Script to run Scrapy spider
├── rag_chatbot.py            # Main RAG chatbot application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd movie-bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional, for API models):
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

## Usage

### 1. Scrape Episode Data

Run the Scrapy spider to collect episode data from IMDb:

```bash
python run_scraper.py [max_seasons]
```

- `max_seasons`: Maximum number of seasons to scrape (default: 40)
- Data will be saved incrementally to `data/simpsons_episodes.csv` and `data/simpsons_episodes.json`

### 2. Start the RAG Chatbot

```bash
python rag_chatbot.py
```

The chatbot will:
- Load scraped episode data
- Initialize the vector store with embeddings
- Download TinyLlama model (if not already present)
- Start the FastAPI server

### 3. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
  "message": "What episode features Homer becoming a food critic?",
  "model_type": "local",
  "api_key": "optional_api_key",
  "model_name": "optional_model_name"
}
```

### Episodes Endpoint
```http
GET /episodes?season=1&limit=10
```

### Health Check
```http
GET /health
```

## Model Options

### Local TinyLlama
- **Model**: TinyLlama 1.1B Chat v0.3 (GGUF format)
- **Size**: ~668MB (Q4_K_M quantization)
- **Requirements**: CPU only, no internet required after download
- **Performance**: Good for basic Q&A, runs entirely locally

### API Models
- **Google Gemini Pro**: Requires Google API key
- **OpenAI GPT-3.5/4**: Requires OpenAI API key
- **Performance**: Better quality responses, requires internet

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: For Google Gemini embeddings and chat
- `OPENAI_API_KEY`: For OpenAI models

### Scrapy Settings
- Concurrent requests: 16
- Download delay: 0.5 seconds
- Auto-throttling enabled
- Retry on failures

### Vector Store
- Embedding model: Google Gemini (fallback to sentence-transformers)
- Chunk size: 1000 characters
- Overlap: 200 characters
- Retrieval: Top 3 most relevant episodes

## Example Queries

- "What episode features Homer becoming a food critic?"
- "Tell me about episodes with Mr. Burns"
- "What are the highest-rated episodes?"
- "Which episodes aired in 1990?"
- "What happens in the Treehouse of Horror episodes?"

## Development

### Adding New Features
1. Modify the spider in `simpsons_scraper/spiders/episodes.py` for different data
2. Update the RAG system in `rag_chatbot.py` for new capabilities
3. Enhance the web interface in `templates/index.html`

### Customizing Models
- Change model parameters in `LocalTinyLlama` class
- Add new API providers in `get_llm()` method
- Modify embedding settings in `initialize_vectorstore()`

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **No Episode Data**: Run the scraper first
   ```bash
   python run_scraper.py
   ```

3. **Model Download Issues**: Check internet connection and disk space

4. **API Key Errors**: Set environment variables or provide keys in web interface

5. **Memory Issues**: Reduce concurrent requests in Scrapy settings

### Performance Tips

- Use local TinyLlama for privacy and offline use
- Use API models for better response quality
- Adjust chunk size and overlap for different use cases
- Monitor memory usage with large datasets

## License

This project is for educational purposes. Please respect IMDb's terms of service when scraping data.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- [TheBloke](https://huggingface.co/TheBloke) for the TinyLlama GGUF model
- [LangChain](https://langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework 