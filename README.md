# 📄 RAG Chatbot (Ollama + LangChain)

A powerful **Retrieval-Augmented Generation (RAG)** chatbot application built with Streamlit that allows you to upload documents and ask intelligent questions about their content using locally-run language models.

## ✨ Features

- 📤 **Multi-format Support**: Upload PDF and TXT files
- 🔍 **Smart Retrieval**: Semantic search using embeddings to find relevant document chunks
- 🤖 **Local LLM**: Runs completely offline using Ollama (no API keys required)
- 💬 **Interactive Chat**: Streamlit-powered conversational interface
- 🔗 **RAG Architecture**: Combines retrieval and generation for accurate, context-aware answers
- 💾 **Session Memory**: Maintains conversation history during the session

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[LangChain](https://www.langchain.com/)** - LLM orchestration and text processing
- **[Ollama](https://ollama.ai/)** - Local language models
  - `llama3.2:1b` - Fast, lightweight language model
  - `nomic-embed-text` - Text embeddings model
- **[PyPDF2](https://github.com/py-pdf/PyPDF2)** - PDF text extraction
- **[NumPy](https://numpy.org/)** - Vector operations

## 📋 Prerequisites

Before running the application, ensure you have:

1. **Python 3.8 or higher** installed
2. **[Ollama](https://ollama.ai/)** installed and running
3. The required Ollama models downloaded:
   ```bash
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ```

### Starting Ollama

Start the Ollama service before running the app:
```bash
ollama serve
```

The default Ollama API runs on `http://localhost:11434`

## 🚀 Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd rag-chatbot
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install PyPDF2
pip install numpy
pip install ollama
```

## 💻 Usage

### 1. Start Ollama Service
```bash
ollama serve
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Using the Chat Interface

1. **Upload Documents**:
   - Click "Upload Files" in the left sidebar
   - Select one or more PDF or TXT files
   - Wait for the "✅ Indexed X chunks" confirmation message

2. **Ask Questions**:
   - Type your question in the chat input box at the bottom
   - The chatbot will retrieve relevant document sections and generate an answer
   - Continue the conversation with follow-up questions

## 🔧 How It Works

### Document Processing Pipeline

```
Upload Files
    ↓
Extract Text (PDF/TXT)
    ↓
Split into Chunks (1000 chars, 300 overlap)
    ↓
Generate Embeddings (nomic-embed-text)
    ↓
Store in Session State
    ↓
Ready for Queries
```

### Query Processing Pipeline

```
User Question
    ↓
Generate Query Embedding
    ↓
Calculate Similarity Scores (Cosine)
    ↓
Retrieve Top 3 Most Relevant Chunks
    ↓
Combine Context + Question
    ↓
Generate Answer (llama3.2:1b)
    ↓
Display Response
```

## ⚙️ Configuration

You can customize the following parameters in the code:

### Chunking Parameters (in `chunk_text()`)
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 300)

### Retrieval Parameters (in `retrieve()`)
- `top_k`: Number of chunks to retrieve (default: 3)

### Model Parameters (at the top of the script)
- `model`: Change the LLM model (default: `llama3.2:1b`)
- `temperature`: Adjust response creativity (default: 0.3, lower = more deterministic)
- `embeddings_model`: Change the embeddings model (default: `nomic-embed-text`)

## 📊 Example Prompts

Once you've uploaded documents, try asking:
- "What is the main topic of this document?"
- "Summarize the key points in this file"
- "What does the document say about [specific topic]?"
- "How does [concept] relate to [another concept]?"

## 🐛 Troubleshooting

### "Connection refused" Error
- Make sure Ollama is running: `ollama serve`
- Verify it's accessible at `http://localhost:11434`

### Model Download Issues
- Download models manually:
  ```bash
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
  ```
- Check available models: `ollama list`

### Slow Responses
- The `llama3.2:1b` model is optimized for speed but smaller than alternatives
- For better quality, try `llama3:latest` (larger, slower)
- Ensure your device has adequate RAM (8GB+ recommended)

### Out of Memory Errors
- Reduce `chunk_size` in the `chunk_text()` function
- Reduce `top_k` in the `retrieve()` function
- Use a smaller model in Ollama

## 📁 Project Structure

```
rag-chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔐 Privacy & Security

- ✅ **Fully Local**: All processing happens on your machine
- ✅ **No API Calls**: No data is sent to external servers
- ✅ **No Authentication Required**: No API keys needed
- ✅ **Private Documents**: Your files are only processed locally

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [RAG Concepts](https://www.langchain.com/docs/use_cases/question_answering/)

**Made  using Streamlit and Ollama**
