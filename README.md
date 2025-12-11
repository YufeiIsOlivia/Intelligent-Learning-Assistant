# Intelligent Learning Assistant: Interactive PDF Learning and Self-Assessment System

A web-based RAG (Retrieval-Augmented Generation) system that transforms static PDF documents into interactive learning experiences. The system enables users to learn from PDF documents through conversational Q&A and automated self-assessment quizzes.

## Features

- **PDF Document Processing**: Upload and parse PDF files with automatic text extraction and chunking
- **Conversational Q&A**: Ask questions about PDF content and receive answers with source citations (page numbers and text snippets)
- **Adaptive Retrieval Strategy**: Automatically classifies questions into types (Fact, Summary, Procedure, Compare/Analyze) and adjusts retrieval strategy accordingly
- **Self-Assessment Quizzes**: Generate multiple-choice questions from PDF content for practice and self-testing
- **Multi-Model Support**: Access multiple LLM models (Claude, GPT, Gemini) via OpenRouter API
- **LLM Evaluation Framework**: Comprehensive evaluation system using LLM-as-a-Judge methodology to compare different models

## System Architecture

The system uses a modular RAG architecture:

- **PDF Parser**: Extracts text while preserving page information using PyPDF2
- **Vector Store**: ChromaDB for semantic search and document retrieval
- **Embedding Model**: OpenAI's text-embedding-ada-002 for generating document embeddings
- **RAG System**: Retrieves relevant chunks and generates answers using LLMs
- **Question Classification**: Automatically classifies questions and adapts retrieval strategy (8-40 chunks based on question type)
- **Web Interface**: Simple, intuitive interface for document upload and interaction

## Requirements

- Python 3.8 or higher
- API Keys:
  - **OpenRouter API Key** (recommended): Get from https://openrouter.ai/keys
  - **OpenAI API Key** (required for embeddings): Get from https://platform.openai.com/api-keys

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YufeiIsOlivia/Insight-Hub.git
cd Insight-Hub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in the project root:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
FORCE_OPENROUTER=true
```

4. Run the server:
```bash
python main.py
```

5. Open your browser and navigate to `http://localhost:8000`

## Usage

### Uploading PDFs
1. Click "Choose File" and select a PDF document
2. Click "Upload PDF" to process and store the document
3. Wait for the upload confirmation message

### Asking Questions
1. Type your question in the input field
2. Optionally select a specific LLM model (default uses system configuration)
3. Click "Ask Question" to get an answer with source citations
4. View the answer with page numbers and text snippets for verification

### Generating Quizzes
1. Click "Generate Quiz" to create multiple-choice questions from the uploaded PDFs
2. Answer the questions to test your understanding
3. Review explanations for each answer

## Question Classification and Adaptive Retrieval

The system automatically classifies questions and adjusts retrieval strategy:

| Question Type | Keywords/Patterns | Retrieval Chunks | Use Case |
|--------------|------------------|------------------|----------|
| **Fact** | "what is", "who is", "when", "where", "define" | 8 | Focused factual queries |
| **Summary** | "summary", "overview", "main points", "about this pdf" | 40 | Document summaries |
| **Procedure** | "step", "process", "how to", "procedure" | 40 | Step-by-step processes |
| **Compare/Analyze** | "compare", "difference", "versus", "analyze" | 30 | Comparative analysis |
| **Default** | Other questions | 25-30 | General questions |

## API Endpoints

- `POST /api/upload`: Upload a PDF file
- `POST /api/ask`: Ask a question about uploaded PDFs (supports optional `model_name` parameter)
- `GET /api/quiz`: Generate multiple-choice quiz questions
- `GET /api/status`: Get system status, document count, and configuration

## Evaluation Framework

The `evaluation/` directory contains tools for evaluating different LLM models:

- **QA Dataset Generation** (`generate_qa_dataset.ipynb`): Automatically generates questions and reference answers from PDFs
- **LLM Evaluation** (`llm_judge_evaluation.ipynb`): Tests multiple models and evaluates them using LLM-as-a-Judge methodology
- **Evaluation Metrics**: Retrieval Relevance, Faithfulness (Groundedness), and Answer Quality

See `evaluation/README.md` for detailed evaluation instructions.

## Project Structure

```
.
├── main.py                      # FastAPI application entry point
├── backend/
│   ├── pdf_parser.py           # PDF parsing and text extraction
│   ├── rag_system.py           # RAG implementation with question classification
│   └── vector_store.py         # ChromaDB vector database management
├── frontend/
│   └── index.html              # Web interface
├── evaluation/
│   ├── generate_qa_dataset.ipynb    # QA dataset generation
│   ├── llm_judge_evaluation.ipynb   # Model evaluation framework
│   ├── qa_dataset.json              # Generated QA dataset
│   ├── llm_judge_results/           # Evaluation results
│   ├── figures/                     # Evaluation visualization figures
│   └── README.md                    # Evaluation guide
├── uploads/                         # Uploaded PDF files (auto-created)
├── vector_db/                       # Vector database storage (auto-created)
├── requirements.txt                 # Python dependencies
└── .env                             # Environment variables (not in Git)
```

## Configuration

### Model Selection
The system supports multiple LLM models via OpenRouter:
- Claude 3.5 Sonnet (`anthropic/claude-3-5-sonnet`)
- GPT-3.5 Turbo (`openai/gpt-3.5-turbo`)
- Gemini 2.5 Flash (`google/gemini-2.5-flash`)

You can specify a model when asking questions, or the system will use the default configured model.

### Environment Variables
- `OPENROUTER_API_KEY`: Required for accessing multiple LLM models via OpenRouter
- `OPENAI_API_KEY`: Required for embeddings (text-embedding-ada-002)
- `FORCE_OPENROUTER`: Set to `true` to force OpenRouter for all LLM calls (useful for evaluation)
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

