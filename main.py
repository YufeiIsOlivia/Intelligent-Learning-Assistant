"""
Main FastAPI application for PDF RAG Q&A system.
"""
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Optional
import shutil
from pathlib import Path

from backend.pdf_parser import PDFParser
from backend.vector_store import VectorStore
from backend.rag_system import RAGSystem

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PDF RAG Q&A System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

VECTOR_DB_DIR = Path("vector_db")
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Initialize components
pdf_parser = PDFParser()
vector_store = VectorStore(persist_directory=str(VECTOR_DB_DIR))
rag_system = None  # Will be initialized on first use


def get_rag_system():
    """Get or initialize RAG system."""
    global rag_system
    if rag_system is None:
        # Try OpenAI first (recommended), then OpenRouter as fallback
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if openai_key:
            # Use OpenAI API (recommended)
            rag_system = RAGSystem(vector_store, openai_key, use_openrouter=False)
        elif openrouter_key:
            # Fallback to OpenRouter
            rag_system = RAGSystem(vector_store, openrouter_key, use_openrouter=True)
        else:
            raise HTTPException(
                status_code=500,
                detail="API key not found. Please set OPENAI_API_KEY in .env file. Get your key from https://platform.openai.com/api-keys"
            )
    return rag_system


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_path = Path("frontend/index.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        JSON response with upload status
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse PDF
        chunks = pdf_parser.parse_pdf(str(file_path))
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Get embeddings for the chunks
        rag = get_rag_system()
        texts = [chunk['text'] for chunk in chunks]
        embeddings = rag.get_embeddings(texts)
        
        # Add to vector store with embeddings
        vector_store.add_documents(chunks, file.filename, embeddings=embeddings)
        
        return JSONResponse({
            "status": "success",
            "message": f"PDF uploaded and processed successfully",
            "filename": file.filename,
            "chunks": len(chunks),
            "total_documents": vector_store.get_collection_size()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/api/ask")
async def ask_question(data: dict):
    """
    Ask a question about the uploaded PDFs.
    
    Args:
        data: Dictionary with 'question' key
        
    Returns:
        JSON response with answer and citations
    """
    question = data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        rag = get_rag_system()
        result = rag.ask_question(question)
        
        return JSONResponse({
            "status": "success",
            "answer": result["answer"],
            "citations": result["citations"]
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Get system status."""
    try:
        # Try to get collection size, but handle case where collection doesn't exist yet
        try:
            total_docs = vector_store.get_collection_size()
        except Exception:
            total_docs = 0
        
        return JSONResponse({
            "status": "success",
            "total_documents": total_docs,
            "has_api_key": bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")),
            "using_openai": bool(os.getenv("OPENAI_API_KEY")),
            "using_openrouter": bool(os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"))
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@app.delete("/api/clear")
async def clear_documents():
    """Clear all uploaded documents from the vector store."""
    try:
        vector_store.delete_collection()
        # Also clear uploaded files
        for file in UPLOAD_DIR.glob("*.pdf"):
            file.unlink()
        
        return JSONResponse({
            "status": "success",
            "message": "All documents cleared"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")


@app.post("/api/generate-quiz")
async def generate_quiz(data: dict = None):
    """
    Generate quiz questions based on uploaded PDFs.
    
    Args:
        data: Optional dictionary with 'num_questions' key (default: 10)
        
    Returns:
        JSON response with quiz questions
    """
    num_questions = 10
    if data and 'num_questions' in data:
        try:
            num_questions = int(data['num_questions'])
            if num_questions < 1 or num_questions > 20:
                num_questions = 10  # Default to 10 if invalid
        except (ValueError, TypeError):
            num_questions = 10
    
    try:
        rag = get_rag_system()
        questions = rag.generate_quiz_questions(num_questions=num_questions)
        
        return JSONResponse({
            "status": "success",
            "questions": questions,
            "total_questions": len(questions)
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

