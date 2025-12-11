# Quick Start Guide

Get your Intelligent Learning Assistant up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- OpenRouter API key (get from https://openrouter.ai/keys)
- OpenAI API key (get from https://platform.openai.com/api-keys)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Keys

Create a `.env` file in the project root directory:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
FORCE_OPENROUTER=true
```

**Important**: 
- Replace `your_openrouter_api_key_here` with your actual OpenRouter API key
- Replace `your_openai_api_key_here` with your actual OpenAI API key
- The `.env` file is already in `.gitignore` and will not be committed to Git
- Never share your API keys or commit them to version control

## Step 3: Start the Server

```bash
python main.py
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Step 4: Open the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

## Step 5: Upload a PDF

1. Click "Choose File" and select a PDF document
2. Click "Upload PDF"
3. Wait for the confirmation message indicating successful upload

## Step 6: Ask Questions

1. Type your question in the input field
2. (Optional) Select a specific LLM model from the dropdown
3. Click "Ask Question"
4. View the answer with source citations (page numbers and text snippets)

### Example Questions to Try

- **Factual**: "What is machine learning?"
- **Summary**: "What are the main topics in this document?"
- **Procedural**: "What are the steps for feature selection?"
- **Comparative**: "What's the difference between supervised and unsupervised learning?"

## Step 7: Generate Quiz (Optional)

1. Click "Generate Quiz" to create multiple-choice questions from the uploaded PDFs
2. Answer the questions to test your understanding
3. Review explanations for each answer

## Running Evaluations (Advanced)

If you want to evaluate different LLM models:

1. **Start the server** (if not already running):
```bash
python main.py
```

2. **Upload PDFs** via the web interface

3. **Generate QA Dataset**:
   - Open `evaluation/generate_qa_dataset.ipynb` in Jupyter
   - Run all cells to generate questions from your PDFs
   - Dataset will be saved as `evaluation/qa_dataset.json`

4. **Run Evaluation**:
   - Open `evaluation/llm_judge_evaluation.ipynb` in Jupyter
   - Update `MODELS_TO_TEST` with models you want to test
   - Run all cells to evaluate models
   - Results will be saved in `evaluation/llm_judge_results/`

See `evaluation/README.md` for detailed evaluation instructions.

## Troubleshooting

### Server won't start
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.8+)
- Check for port conflicts (default port is 8000)
- Try: `python3 main.py` instead of `python main.py`

### API errors
- Verify your API keys are correct in `.env`
- Check your API key quotas and limits at the provider websites
- Ensure `FORCE_OPENROUTER=true` is set if using OpenRouter
- Check the terminal for detailed error messages

### No answers generated
- Make sure you've uploaded at least one PDF
- Check `/api/status` endpoint to verify documents are in the system
- Try a simpler question first
- Verify the PDF contains relevant content

### Questions not working
- Ensure the PDF contains relevant content
- Try rephrasing your question
- Check the browser console (F12) for errors
- Verify the server is running and accessible

### Import errors
- Make sure you're in the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Try creating a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

