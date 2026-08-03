# Document QA Agent (RAG-based)

An AI agent that reads documents, answers questions strictly based on the content of those documents using Retrieval-Augmented Generation (RAG), and replies to the user with grounded answers. Includes safe failure handling for cases where the answer isn't found in the document.

## Features
- Accepts user-provided documents as the knowledge source (no external/general knowledge used for answering)
- Chunks documents into smaller segments for effective retrieval
- Embeds chunks and stores them in a vector store for semantic search
- Retrieves the most relevant chunks for a given user query (RAG pipeline)
- Generates answers grounded only in the retrieved document content
- Safe failure handling: if the answer isn't present in the document, the agent responds with a clear "not found in document" message instead of guessing or hallucinating
- Handles edge cases like empty documents, empty queries, or unsupported file formats gracefully

.
## Model Details
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (via `SentenceTransformer`)
- Chat/generation model: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `[commit hash — see "Files and versions" tab on the model's Hugging Face page, or main if not pinned]`
- Framework/library versions: `[e.g. sentence-transformers==3.0.1, transformers==4.42.0, torch==2.3.0]`

## Hardware Requirements
- Minimum to run: [e.g. "4 CPU cores, 8GB RAM, no GPU required"]
- Hardware used for this run:
  - CPU: [e.g. Intel i7-12700H]
  - RAM: [e.g. 16GB]
  - GPU/Accelerator: [e.g. NVIDIA RTX 3060 6GB / "None — ran on CPU only"]

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone [your repo URL]
   cd [repo-folder-name]
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (if needed):
   ```bash
   export API_KEY=your_key_here
   ```
4. Run the agent:
   ```bash
   python main.py
   ```



Example:
```
Input: ["Our daily dashboard exports stopped appearing at the expected time after an Admin changed the workspace timezone yesterday. The schedule still looks active. What should we check, and can the missed export be recovered?"]


Output: ["Could you clarify your request? I need more specific details to help.] # Cus we dont have any relevant data ,This is Safe Failure
```

