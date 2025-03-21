
# 🔐 secure-codebase-chatbot

A **privacy-first, AI-powered chatbot** for navigating and understanding private codebases. Designed to integrate with **Azure AI Services** and operate securely on private infrastructure—ideal for sensitive environments where keeping code confidential is critical.

---

## 🚀 Features

### 1. **Private Codebase Analysis**  
- Crawls local/private repositories (no third-party code exposure).  
- Generates structured metadata for each project (functions, files, descriptions).

### 2. **Embeddings-based Search Engine**  
- Builds a private **vector database** (FAISS) for fast, secure lookups.  
- Supports advanced semantic search for code structure and documentation.

### 3. **LLM-Powered Insights**  
- Integrates with **Azure OpenAI Service** or other on-prem LLM deployments.  
- Produces code summaries, documentation, and intelligent responses to code queries.

### 4. **Customizable Chatbot Interface**  
- Connects to **Open WebUI** via pipelines.  
- Query the chatbot securely via browser (localhost) or internal networks.

---

## ⚙️ **Installation**

### 1. Prerequisites
- **Python 3.11+**
- **Ollama** (for running local models)
- **Open WebUI** (chat interface)
- Azure AI Services or a locally hosted LLM

### 2. Clone and install dependencies
```bash
git clone https://github.com/your-username/secure-codebase-chatbot.git
cd secure-codebase-chatbot
pip install -r requirements.txt
```

---

## 🗂️ **How It Works**

1. **Crawl a private repo**
   - Configure your repo settings in `config/`.
   - Run the crawler to generate metadata and embeddings:

    ```bash
    python embeddings.py
    ```

2. **Launch the chatbot backend**
   ```bash
   .\.venv\Scripts\activate
   python rag_agent_app.py
   ```

3. **Serve the pipelines (for Open WebUI integration)**
   ```bash
   bash start_pipelines.sh
   ```

4. **Launch Open WebUI**
   ```bash
   open-webui serve
   ```

5. **Connect everything via browser**  
   Open `http://localhost:8080` and link the pipeline:
   - **Admin Panel > Settings > Connections > OpenAI API Connection**  
     - API URL: `http://localhost:9099`  
     - API Key: `0p3n-w3bu!`

   - **Admin Panel > Pipelines > Pipeline Valves**  
     - Connect your chatbot pipeline to the endpoint shown by `rag_agent_app.py`.

6. **Chat with your private repo!**  
   - Select the custom pipeline inside Open WebUI chat window (top-left dropdown).  
   - Ask questions like:
     - "What does the `process_data()` function do?"  
     - "Summarize the purpose of `user_controller.py`."

---

## Azure Deployment Diagram 
```mermaid
graph TD
    A["Private GitHub Repo / GitHub Enterprise 🔒"] -->|🔒 Pull Code| B1["Repo Crawler & Metadata Extractor 🔒"]
    
    subgraph Server["Secure Azure VM / On-Prem Server 🔒"]
        B1["Repo Crawler & Metadata Extractor 🔒"]
        B2["FAISS Vector DB (Private) 🔒"]
        B3["Ollama LLM API (Self-hosted) 🔒"]
        B4["Flask App (RAG Service) 🔒"]
        B5["Open WebUI (Chatbot Interface) 🔒"]
    end
    
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B4 --> B5
    B5 -->|🔒 Secure HTTPS / Internal IP| C["Internal Developer Machines 🔒"]
    
    style A fill:#d9e8fb,stroke:#333,stroke-width:2px
    style B1 fill:#e0f7da,stroke:#333
    style B2 fill:#e0f7da,stroke:#333
    style B3 fill:#e0f7da,stroke:#333
    style B4 fill:#e0f7da,stroke:#333
    style B5 fill:#e0f7da,stroke:#333
    style C fill:#fbcfe8,stroke:#333,stroke-width:2px
    
    click B1 "https://learn.microsoft.com/en-us/azure/virtual-network/virtual-networks-overview" "Azure VNet: Private traffic only"

```
---

## 🧰 **Development Workflow**

- Code and test **locally** or inside your private network.
- Your codebase **never leaves your infrastructure**—no cloud uploads.
- Fully compatible with **air-gapped environments**.

---

## 🧪 **Debugging & Tests**

To verify core features, run:
```bash
python tests/test_connection.py
python tests/test_comment.py
python tests/test_summary.py
python tests/test_metadata_extraction.py
python tests/test_embeddings.py
python tests/test_query.py
```

---

## 🔐 **Why Privacy-First?**

Unlike traditional dev chatbots, this system is designed for teams working on **proprietary or regulated codebases**. Keep your sensitive code **private, secure, and compliant** while still leveraging the power of modern AI tooling.

---

## 📝 Notes

- For Windows systems with strict cert requirements:
  ```bash
  python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --cert C:\certificates\ZscalerSHA256.pem
  ```
- If you encounter file formatting issues on Git Bash:
  ```bash
  dos2unix start_pipelines.sh
  ```

---

## 💡 **Future Additions**
- Multimodal input (voice commands + code navigation).
- CI/CD pipeline integration.
- Fine-tuned LLM deployment on private servers.
