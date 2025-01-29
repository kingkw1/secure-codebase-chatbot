# repo_chatbot

## Process Overview

#### 1. **Code Describer & Interpreter**: Descriptive summaries & comments
#### 2. **Document Crawler**: Structured metadata linked to each repo
#### 3. **Embeddings-Based Indexer**: Vector database
#### 4. **LLM Integration**: Query-able LLM informed about repos
#### 5. **Query Bot Deployment**: Chatbot interface


## Installation

### 1. Install Python version 3.11

### 2. Install Depedancies
```markdown
pip install -r requirements.txt
```

### Install Pipeline dependencies
```markdown
pip install -r pipelines/requirements-minimum.txt
```
### 3. Install ollama

### 4. Install ollama models
ollama pull llama3.2
ollama pull codellama

## Zscaler Certificate Debugging:
May have a certificate error if receiving the following error:


    (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:992)')))

If this happens, it is likely an issue with zscaler and managing certificates. Do the following:

### 1. Get Zscaler certificates

### 2. Place certificates in a folder in C drive
C:\certificates\ZscalerSHA256.pem

### 3. Set the path to use these certificates
```markdown
$env:REQUESTS_CA_BUNDLE = "C:\certificates\ZscalerSHA256.pem"
```

### 4. Verify that the environment variable was set
```markdown
echo $env:REQUESTS_CA_BUNDLE
```

### 5. Retry running the rag_agent.py file
This solved the certificate error for me on a company laptop. 

#### Note: The solution described above may be a temporary fix, and the environment variable may need to be set every time computer restarts. To make it permanent, set the variable as shown below. Adjust the path accordingly.
```markdown
[System.Environment]::SetEnvironmentVariable("REQUESTS_CA_BUNDLE", "C:\certificates\ZscalerSHA256.pem", "User")
```

## Initializing
### 1. Run Ollama

### 2. Serve the flask app
```markdown
.\.venv\Scripts\activate
python rag_agent_app.py
```

### 3. Serve the pipelines
```markdown
bash start_pipelines.sh
```

- Note: After the pipelines have been served, and open-webui finds them, the pipelines can be edited and then updated within the webui from the admin panel, rather than restarting the pipeline server

### 4. Serve open-webui 
```markdown
.\.venv\Scripts\activate
open-webui serve
```

### 5. Open open-webui in browser
http://localhost:8080

### 6. Connect the pipeline within the browser
- Note: Can be skipped if pipeline existed within the directory: repo_chatbot/pipelines/pipelines, and has not been changed

http://localhost:8080/admin/settings
> Settings > Pipelines > Pipeline Valves 
- Verify that "flask_app_pipeline (pipe)" is within the Pipelines Valves
- Upload pipeline here if needed

### 7. Select the pipeline from wthin chat window.
http://localhost:8080
> select "Flask App Connector Pipeline" from the top left drop down

### Chat away!

## Debugging: Run tests
Note: These tests are largely subjective in nature

#### 1. test_connection
#### 2. test_comment
#### 3. test_summary
#### 4. test_metadata_extraction
#### 5. test_embeddings
#### 6. test_query
