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

## Crawling your repo
### 1. Modify config file

### 2. Crawl the repo by running embeddings.py

### 3. Initialize the chatbot

## Initializing Chatbot
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

- Note: If receiving a bunch of syntax errors, do the following IN A GIT BASH WINDOW:
    ```markdown
    cd your/project/directory
    dos2unix start_pipelines.sh
    ```

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

#### Connect pipelines
> Settings > Connections > Manage OpenAI API Connections
- Add a new connection:
    - API URL: http://locahost:9099 
    - API key: 0p3n-w3bu!
- (not really sure why we need to enter this as an OpenAI API connection, except that the Ollama API connections are missing fields that we need)

#### Connect the custom pipelines
> Settings > Pipelines > Pipeline Valves 
- If "flask_app_pipeline (pipe)" is NOT within the Pipelines Valves, then upload pipeline here
- If pipelines says "Pipelines not detected", confirm that pipelines is connected as indicated in the step above
- Select the chatbot's endpoint api: 
    - Endpoint Api: <your_endpoint_api>>
    - To get <your_endpoint_api>, look at the terminal window in which we ran rag_agent_app.py. It should say the following:

```markdown
PS address\to\your\project\repo_chatbot> python rag_agent_app.py
 * Serving Flask app 'rag_agent_app'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://10.2.0.2:5001   (THIS IS <YOUR_ENDPOINT_API>)
```

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


## Certificate errors on company machines:
```
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --cert C:\certificates\ZscalerSHA256.pem  
```