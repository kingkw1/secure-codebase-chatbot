# repo_chatbot

## Process Overview

1. ### **Code Describer & Interpreter**: Descriptive summaries & comments
2. ### **Document Crawler**: Structured metadata linked to each repo
3. ### **Embeddings-Based Indexer**: Vector database
4. ### **LLM Integration**: Query-able LLM informed about repos
5. ### **Query Bot Deployment**: Chatbot interface


## Running tests
Before any tests can be run, ollama must be installed and serving a model. 

ollama serve

ollama run codellama

## open-webui
open-webui serve

## Serve flask
python c:/Users/kingk/OneDrive/Documents/Projects/repo_chatbot/loaded_llm.py

## Serve pipelines (No longer need to serve?)
cd pipelines
bash start.sh

## docker -- (No longer using docker)
docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama
