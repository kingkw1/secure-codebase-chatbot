import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - ALL from environment variables
endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")
admin_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")

# Sample documents to upload
documents = [
    {
        "id": "1",
        "content": "LLM chatbot for querying internal GitHub repositories.",
        "category": "AI",
    },
    {
        "id": "2",
        "content": "FAISS vector database for similarity search within code repositories.",
        "category": "Database",
    },
    {
        "id": "3",
        "content": "Deploy and secure a Flask-based RAG service on Azure VM.",
        "category": "DevOps",
    },
    {
        "id": "4",
        "content": "Integrating Open WebUI to provide chatbot interface to developers.",
        "category": "Frontend",
    },
    {
        "id": "5",
        "content": "Private GitHub repo crawler and metadata extractor running on a secure server.",
        "category": "Security",
    }
]

def upload_documents():
    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key)
    )

    result = client.upload_documents(documents=documents)
    print("Upload results:")
    for r in result:
        print(f"Document ID: {r.key}, Status: {r.status_code}")
        

if __name__ == "__main__":
    upload_documents()
