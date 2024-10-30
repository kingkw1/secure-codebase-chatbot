import requests

def test_connection(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Connection successful!")
        else:
            print(f"Failed to connect. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    ollama_server_url = "http://localhost:11434"
    test_connection(ollama_server_url)