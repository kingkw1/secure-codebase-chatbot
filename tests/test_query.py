import requests

url = "http://127.0.0.1:5001/query"
payload = {"query": "Explain the purpose of the mean function."}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())