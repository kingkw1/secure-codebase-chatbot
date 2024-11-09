import requests

url = "http://10.5.0.2:5001/query"
query = "Explain the purpose of the mean function."
payload = {"query": query}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)
    print("Response Text:", response.text)