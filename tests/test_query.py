import requests

url = "http://127.0.0.1:5001/query"
payload = {"query": "Explain the purpose of the mean function."}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)
    print("Response Text:", response.text)