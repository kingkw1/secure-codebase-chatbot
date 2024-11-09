import requests

# Define the URL of the Flask API
url = "http://10.5.0.2:5001/query"

# Define the payload (your query)
data = {
    'query': 'Is there a function to calcuate the average in the repository?'
}

# Send the POST request to the Flask app
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    print('Response:', response.json())
else:
    print('Error:', response.status_code, response.text)
