import requests

# Send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

print(f"Status Code: {r.status_code}")

# Access and print the welcome message from the correct key
result = r.json()
print(f"Result: {result[0]}")

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r = requests.post("http://127.0.0.1:8000/data", json=data)
result = r.json()
print(f"Status Code: {r.status_code}")
print(f"Result: {result.get('result', 'Key not found')}")
