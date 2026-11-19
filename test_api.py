
import requests

url = "http://127.0.0.1:8000/predict"

student_data = {
    "age": 20,
    "study_hours": 5,
    "attendance": 90
}

response = requests.post(url, json=student_data)

print("Status code:", response.status_code)
print("Prediction:", response.json())
