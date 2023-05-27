import requests

# API endpoint URL
url = 'http://localhost:8000/predict'

# Query parameters
params = {
    'level': 3.1  # Provide the employee level for prediction
}

# Send GET request to the API
response = requests.get(url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the predicted salary from the response
    result = response.json()
    predicted_salary = result['predicted_salary']

    # Print the predicted salary
    print(f"Predicted Salary: {predicted_salary}")
else:
    # Print an error message if the request was unsuccessful
    print("Error occurred during the request")
