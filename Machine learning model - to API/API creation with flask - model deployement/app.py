from flask import Flask, request, jsonify

app = Flask(__name__)

# Training the Polynomial Regression model on the dataset
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Endpoint for predicting the salary based on employee level
@app.route('/predict', methods=['GET'])
def predict_salary():
    employee_level = float(request.args.get('level'))

    # Perform the prediction based on the employee level
    salary_prediction = lin_reg_2.predict(poly_reg.fit_transform([[employee_level]]))

    # Return the predicted salary
    result = {
        'employee_level': employee_level,
        'predicted_salary': salary_prediction[0]
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)


# python -m waitress --listen=0.0.0.0:8000 app:app