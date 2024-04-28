import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS module
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set the path for templates directory
template_dir = os.path.abspath(os.path.dirname(__file__))
app.template_folder = os.path.join(template_dir)

# Load the machine learning model
model = LinearRegression()
df = pd.read_csv("insurance.csv")  # Make sure to have your CSV file in the same directory
X = df[['age', 'sex', 'bmi', 'children', 'smoker']]
y = df['charges']
model.fit(X, y)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the HTML form
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # Female=0, Male=1
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])  # No=0, Yes=1
        
        # Make prediction using the model
        prediction = model.predict([[age, sex, bmi, children, smoker]])
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
