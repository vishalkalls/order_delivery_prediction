# app.py
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    price = float(request.form['price'])
    freight_value = float(request.form['freight_value'])
    payment_value = float(request.form['payment_value'])
    payment_installments = int(request.form['payment_installments'])

    # Create DataFrame for prediction
    data = pd.DataFrame([[price, freight_value, payment_value, payment_installments]],
                        columns=['price', 'freight_value', 'payment_value', 'payment_installments'])
    
    # Make prediction
    prediction = model.predict(data)

    # Return result
    result = "✅ Order will be delivered successfully!" if prediction[0] == 1 else "❌ Order might not be delivered on time."
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
