# Import necessary libraries
import pickle
import pandas as pd

# ✅ Load the trained model from 'model.pkl'
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Check model type after loading
print(f"✅ Model Type: {type(model)}")

# ✅ Define new sample data for prediction
new_data = pd.DataFrame({
    'price': [54.90],
    'freight_value': [20.10],
    'payment_value': [75.00],
    'payment_installments': [2]
})

# ✅ Print data type to confirm correctness
print(f"✅ New Data for Prediction:\n{new_data}\n")
print(f"✅ Data Type: {type(new_data)}")

# ✅ Make predictions
prediction = model.predict(new_data)

# ✅ Save predictions to a CSV file
new_data['prediction'] = prediction
new_data.to_csv('predictions.csv', index=False)
print("✅ Predictions saved to 'predictions.csv'")

# ✅ Print prediction result
if prediction[0] == 1:
    print("✅ Order will be delivered successfully!")
else:
    print("❌ Order delivery might be delayed or unsuccessful.")
