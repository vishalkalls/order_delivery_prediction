# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import joblib

# Load datasets
orders = pd.read_csv('data/olist_orders_dataset.csv')
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
customers = pd.read_csv('data/olist_customers_dataset.csv')

# Merge datasets to create final dataframe
final_df = orders.merge(order_items, on='order_id', how='inner')
final_df = final_df.merge(payments, on='order_id', how='inner')
final_df = final_df.merge(customers, on='customer_id', how='inner')

# Drop unnecessary columns
final_df.drop(['customer_unique_id', 'customer_city', 'customer_state'], axis=1, inplace=True)

# Handle missing values
final_df.fillna(0, inplace=True)

# Create target variable (1 for delivered, 0 otherwise)
data = final_df.copy()
data['delivered'] = final_df['order_status'].apply(lambda x: 1 if x == 'delivered' else 0)

# Select relevant features for training
features = [
    'price', 'freight_value', 'payment_value', 'payment_installments'
]

# Check if required features exist
missing_features = [f for f in features if f not in final_df.columns]
if missing_features:
    raise ValueError(f"❌ Missing features: {missing_features}")

# Prepare input data (X) and target (y)
X = final_df[features]
y = data['delivered']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to 'model.pkl'
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model training complete and saved as 'model.pkl'")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate Model Performance
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Generate Classification Report
print("📄 Classification Report:\n", classification_report(y_test, y_pred))

# Generate Confusion Matrix
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
