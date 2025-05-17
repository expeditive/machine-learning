import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')

# Step 2: Split features and labels
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Save the trained model
joblib.dump(model, 'diabetes_model.pkl')

print("Model trained and saved as 'diabetes_model.pkl'")
