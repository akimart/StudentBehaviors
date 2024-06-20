import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load the dataset
data = pd.read_csv('Rockpaper.csv')

# Drop unnecessary columns
data = data.drop(columns=['TeamId', 'SubmissionDate'])

# Define features and target
X = data.drop('Score', axis=1)
y = data['Score']

# One-hot encode categorical columns
X = pd.get_dummies(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Decision Tree Regression model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'DTRModel.pkl')
