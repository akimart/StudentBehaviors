import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('Rockpaper.csv')

# Drop unnecessary columns
data = data.drop(columns=['TeamId', 'SubmissionDate'])

# Encode 'TeamName' as numeric
label_encoder = LabelEncoder()
data['TeamName'] = label_encoder.fit_transform(data['TeamName'])

# Define features and target
X = data.drop('Score', axis=1)
y = data['Score'].apply(lambda x: 'High' if x > 500 else 'Low')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define preprocessing for numeric and categorical data
numerical_cols = []
categorical_cols = ['TeamName']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Build a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])

# Define hyperparameters for GridSearch
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Apply GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearch
print("Best parameters:", grid_search.best_params_)

# Predict and evaluate
y_pred = grid_search.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Create heatmap of the classification report
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='Blues', fmt='.2f')
plt.title('Heatmap of Classification Report')
plt.show()

# Print classification report
print(report_df)
