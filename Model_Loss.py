import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Rockpaper.csv'  # Path file yang diunggah
data = pd.read_csv(file_path)

# Drop the 'Unnamed: 0' column as it is just an index
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Define features and target variable
X = data.drop(columns=['TeamName'])
y = data['TeamName']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing for numerical data and categorical data
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define the preprocessing steps for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create a preprocessing and modeling pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Preprocess the data
X_train_processed = model_pipeline.fit_transform(X_train)
X_test_processed = model_pipeline.transform(X_test)

# Hyperparameters
input_dim = X_train_processed.shape[1]
hidden_neurons = 64
output_neurons = len(label_encoder.classes_)
dropout_rate = 0.5
epochs = 100
batch_size = 32

# Build the ANN model
model = Sequential()
model.add(Dense(hidden_neurons, input_dim=input_dim, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(hidden_neurons, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(output_neurons, activation='softmax'))  # Assuming multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_processed, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_processed, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Visualisasi hasil pelatihan

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
