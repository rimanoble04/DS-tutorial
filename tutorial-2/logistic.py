import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset from sklearn
iris = datasets.load_iris()
X = iris.data  # Features: sepal_length, sepal_width, petal_length, petal_width
y = iris.target  #species

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Logistic Regression model
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


sample_data = [[5.1, 3.5, 1.4, 0.2]]  #[sepal_length, sepal_width, petal_length, petal_width]
sample_data_scaled = scaler.transform(sample_data)
predicted_class = model.predict(sample_data_scaled)
predicted_species = iris.target_names[predicted_class][0]
print(f"Predicted species for sample {sample_data}: {predicted_species}")
