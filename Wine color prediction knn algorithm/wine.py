# Import necessary libraries
import numpy as np  # NumPy for numerical operations
from sklearn.model_selection import train_test_split  # To split the dataset
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier

# Sample wine dataset with two features: alcohol content and acidity
data = np.array([[12.5, 0.6], [11.3, 0.8], [13.1, 0.7], [10.9, 0.5], [12.8, 0.6], [9.9, 0.75]])
labels = np.array(['red', 'white', 'red', 'white', 'red', 'white'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a KNN classifier with k=3 (3 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Create a new wine sample for prediction with alcohol content 11.0 and acidity 0.7
new_wine = np.array([[11.0, 0.6]])

# Predict the color of the new wine using the trained model
predicted_color = knn.predict(new_wine)

# Print the predicted wine color
print("Predicted wine color:", predicted_color[0])
