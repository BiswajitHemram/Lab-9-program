#Step 1
# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

#Step 2
# Generating some sample data
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)

#Step 3
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4
# Creating a Logistic Regression model
model = LogisticRegression()

#Step 5
# Training the model
model.fit(X_train, y_train)

#Step 6
# Making predictions on the test data
y_pred = model.predict(X_test)

#Step 7
# Calculating accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Step 8
# Plotting decision boundary (for 2D data only)
if X.shape[1] == 2:
    # Create meshgrid of feature 1 and feature 2 values
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each grid point
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plotting the decision boundary
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

#Step 9
# Plotting the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()