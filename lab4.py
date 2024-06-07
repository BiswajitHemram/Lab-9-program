# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 2
# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3
# Creating decision tree classifier
clf = DecisionTreeClassifier()

# Step 4
# Training the classifier
clf.fit(X, y)

# Step 5
# Visualizing the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()