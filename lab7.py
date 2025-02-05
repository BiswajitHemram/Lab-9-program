# Python code to implement Bag of Words

# step 1 - Import

from sklearn.feature_extraction.text import CountVectorizer

# Step 2
# Sample documents
documents = [
"Game of Thrones is an amazing tv series!",
"Game of Thrones is the best tv series!",
"Game of Thrones is so great",
"Game of Thrones can be improved in its visuals"
]

# Step 3
# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Step 4
# Fit the vectorizer to the documents and transform them into a BoW matrix
X = vectorizer.fit_transform(documents)
# print(X)

# Step 5
# Get the vocabulary (unique words) and the BoW matrix
vocabulary = vectorizer.get_feature_names_out()
bow_matrix = X.toarray()

# Step 6
# Display the vocabulary and the BoW matrix
print("Vocabulary (Unique Words):")
print(vocabulary)

print("\nBag of Words Matrix:")
print(bow_matrix)