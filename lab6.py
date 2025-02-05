import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet


# Download necessary NLTK resources (only required once)
nltk.download('punkt')
nltk.download('wordnet')

# Sample text
text = "Tokenization is an important NLP technique. It involves breaking down text into smaller units, such as words or subwords."


# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)


# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(token) for token in tokens]
print("Stemmed words:", stemmed_words)



# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]
print("Lemmatized words:", lemmatized_words)