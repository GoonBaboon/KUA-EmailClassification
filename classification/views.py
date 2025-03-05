import pickle
import os
from django.shortcuts import render
from .forms import EmailForm  # Create a Django form to get input


import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer  # Importing the Porter Stemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stopwords and stemmer
stop_words = set(stopwords.words('english'))  # Load stopwords once
ps = PorterStemmer()  # Initialize the stemmer

def transform(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove non-alphanumeric characters
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Stemming
    tokens = [ps.stem(word) for word in tokens]  # Apply stemming to each word

    return " ".join(tokens)

# Example Usage
text = "This is an example sentence with some common stopwords!"
transformed_text = transform(text)
print(transformed_text)


# Get the absolute path to model.pkl & vectorizer.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load the trained model and vectorizer
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Django View for Email Classification
# Django View for Email Classification
def index(request):
    prediction = None  # Default value
    if request.method == 'POST':
        form = EmailForm(request.POST)
        if form.is_valid():
            email_text = form.cleaned_data['email_text']

            # Preprocess the text
            transformed_text = transform(email_text)

            # Convert text into a format the model understands
            vectorized_text = vectorizer.transform([transformed_text])

            # Predict using the model
            result = model.predict(vectorized_text)[0]  # 0 or 1

            # Convert 0/1 to "Spam" or "Not Spam"
            prediction = "Spam" if result == 1 else "Not Spam"

    else:
        form = EmailForm()

    return render(request, 'index.html', {'form': form, 'prediction': prediction})

