from flask import Flask, request, jsonify, render_template
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')

# Load the vectorizer and model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
nb_classifier = joblib.load('nb_classifier_model.pkl')

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word.lower() not in stop_words]
    return ' '.join(tokens)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        case_text = request.form['case_text']
        preprocessed_text = preprocess_text(case_text)
        vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
        prediction = nb_classifier.predict(vectorized_text)
        return jsonify({'category': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
