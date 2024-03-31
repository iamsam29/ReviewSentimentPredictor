import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the Naive Bayes classifier
with open('model/naive_bayes.pkl', 'rb') as file:
    nb_clf = joblib.load(file)

# Define a function for text preprocessing
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W|\d', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a single string
    text = ' '.join(words)

    return text

# Initialize Flask app
app = Flask(__name__)

# Define a route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from the form
        text = request.form['text']

        # Preprocess the text
        text = preprocess_text(text)

        # Make a prediction using the Naive Bayes classifier
        prediction = nb_clf.predict([text])

        # Render the results template with the prediction
        return render_template('results.html', prediction=prediction[0])

    return render_template('index.html')

# Define a route for the results template
@app.route('/results/<prediction>')
def results(prediction):
    # Render the results template with the prediction
    return render_template('results.html', prediction=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)