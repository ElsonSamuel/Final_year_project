import re, os
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify, render_template
import nltk

nltk.download('stopwords')

# Initialize Flask app

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'template'))

# Load the pre-trained model and vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize Porter Stemmer
port_stem = PorterStemmer()

# Function for stemming and preprocessing input text
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Home route serving an HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Replace with your HTML file for the home page

# Route for processing comments
@app.route('/process_comment', methods=['POST'])
def process_comment():
    data = request.json
    comment = data.get('comment', '')

    if not comment:
        return jsonify({'error': 'Comment is required'}), 400

    try:
        # Preprocess the input comment
        processed_comment = stemming(comment)
        vectorized_comment = vectorizer.transform([processed_comment])
        prediction = model.predict(vectorized_comment)
        
        if prediction[0] == 1:
            sentiment = "Positive 😊"
        else:
            sentiment = "Negative 😞"
            
        return jsonify({'comment': comment, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

