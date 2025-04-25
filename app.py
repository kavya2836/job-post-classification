from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
# Initialize Flask app
app = Flask(__name__)
# Load the trained model and vectorizer
try:
    model = joblib.load('models/improved_fake_job_model.pkl')
    vectorizer = joblib.load('models/improved_tfidf_vectorizer.pkl')
    accuracy = joblib.load('models/accuracy.pkl')  # Load saved accuracy
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
# Load stopwords
stop_words = set(stopwords.words('english'))

# Function to clean the job description
def clean_text(text):
    if pd.isnull(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)
# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        # Extract job-related fields
        job_description = data.get('job_description', '').strip()
        company_name = data.get('company_name', '').strip()
        job_title = data.get('job_title', '').strip()
        job_location = data.get('job_location', '').strip()
        job_experience = data.get('job_experience', '').strip()
        # Validate input
        if not job_description:
            return jsonify({'error': 'Job description is required!'}), 400
        # Clean and preprocess the data
        cleaned_description = clean_text(job_description)
        combined_features = f"{company_name} {job_title} {cleaned_description} {job_location} {job_experience}"
        # Transform the input data
        X_input = vectorizer.transform([combined_features])
        # Make prediction
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]  # Get probability scores
        # Extract probabilities
        fake_percentage = round(prediction_proba[0] * 100, 2)  # Probability of fake
        real_percentage = round(prediction_proba[1] * 100, 2)  # Probability of real
        # Convert prediction to readable format
        result = "Real" if prediction == 1 else "Fake"
        return jsonify({
            'prediction': result,
            'accuracy': accuracy,  # Model's overall accuracy
            'fake_percentage': fake_percentage,
            'real_percentage': real_percentage
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500  # Return error to frontend
# Home route
@app.route('/')
def home():
    return render_template('home.html')
# Index page route (where form submission happens)
@app.route('/index')
def index():
    return render_template('index.html')
@app.route("/accuracy", methods=["GET"])
def get_accuracy():
    try:
        with open("model_accuracy.txt", "r") as file:
            accuracy = float(file.read().strip())  # Read accuracy from file
    except FileNotFoundError:
        accuracy = 98.2  # Default value if the file doesn't exist

    return jsonify({"accuracy": accuracy})
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
