from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the spam detection model and the TfidfVectorizer
model = joblib.load('saved_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Transform the input message using the loaded vectorizer
    input_data_features = vectorizer.transform([message])

    # Predict if the message is spam or not
    prediction = model.predict(input_data_features)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
