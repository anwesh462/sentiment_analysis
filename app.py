from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ''
    if request.method == 'POST':
        review = request.form['review']
        vect = vectorizer.transform([review])
        pred = model.predict(vect)[0]
        sentiment = 'Positive 😊' if pred == 1 else 'Negative 😞'
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
