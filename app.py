# Importing the Libraries
import os
import pickle
import urllib
import flask
from flask import Flask, request, render_template
from flask_cors import CORS
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split


# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='templates')

with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


def fake_news_det(news):
    tfvect.fit_transform(x_train)
    tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/link')
def link():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')


# Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict', methods=['GET' , 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary

    # Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))


# Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predicted', methods=['POST'])
def predicted():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
