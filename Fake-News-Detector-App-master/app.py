from flask import Flask, jsonify, request, render_template
from prediction_model import PredictionModel
import pandas as pd
from random import randrange
from forms import OriginalTextForm
from newsapi import NewsApiClient
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = '4c99e0361905b9f941f17729187afdb9'

newsapi = NewsApiClient(api_key='c0893704897b49e1a11f745b0f106985')
DATASET_PATH = "random_dataset.csv"

# ===== Safe function to get real vs fake counts =====
def get_real_fake_counts():
    try:
        df = pd.read_csv(DATASET_PATH)
        # Ensure the CSV has 'label' column
        if 'label' not in df.columns:
            raise ValueError("CSV missing 'label' column")
        counts = df['label'].value_counts()
        data_storage = {
            'Category': ['Real News', 'Fake News'],
            'Value': [int(counts.get(1, 0)), int(counts.get(0, 0))]
        }
    except Exception as e:
        print("Pie chart error:", e)
        # Fallback demo data
        data_storage = {'Category': ['Real News', 'Fake News'], 'Value': [50, 50]}
    return data_storage

@app.route("/", methods=['GET', 'POST'])
def home():
    form = OriginalTextForm()
    pie_data = get_real_fake_counts()

    if form.generate.data:
        data = pd.read_csv(DATASET_PATH)
        index = randrange(0, len(data)-1)
        original_text = data.loc[index].text
        form.original_text.data = str(original_text)
        return render_template('home.html', form=form, output=False, pie_data=pie_data)

    elif form.predict.data:
        if len(str(form.original_text.data)) > 10:
            model = PredictionModel(form.original_text.data)
            prediction = model.predict()
            return render_template('home.html', form=form, output=prediction, pie_data=pie_data)

    return render_template('home.html', form=form, output=False, pie_data=pie_data)

@app.route('/predict/<original_text>', methods=['GET'])
def predict(original_text):
    model = PredictionModel(original_text)
    return jsonify(model.predict())

@app.route('/random', methods=['GET'])
def random_news():
    data = pd.read_csv(DATASET_PATH)
    index = randrange(0, len(data)-1)
    return jsonify({
        'title': data.loc[index].title,
        'text': data.loc[index].text,
        'label': str(data.loc[index].label)
    })

# ===== Pie Chart Page =====
@app.route('/pie_chart')
def pie_chart():
    pie_data = get_real_fake_counts()
    return render_template('pie_chart.html',
                           categories=pie_data['Category'],
                           values=pie_data['Value'])

@app.route('/top_india_news', methods=['GET'])
def top_india_news():
    top_headlines = newsapi.get_top_headlines(country='in', language='en', page_size=10)
    news_predictions = []
    for article in top_headlines['articles']:
        title = article['title']
        model = PredictionModel(title)
        prediction = model.predict()
        label = "Real ✅" if prediction == 1 else "Fake ❌"
        news_predictions.append({'title': title, 'prediction': label})
    
    return render_template('home.html',
                           form=OriginalTextForm(),
                           output=False,
                           news_predictions=news_predictions,
                           pie_data=get_real_fake_counts())

if __name__ == '__main__':
    app.run(debug=True)
