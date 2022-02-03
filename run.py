import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #genre and aid_related status
    aid_1 = df[df['aid_related']==1].groupby('genre').count()['message']
    aid_0 = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_names = list(aid_1.index)

    # let's calculate distribution of classes with 1
    class_1 = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)

    #sorting values in ascending
    class_1 = class_1.sort_values(ascending = False)

    #series of values that have 0 in classes
    class_0 = (class_1 -1) * -1
    class_name = list(class_1.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_1,
                    name = 'Need Aid', 
                    marker= dict(color= 'rgb(142,124,195)')),
                Bar(
                    x=genre_names,
                    y= aid_0,
                    name = 'Not aid related',
                    marker= dict(color= 'rgb(100,100,140)')
                )
            ],

            'layout': {
                'title': 'Distribution of message by channel',
                'yaxis': {
                    'title': "Volume"
                },
                'xaxis': {
                    'title': "Channel"
                },
                'barmode' : 'stack'
            }
        },
        {
            'data': [
                Bar(
                    x=class_name,
                    y=class_1,
                    name = 'Related'
 
                ),
                Bar(
                    x=class_name,
                    y=class_0,
                    name = 'Not related',
                    marker = dict(
                            color = 'rgb(212, 228, 247)'
                                )
                )
            ],

            'layout': {
                'title': 'Distribution of messages across disaster categories',
                'yaxis': {
                    'title': "Proportion of messages"
                },
                'xaxis': {
                    'title': "Disaster Categories"
                },
                'barmode' : 'stack'
            }
        }
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()