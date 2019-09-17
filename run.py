import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import numpy as np
import plotly.graph_objs as g
app = Flask(__name__)

def tokenize(text):
    '''
    Function to break down a document (sentence) into tokens (words)
    while removing stopwords, regular expressions, white space and punctuations
    Args:
        text - the sentence (document) to tokenize
    Returns:
        A list of words needed to train a machine learning model.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df1 = pd.read_sql('DisasterResponse.db', engine)
df = df1[~df1['offer-0'].isnull()]

# load model
model = joblib.load("../models/classifier3.pkl")
#model.n_jobs = 1
#n_jobs = 1

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Function to display visuals from the imported training dataset
    Args:
        None
    Returns:
        None (Dispalys visuals)
    '''
    # extract data needed for visuals

    #First figure
    categ = {}
    for i in df.columns[4:]:
        if i == 'child_alone-0':
            categ[i] = 0
        else:
            categ[i] = df.loc[:,i].value_counts()[1]
    cat_names = list(pd.Series(categ).index)
    use_last2 = lambda x:x[:-2]
    cat_names = pd.Series(cat_names).apply(use_last2).values
    cat_val = list(pd.Series(categ).values)
    catvalues1 = pd.DataFrame(cat_names)
    catvalues1['names'] = pd.DataFrame(cat_names)
    catvalues1['values'] = pd.DataFrame(cat_val)
    catvalues2 = catvalues1[['values','names']]
    catvalues2.sort_values(['values'], ascending = True,axis = 0, inplace=True)
    values = catvalues2['values']
    names1 = catvalues2['names']
    #colorcode = np.arange(4*len(names1))
    colorcode = np.arange(1*len(names1))

    
    #Second figure
    #import seaborn as sns
    corr = df.iloc[:,4:].corr().values
    cols1 = catvalues1['names'].values
    #cols1 = df.columns.values
    #corr = corr.to_json()
    
    
    #Third figure
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #import plotly.graph_objects as go
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=catvalues2['values'],
                    y=names1,
                    orientation='h',
                    marker=dict(color=colorcode)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
        {
            'data': [
                   g.Heatmap(
            z = corr,
            y=cols1,
            x=cols1,
            type = 'heatmap'
           )
             ],
            'layout': {
                'title': 'How are the message categories related?',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
         {
            'data': [
                Bar(
                    x=genre_counts,
                    y=genre_names,
                    orientation='h',
                    marker=dict(color='blue')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Counts"
                }
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
    '''
    Function to take in user querry and classify it while bringing out the
    classification results visuals
    Args:
        None
    Returns:
        Classification results
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    use_last2 = lambda x:x[:-2]
    cat_names1 = pd.Series(df.columns[4:]).apply(use_last2).values
    cat_names1 = list(cat_names1)
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(cat_names1, classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    '''
    Function to impement all thefunctions need for a job
    Args:
        None
    Returns:
        None
    '''
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()