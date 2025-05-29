import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from wordcloud import WordCloud
import plotly.express as px
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Initialize Dash app with Bootstrap CSS for badges and styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global storage for uploaded dataframe (in memory for simplicity)
uploaded_df = {'data': None}

# --- Helper functions ---

def simple_preprocess(text, custom_stopwords):
    # lowercase, remove punctuation, remove stopwords
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in custom_stopwords]
    return ' '.join(words)

def get_sentiment(text):
    # simple sentiment polarity
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def perform_topic_modeling(texts, n_topics=3):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)
    topics = lda.transform(dtm).argmax(axis=1)
    # Get top keywords for each topic
    words = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_idx, comp in enumerate(lda.components_):
        word_idx = comp.argsort()[::-1][:10]
        keywords = ", ".join(words[i] for i in word_idx)
        topic_keywords.append(keywords)
    # Map topic index to string label
    index_to_name = {i: f"Topic {i+1}: {topic_keywords[i]}" for i in range(n_topics)}
    return topics, topic_keywords, index_to_name

# --- Layout ---

app.layout = html.Div([
    html.H2("Deep aura NLP Dashboard"),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
        style={
            'width': '98%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '3px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    html.Div(id='upload-feedback'),
    
    html.Div([
        html.Label("Select Text Column:"),
        dcc.Dropdown(id='text-column-dropdown', options=[], value=None)
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '10px'}),
    
    html.Div([
        html.Label("Preprocessing Steps (comma separated, e.g. lowercase,remove_punct):"),
        dcc.Input(id='preprocess-options', type='text', value='lowercase,remove_punct', style={'width': '100%'}),
    ], style={'width': '48%', 'display': 'inline-block', 'margin': '10px'}),
    
    html.Div([
        html.Label("Custom Stopwords (comma separated):"),
        dcc.Input(id='custom-stopwords', type='text', value='', style={'width': '100%'}),
    ], style={'width': '48%', 'display': 'inline-block', 'margin': '10px'}),
    
    html.Div([
        html.Label("Number of Topics for LDA:"),
        dcc.Input(id='num-topics', type='number', value=3, min=2, max=10, step=1),
    ], style={'width': '20%', 'display': 'inline-block', 'margin': '10px'}),
    
    html.Button("Generate Analysis", id='generate-button', n_clicks=0, style={'margin': '10px'}),
    html.Button("Remove Words", id='remove-words-button', n_clicks=0, style={'margin': '10px'}),
    
    html.Div(id='removed-words-feedback', style={'margin': '10px'}),
    
    dcc.Loading([
        dcc.Graph(id='wordcloud-graph')
    ], type='circle'),
    
    dcc.Loading([
        dcc.Graph(id='sentiment-bar')
    ], type='circle'),
    
    html.Div(id='processed-table-preview', style={'margin': '10px', 'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'scroll'})
])

# --- Callbacks ---

@app.callback(
    Output('upload-feedback', 'children'),
    Output('text-column-dropdown', 'options'),
    Output('text-column-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_csv(contents, filename):
    if contents is None:
        return "", [], None
    
    content_type, content_string = contents.split(',')
    import base64
    import io
    
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return f"Unsupported file type: {filename}", [], None
    except Exception as e:
        return f"Error loading file: {str(e)}", [], None
    
    uploaded_df['data'] = df
    options = [{'label': c, 'value': c} for c in df.columns if df[c].dtype == 'object']
    if not options:
        return "No text columns found in uploaded data.", [], None
    return f"Loaded {filename} with {df.shape[0]} rows, {df.shape[1]} columns.", options, options[0]['value']

@app.callback(
    [Output('wordcloud-graph', 'figure'),
     Output('processed-table-preview', 'children'),
     Output('sentiment-bar', 'figure'),
     Output('removed-words-feedback', 'children')],
    [Input('generate-button', 'n_clicks'),
     Input('remove-words-button', 'n_clicks')],
    [State('preprocess-options', 'value'),
     State('text-column-dropdown', 'value'),
     State('num-topics', 'value'),
     State('custom-stopwords', 'value'),
     State('processed-table-preview', 'children')],
    prevent_initial_call=True
)
def process_and_display(gen_clicks, remove_clicks, steps, column, n_topics, custom_stopword_string, current_table):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    df = uploaded_df.get('data')
    if df is None or column not in df:
        return {}, "Missing data/column.", {}, "No data or column selected."
    
    # Parse custom stopwords list
    custom_stopwords = [w.strip().lower() for w in custom_stopword_string.split(',')] if custom_stopword_string else []
    
    def full_processing_pipeline(text, steps, stopwords):
        text = str(text)
        if 'lowercase' in steps:
            text = text.lower()
        if 'remove_punct' in steps:
            text = re.sub(r'[^\w\s]', '', text)
        # You can extend more steps here
        words = text.split()
        words = [w for w in words if w not in stopwords]
        return ' '.join(words)
    
    if button_id == 'generate-button':
        text_col = df[column].fillna('').astype(str)
        processed = text_col.apply(lambda x: full_processing_pipeline(x, steps.split(','), custom_stopwords))
        sentiments = processed.apply(get_sentiment)
        topic_assignments, topic_keywords, index_to_name = perform_topic_modeling(processed.tolist(), n_topics)
        
        out_df = pd.DataFrame({
            'Original': text_col,
            'Processed': processed,
            'Sentiment': sentiments,
            'Topic': [index_to_name[i] for i in topic_assignments]
        })
        
        wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(processed))
        wc_fig = px.imshow(wc.to_array())
        wc_fig.update_layout(title='Word Cloud', xaxis_showticklabels=False, yaxis_showticklabels=False)
        
        sentiment_counts = out_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        sent_fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', title='Sentiment Distribution')
        
        table = dash_table.DataTable(
            data=out_df.head().to_dict('records'),
            columns=[{'name': i, 'id': i} for i in out_df.columns],
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'scroll'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto',
                        'minWidth': '100px', 'width': '150px', 'maxWidth': '200px'},
            page_size=5
        )
        
        feedback = html.Div([
            html.Span("Initial processing complete", style={'color': 'green'}),
            dbc.Badge(f"{len(custom_stopwords)} words removed", color="success" if custom_stopwords else "secondary")
        ])
        
        return wc_fig, table, sent_fig, feedback
    
    elif button_id == 'remove-words-button' and current_table:
        if not custom_stopwords:
            return dash.no_update, dash.no_update, dash.no_update, "No words specified for removal"
        
        current_data = pd.DataFrame(current_table['props']['data'])
        
        def remove_words(text):
            words = text.split()
            return ' '.join([word for word in words if word.lower() not in custom_stopwords])
        
        current_data['Processed'] = current_data['Processed'].apply(remove_words)
        
        wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(current_data['Processed']))
        wc_fig = px.imshow(wc.to_array())
        wc_fig.update_layout(title='Word Cloud', xaxis_showticklabels=False, yaxis_showticklabels=False)
        
        sentiment_counts = current_data['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        sent_fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', title='Sentiment Distribution')
        
        table = dash_table.DataTable(
            data=current_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in current_data.columns],
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'scroll'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto',
                        'minWidth': '100px', 'width': '150px', 'maxWidth': '200px'},
            page_size=5
        )
        
        feedback = html.Div([
            html.Span(f"Removed words: {', '.join(custom_stopwords)}", style={'color': 'red'}),
            dbc.Badge(f"{len(custom_stopwords)} words removed", color="danger")
        ])
        
        return wc_fig, table, sent_fig, feedback
    
    return dash.no_update, dash.no_update, dash.no_update, ""

# --- Run server ---

if __name__ == '__main__':
    app.run(debug=True)
