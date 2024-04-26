import re
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
#from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# from sklearn.metrics.pairwise import cosine_similarity
#model_distilbert = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)

# Clearn Input data
def clean_input(query):
    # Remove timestamp and special characters
    subtitle = re.sub(r'<[^>]*>', '', subtitle)  # Remove HTML tags
    subtitle = re.sub(r'\r\n', ' ', subtitle)  # Replace newlines with spaces
    subtitle = re.sub(r'[^a-zA-Z\s]', '', subtitle)  # Remove non-alphabetic characters
    subtitle = re.sub(r'\s+', ' ', subtitle).strip() # Remove extra spaces

    # Convert to lowercase
    subtitle = subtitle.lower()

    # Tokenization
    words = word_tokenize(subtitle)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join tokens back into a single string
    cleaned_subtitle = ' '.join(words)
    return cleaned_subtitle

def embedding_input(query):
    pass

"""# Load pre-trained SentenceTransformer model
model_distilbert = SentenceTransformer('all-MiniLM-L6-v2')

# Load the subtitle data
subset_df = pd.read_csv('embedding.csv', nrows=100)

# Encode subtitles in batches
batch_size = 32
num_subtitles = len(subset_df)
subtitle_embeddings = []
for i in range(0, num_subtitles, batch_size):
    batch_subtitles = subset_df['clean_text_lemma'].iloc[i:i+batch_size].values
    batch_embeddings = model_distilbert.encode(batch_subtitles)
    subtitle_embeddings.append(batch_embeddings)

# Concatenate batch embeddings into a single array
subtitle_embeddings = np.concatenate(subtitle_embeddings)"""

"""
# Function to perform semantic search
def semantic_search(query, top_n=5):
    # Encode the query using the SentenceTransformer model
    query_embedding = model_distilbert.encode([query])[0]

    # Calculate cosine similarity between the query embedding and all subtitle embeddings
    similarities = cosine_similarity([query_embedding], subtitle_embeddings)[0]

    # Get indices of top N most similar subtitles
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return the top N most similar subtitles
    return subset_df['name'].iloc[top_indices].tolist()
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    #query = clean_input(query)
    results = [
        'operation.fortune.ruse.de.guerre.(2023).eng.1cd',
        'operation.fortune.ruse.de.guerre.(2023).eng.1cd',
        'survivors.remorse.s01.e03.how.to.build.a.brand.(2014).eng.1cd',
        'queer.eye.s06.e05.crawzaddy.(2021).eng.1cd',
        'tales.of.wells.fargo.s05.e20.the.hand.that.shook.the.hand.(1961).eng.1cd',
        'survivors.remorse.s01.e03.how.to.build.a.brand.(2014).eng.1cd',
        'nightingales.s02.e05.reach.for.the.sky.(1993).eng.1cd',
        'operation.fortune.ruse.de.guerre.(2023).eng.1cd',
        'magnum.p.i.s02.e19.may.the.best.one.win.(2020).eng.1cd',
        'the.voice.s22.e23.live.semifinal.top.8.eliminations.(2022).eng.1cd'
]

    #results = ["APPLE", "BoY", "CAT", "DOG", "ELEPHANT", "FISH", "GOAT", "HAT", "ICE", query]  #semantic_search(query)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)