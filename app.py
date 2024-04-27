import re
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)

# convert string to float32
def convert_to_float32(embedding_str):
    # Remove square brackets and split by space
    embedding_values = embedding_str[1:-1].split()

    # Convert string values to float
    embedding_values = [float(value) for value in embedding_values]

    # Convert to numpy array
    embedding_array = np.array(embedding_values)

    # Convert data type to numpy.float32
    embedding_array = embedding_array.astype(np.float32)

    return embedding_array

# Load Subtitle Embedding
def load_data(path):
    data = pd.read_csv(path)
    return data['name'], data['embedding'].apply(convert_to_float32)

# path = r"D:\Search Engine Data\final_embedding_subtitles.csv"
# name, embedding = load_data(path)
# print(type(name), type(embedding), type(embedding[0]))

# Clearn Input data
def clean_input(subtitle):
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
    return model.encode(query)

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
########################################################################################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query = clean_input(query)
    embed_query = embedding_input(query)
    print(type(embed_query), type(embed_query[0]))
    

    return render_template('results.html', results=results)
####################################################################################################################
if __name__ == '__main__':
    app.run(debug=True)