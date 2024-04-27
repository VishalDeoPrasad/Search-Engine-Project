import re
import joblib
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

model_path = r'model/sentence_transformer_model.joblib'
# Load the model using joblib
model = joblib.load(model_path)

print("Model loaded successfully from", model_path)
print(model)

# Load Embedding
def load_embedding(file_path):

    # Load the embeddings array from the binary file
    loaded_embeddings_array = np.load(file_path)

    # Now you can use the loaded_embeddings_array as needed
    print("Embeddings array loaded successfully from:", file_path)
    return loaded_embeddings_array

file_path = r"subtitles data/embeddings.npy"
embeddings = load_embedding(file_path)
print(type(embeddings), type(embeddings[0]), type(embeddings[0][0]))

# Load the TV-shows and movie name
def load_name(file_path):
    name = pd.read_csv(file_path)

    # Now you can use the loaded_embeddings_array as needed
    print("Movie name loaded successfully from:", file_path)

    return name
    
file_path = r"subtitles data/tv_movie_names.csv"
name = load_name(file_path)
print(type(name))

# Clean Input data and embedded the data
def clean_and_embedd_subtitle(query):
    # Remove timestamp and special characters
    query = re.sub(r'<[^>]*>', '', query)  # Remove HTML tags
    query = re.sub(r'\r\n', ' ', query)  # Replace newlines with spaces
    query = re.sub(r'[^a-zA-Z\s]', '', query)  # Remove non-alphabetic characters
    query = re.sub(r'\s+', ' ', query).strip() # Remove extra spaces

    # Convert to lowercase
    query = query.lower()

    # Tokenization
    words = word_tokenize(query)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join tokens back into a single string
    cleaned_subtitle = ' '.join(words)
    emb_query = model.encode(cleaned_subtitle)
    return emb_query

# perform the cosine similarity
def similarity_finder(search_query, k, emb):

    # Calculate cosine similarity between the query and the documents
    similarity_scores = cosine_similarity(search_query, emb)

    # Sort the similarity scores and get the indices of most similar documents
    similar_doc_indices = similarity_scores.argsort()[0][::-1][:min(k, len(name))]

    # Retrieve and return the most similar documents
    similar_texts = [name['name'].iloc[i] for i in similar_doc_indices]

    return similar_texts

########################################################################################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    emb_query = clean_and_embedd_subtitle(query)
    print(type(emb_query), type(emb_query[0]))
    similar_text = similarity_finder([emb_query], 10, embeddings)  # Get up to 10 most similar documents

    return render_template('results.html', results=similar_text)
#######################################################################################################
if __name__ == '__main__':
    app.run(debug=True)