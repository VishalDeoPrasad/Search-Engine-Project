# Search-Engine-Project
### Background:
In the fast-evolving landscape of digital content, effective search engines play a pivotal role in connecting users with relevant information. For Google, providing a seamless and accurate search experience is paramount. This project focuses on improving the search relevance for video subtitles, enhancing the accessibility of video content.

### Objective:
Develop an advanced search engine algorithm that efficiently retrieves subtitles based on user queries, with a specific emphasis on subtitle content. The primary goal is to leverage natural language processing and machine learning techniques to enhance the relevance and accuracy of search results.

### Keyword based vs Semantic Search Engines:
__Keyword Based Search Engine:__ These search engines rely heavily on exact keyword matches between the user query and the indexed documents.
__Semantic Search Engines:__ Semantic search engines go beyond simple keyword matching to understand the meaning and context of user queries and documents.
__Comparison__: While keyword-based search engines focus primarily on matching exact keywords in documents, semantic-based search engines aim to understand the deeper meaning and context of user queries to deliver more relevant and meaningful search results. 

### Core Logic:
To compare a user query against a video subtitle document, the core logic involves three key steps:
1. Preprocessing of data: 
    1. If you have limited compute resources, you can take a random 30% of the data.
    1. Clean: A possible cleaning step can be to remove time-stamps  (Note: Cleaning the Text data is crucial before vectorization)  
    1. Vectorize the given Subtitle Documents
2. Take the user query and vectorize the User Query.
3. Cosine Similarity Calculation:
Compute the cosine similarity between the vector of the documents and the vector of the user query.
This similarity score determines the relevance of the documents to the user's query.
4. Return the most similar documents

### About dataset
Database contains a sample of 82498 subtitle files from opensubtitles.org. 

Most of the subtitles are of movies and tv-series which were released after 1990 and before 2024.

Database File Name: eng_subtitles_database.db
Database contains a table called 'zipfiles' with three columns.
1. num: Unique Subtitle ID reference for www.opensubtitles.org 
2. name: Subtitle File Name
3. content: Subtitle file were compressed and stored as a binary using 'latin-1' encoding.

You can use 'num' to get more details about each subtitle by going to the following link:
https://www.opensubtitles.org/en/subtitles/{num}
**Replace {num} with Unique Subtitle ID.
