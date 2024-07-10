import os
import sys

from dotenv import load_dotenv


import requests
import json
from astrapy.db import AstraDBCollection, AstraDB
from openai import OpenAI
import ftfy
import pprint
import re
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('popular')

load_dotenv()

# Grab the Astra token and api endpoint from the environment
old_token = os.getenv("OLD_ASTRA_DB_APPLICATION_TOKEN")
old_api_endpoint = os.getenv("OLD_ASTRA_DB_API_ENDPOINT")
old_keyspace = os.getenv("OLD_ASTRA_DB_KEYSPACE")
old_collection_name = os.getenv("OLD_ASTRA_DB_COLLECTION_NAME")

new_token = os.getenv("NEW_ASTRA_DB_APPLICATION_TOKEN")
new_api_endpoint = os.getenv("NEW_ASTRA_DB_API_ENDPOINT")
new_keyspace = os.getenv("NEW_ASTRA_DB_KEYSPACE")
new_collection_name = os.getenv("NEW_ASTRA_DB_COLLECTION_NAME")

# API key for OpenAI
openai_api_key = os.getenv("openai_key")

model = os.getenv("VECTOR_MODEL")

# Initialize AstraDB instance and AstraDBCollection instances for input and output collections
old_astra_db = AstraDB(token=old_token, api_endpoint=old_api_endpoint)
in_collection = AstraDBCollection(collection_name=old_collection_name, astra_db=old_astra_db)

new_astra_db = AstraDB(token=new_token, api_endpoint=new_api_endpoint)
out_collection = AstraDBCollection(collection_name=new_collection_name, astra_db=new_astra_db)

# Client for OpenAI API
client = OpenAI(api_key = openai_api_key)

#NLTK Sentence Tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initial state for pagination
nextPageState = ""

articles_to_process = 1
batch_size = 20
total_articles = 1481

for i in range(0, total_articles, batch_size):
    current_batch_size = batch_size
    batch_max = i+batch_size
    if batch_max > articles_to_process:
        current_batch_size = articles_to_process - i
        
    if nextPageState == "":
        data = in_collection.find()
        nextPageState = data['data']['nextPageState']
        ids = [article['_id'] for article in data['data']['documents'][0:int(current_batch_size)]]
        articles = ["".join(article['content']) for article in data['data']['documents'][0:int(current_batch_size)]]
        for i in range(len(articles)):
            #Recombine the shredded article
            article = ftfy.fix_text(re.sub(r'(\s*\n\s*)(\s*\n\s*)(\s*\n\s*)+', "\n\n", articles[i]))
            id = ids[i]
            if article != "[embedded content]":
                article_sentences = tokenizer.tokenize(article)
                article_sentence_embeddings = sentence_model.encode(article_sentences)
                article_sentence_embedding_similarities = []
                for j in range(len(article_sentence_embeddings) - 1):
                    A = np.array(article_sentence_embeddings[j])
                    B = np.array(article_sentence_embeddings[j+1])
                    article_sentence_embedding_similarities.append(np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)))
                #print(article_sentence_embedding_similarities)
                article_semantic_chunks = []
                chunk_storage = ""
                for j in range(0, len(article_sentence_embeddings)-1):
                    similarity_threshold = 0.5
                    if article_sentence_embedding_similarities[j] <= similarity_threshold:
                        if chunk_storage == "":
                            article_semantic_chunks.append(article_sentences[j])
                        else:
                            article_semantic_chunks.append(" ".join([chunk_storage, article_sentences[j]]))
                            chunk_storage = ""
                    else:
                        chunk_storage = chunk_storage + article_sentences[j]
                if article_sentence_embedding_similarities[-1] <= similarity_threshold:
                    article_semantic_chunks[-1] = " ".join([article_semantic_chunks[-1], article_sentences[-1]])
                else:
                    article_semantic_chunks.append(article_sentences[-1])
                for j in range(len(article_semantic_chunks)):
                    snippet_text = article_semantic_chunks[j]
                    embedding = client.embeddings.create( input=snippet_text, model=model).data[0].embedding
                    out_collection.insert_one(document={"$vector": embedding, "content": snippet_text, "article_id": id, "chunk_index": j})
            else:
                print("Skipping broken article.")
        print(nextPageState)
        if current_batch_size < batch_size:
            break
    elif nextPageState == None:
        break
    else:
        data = in_collection.find(options={"pageState":nextPageState}, sort = None)
        nextPageState = data['data']['nextPageState']
        ids = [article['_id'] for article in data['data']['documents'][0:int(current_batch_size)]]
        articles = ["".join(article['content']) for article in data['data']['documents'][0:int(current_batch_size)]]
        for i in range(len(articles)):
            #Recombine the shredded article
            article = ftfy.fix_text(re.sub(r'(\s*\n\s*)(\s*\n\s*)(\s*\n\s*)+', "\n\n", articles[i]))
            id = ids[i]
            if article != "[embedded content]":
                article_sentences = tokenizer.tokenize(article)
                article_sentence_embeddings = sentence_model.encode(article_sentences)
                article_sentence_embedding_similarities = []
                for j in range(len(article_sentence_embeddings) - 1):
                    A = np.array(article_sentence_embeddings[j])
                    B = np.array(article_sentence_embeddings[j+1])
                    article_sentence_embedding_similarities.append(np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)))
                #print(article_sentence_embedding_similarities)
                article_semantic_chunks = []
                chunk_storage = ""
                for j in range(0, len(article_sentence_embeddings)-1):
                    similarity_threshold = 0.5
                    if article_sentence_embedding_similarities[j] <= similarity_threshold:
                        if chunk_storage == "":
                            article_semantic_chunks.append(article_sentences[j])
                        else:
                            article_semantic_chunks.append(" ".join([chunk_storage, article_sentences[j]]))
                            chunk_storage = ""
                    else:
                        chunk_storage = chunk_storage + article_sentences[j]
                if article_sentence_embedding_similarities[-1] <= similarity_threshold:
                    article_semantic_chunks[-1] = " ".join([article_semantic_chunks[-1], article_sentences[-1]])
                else:
                    article_semantic_chunks.append(article_sentences[-1])
                for j in range(len(article_semantic_chunks)):
                    snippet_text = article_semantic_chunks[j]
                    embedding = client.embeddings.create( input=snippet_text, model=model).data[0].embedding
                    out_collection.insert_one(document={"$vector": embedding, "content": snippet_text, "article_id": id, "chunk_index": j})
            else:
                print("Skipping broken article.")
        print(nextPageState)
        if current_batch_size < batch_size:
            break

            
            
