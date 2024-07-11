from openai import OpenAI

import os
from dotenv import load_dotenv
from astrapy.db import AstraDBCollection

load_dotenv()

# Grab the Astra token and api endpoint from the environment
token = os.getenv("NEW_ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("NEW_ASTRA_DB_API_ENDPOINT")
keyspace = os.getenv("NEW_ASTRA_DB_KEYSPACE")
openai_api_key = os.getenv("OPENAI_API_KEY")
collection_name = os.getenv("NEW_ASTRA_DB_COLLECTION_NAME")
dimension = os.getenv("VECTOR_DIMENSION")
model = os.getenv("VECTOR_MODEL")

# Client for OpenAI API
client = OpenAI(api_key = openai_api_key)

def get_similar_docs(query, number):
    if not keyspace:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint)
    else:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint, namespace=keyspace)
    embedding = list(client.embeddings.create( input=query, model=model).data[0].embedding)
    relevant_docs = collection.vector_find(embedding, limit=number)
    doc_contents = []
    for doc in relevant_docs:
        article_id = doc['article_id']
        chunk_index = doc['chunk_index']
        context_threshold = 3
        full_context = []
        for i in range(max(0, chunk_index - context_threshold)):
            full_context.append(collection.find({"article_id": article_id, "chunk_index": i})['data']['documents'][0]["content"])
        #print(full_context)
        doc_contents.append("".join(full_context))
    return doc_contents

# prompt that is sent to openai using the response from the vector database and the users original query
prompt_boilerplate = "Answer the question posed in the user query section using the provided context"
user_query_boilerplate = "USER QUERY: "
document_context_boilerplate = "CONTEXT: "
final_answer_boilerplate = "Final Answer: "

def build_full_prompt(query):
    relevant_docs = get_similar_docs(query, 3)
    docs_single_string = "\n".join(relevant_docs)

    nl = "\n"
    filled_prompt_template = prompt_boilerplate + nl + user_query_boilerplate+ query + nl + document_context_boilerplate + docs_single_string + nl + final_answer_boilerplate
    print(filled_prompt_template)
    return filled_prompt_template

def send_to_openai(full_prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": full_prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

