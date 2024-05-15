# External libraries
import tensorflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import argparse
import pickle

# NLTK
from nltk.corpus import wordnet as wn

# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def get_AI_response(AI_model, prompt):
    response = AI_model.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    ).content
    return response


def main():
    parser = argparse.ArgumentParser(description= "Specify a LLM-generated response or file and (optionally) the model that generated it")
    parser.add_argument('response', type = str, help="LLM-generated response")
    args = parser.parse_args()

    load_dotenv()
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

    # Load the classifier model
    with open('svc_role_classifier.pkl', 'rb') as f:
        svc_classifier = pickle.load(f)

    # Load the tf_idf vectorizer
    with open('svc_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    predicted_role = svc_classifier.predict(vectorizer.transform([args.response]))
    query = f"What question are you asked if you can generate the following answer: {args.response}"
    
    if predicted_role != ['none']:
        print(f"you are a {predicted_role}" + get_AI_response(chat, query))
    else:
        print(get_AI_response(chat, query))
    

main()

