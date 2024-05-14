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


def get_synonyms(word):
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def get_cosine_similarity(string1, string2):
    '''Calculates cosine similarity (with synonyms)'''
    # Tokenize strings into words
    words1 = string1.split()
    words2 = string2.split()

    # Collect all unique words from both strings
    unique_words = set(words1 + words2)

    # Construct sentences with synonyms
    sentences = []
    for word in unique_words:
        synonyms = get_synonyms(word)
        for synonym in synonyms:
            sentence1 = ' '.join([synonym if w == word else w for w in words1])
            sentence2 = ' '.join([synonym if w == word else w for w in words2])
            sentences.append(sentence1)
            sentences.append(sentence2)

    # Compute TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    # Compute cosine similarity between TF-IDF vectors
    similarity = cosine_similarity(tfidf_matrix[0::2], tfidf_matrix[1::2])

    # Get maximum similarity score
    max_similarity = similarity.max()

    return max_similarity


def main():
    parser = argparse.ArgumentParser(description= "Specify a LLM-generated response or file and (optionally) the model that generated it")
    parser.add_argument('response', type = str, help="LLM-generated response")
    args = parser.parse_args()

    load_dotenv()
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
    print(get_AI_response(chat, "explain the difference between code and psuedocode in simple terms"))

    # Load the classifier model
    with open('svc_role_classifier.pkl', 'rb') as f:
        svc_classifier = pickle.load(f)

    # Load the tf_idf vectorizer
    with open('svc_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    predicted_role = svc_classifier.predict(vectorizer.transform([args.response]))
    

    

main()

