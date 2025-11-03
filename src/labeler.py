from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

# small lexicon to create initial pseudo-labels
POS = set("good great excellent happy love satisfied pleased awesome like".split())
NEG = set("bad poor terrible awful hate unhappy frustrated upset angry dissatisfied".split())

def lexicon_label(text):
    tokens = str(text).split()
    pos = sum(1 for t in tokens if t in POS)
    neg = sum(1 for t in tokens if t in NEG)
    if pos>neg and pos>0:
        return "Positive"
    if neg>pos and neg>0:
        return "Negative"
    return "Neutral"

def create_pseudo_labels(df, text_col="text"):
    df = df.copy()
    df["clean_text"] = df[text_col].astype(str)
    df["pseudo_label"] = df["clean_text"].apply(lexicon_label)
    return df

def train_refined_classifier(df, text_col="clean_text", label_col="pseudo_label"):
    # train a simple TF-IDF + LogisticRegression on pseudo labels to refine predictions
    df = df.copy()
    df = df[[text_col, label_col]].dropna()
    X = df[text_col].values
    y = df[label_col].values
    # If all labels are the same, return a dummy predictor
    if len(set(y))==1:
        model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500))
        model.fit(X, y)
        return model
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=20000), LogisticRegression(max_iter=500))
    model.fit(X, y)
    return model

def predict_labels(model, texts):
    return model.predict(texts)