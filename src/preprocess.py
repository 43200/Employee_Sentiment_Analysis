import re
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # basic cleaning
    text = text.replace("\n"," ").replace("\r"," ")
    text = re.sub(r'http\S+','', text)
    text = re.sub(r'[^A-Za-z0-9\s\']',' ', text)
    text = re.sub(r'\s+',' ', text).strip().lower()
    return text