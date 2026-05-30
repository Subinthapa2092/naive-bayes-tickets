"""Shared text preprocessing and numeric feature extraction."""

import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NEGATIONS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 'without',
    'cannot', "can't", "won't", "don't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
}
STOP_WORDS = set(stopwords.words('english')) - NEGATIONS

KEYWORD_GROUPS = {
    'flag_refund': r'(?:refund|reimburse|money back|return|chargeback)',
    'flag_technical': r'(?:error|bug|crash|not work|broken|fail|glitch|freeze|hang|malfunction)',
    'flag_cancellation': r'(?:cancel|cancellation|terminate|end subscription|unsubscribe)',
    'flag_product': r'(?:how to|feature|compatible|specification|version|upgrade|install)',
    'flag_billing': r'(?:bill|invoice|payment|charge|overcharg|subscription|price|cost)',
}

NUMERIC_FEATS = [
    'char_count', 'word_count_raw', 'avg_word_len', 'sentence_count',
    'exclamation_count', 'question_count', 'upper_ratio',
    'word_count_clean', 'unique_word_ratio',
]

_lemmatizer = WordNetLemmatizer()


def replace_product_placeholder(description: str, product: str = '') -> str:
    if not isinstance(description, str):
        return ''
    return description.replace('{product_purchased}', product or '')


def build_ticket_text(subject: str, description: str, product: str = '') -> str:
    description = replace_product_placeholder(description, product)
    return f'{subject or ""} {description or ""}'.strip()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'#?\b[a-z]*\d+[a-z0-9]*\b', ' ', text)
    text = re.sub(r"[^a-z\s']", ' ', text)
    text = re.sub(r"(?<![a-z])'", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def add_numeric_features(df: pd.DataFrame, text_col: str = 'ticket_description') -> pd.DataFrame:
    df = df.copy()
    text = df[text_col].fillna('')
    text_lower = text.str.lower()

    df['char_count'] = text_lower.str.len()
    df['word_count_raw'] = text_lower.str.split().str.len()
    df['avg_word_len'] = df['char_count'] / (df['word_count_raw'] + 1e-9)
    df['sentence_count'] = text_lower.str.count(r'[.!?]+')
    df['exclamation_count'] = text_lower.str.count(r'!')
    df['question_count'] = text_lower.str.count(r'\?')
    df['upper_ratio'] = text_lower.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-9)
    )
    df['word_count_clean'] = df['clean_description'].str.split().str.len()
    df['unique_word_ratio'] = df['clean_description'].apply(
        lambda x: len(set(x.split())) / (len(x.split()) + 1e-9)
    )
    return df


def extract_numeric_features(raw_text: str, cleaned_text: str) -> pd.DataFrame:
    row = pd.DataFrame([{
        'ticket_description': raw_text,
        'clean_description': cleaned_text,
    }])
    row = add_numeric_features(row)
    return row[NUMERIC_FEATS]
