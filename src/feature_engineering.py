"""
Build processed data, feature matrices, and preprocessing artifacts from raw CSV.

Usage:
    python src/feature_engineering.py
"""

import os
import pickle
import warnings
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from preprocessing import NUMERIC_FEATS, add_numeric_features, build_ticket_text, preprocess_text

warnings.filterwarnings('ignore')

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

PROJ_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJ_ROOT / 'data' / 'raw' / 'customer_support_tickets.csv'
PROCESSED_DIR = PROJ_ROOT / 'data' / 'processed'
FEATURES_DIR = PROJ_ROOT / 'data' / 'features'
MODELS_DIR = PROJ_ROOT / 'models'

RANDOM_SEED = 42
TEST_SIZE = 0.20


def run():
    for d in (PROCESSED_DIR, FEATURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('FEATURE ENGINEERING')
    print('=' * 60)

    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df[['ticket_subject', 'ticket_description', 'product_purchased', 'ticket_type']].dropna()
    df = df.reset_index(drop=True)
    print(f'\nLoaded {len(df)} tickets')

    df['ticket_description'] = df.apply(
        lambda row: build_ticket_text(
            row['ticket_subject'],
            row['ticket_description'],
            row['product_purchased'],
        ),
        axis=1,
    )
    df['clean_description'] = df['ticket_description'].apply(preprocess_text)
    df = df[df['clean_description'].str.strip() != ''].reset_index(drop=True)
    df = add_numeric_features(df)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['ticket_type'])
    print(f'Classes: {dict(zip(le.classes_, range(len(le.classes_))))}')

    X_text = df['clean_description']
    y = df['label']
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    train_idx = X_train_text.index
    test_idx = X_test_text.index
    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'

    df[['clean_description', 'label', 'split'] + NUMERIC_FEATS].to_csv(
        PROCESSED_DIR / 'tickets_nb_ready.csv', index=False
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        strip_accents='unicode',
        norm='l2',
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(train_df[NUMERIC_FEATS])
    X_test_num = scaler.transform(test_df[NUMERIC_FEATS])

    X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_num)])
    X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_num)])

    for name, obj in [
        ('label_encoder.pkl', le),
        ('tfidf_vectorizer.pkl', tfidf),
        ('scaler.pkl', scaler),
    ]:
        with open(MODELS_DIR / name, 'wb') as f:
            pickle.dump(obj, f)

    save_npz(FEATURES_DIR / 'X_train.npz', X_train_final)
    save_npz(FEATURES_DIR / 'X_test.npz', X_test_final)
    np.save(FEATURES_DIR / 'y_train.npy', y_train.values)
    np.save(FEATURES_DIR / 'y_test.npy', y_test.values)

    print(f'X_train: {X_train_final.shape}  X_test: {X_test_final.shape}')
    print(f'Artifacts: {MODELS_DIR}')
    print(f'Features:  {FEATURES_DIR}')
    print('\nNext: python src/train.py')


if __name__ == '__main__':
    run()
