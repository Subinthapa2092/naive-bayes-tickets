import pandas as pd
import numpy as np
import os
import pickle
from scipy.sparse import hstack, save_npz, load_npz

BASE_DIR      = r'C:\Users\bibek\naive-bayes-tickets'
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tickets_nb_ready.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
SRC_DIR     = os.path.join(BASE_DIR, 'src')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'data', 'features')

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(PROCESSED_PATH)

train_df = df[df['split'] == 'train'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)

print(f'Train: {len(train_df)}  |  Test: {len(test_df)}')

with open(os.path.join(SRC_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

with open(os.path.join(SRC_DIR, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)
    
X_train_tfidf = tfidf.transform(train_df['clean_description'])
X_test_tfidf  = tfidf.transform(test_df['clean_description'])

NUMERIC_FEATS = [
    'char_count', 'word_count_raw', 'avg_word_len', 'sentence_count',
    'exclamation_count', 'question_count', 'upper_ratio',
    'word_count_clean', 'unique_word_ratio',
    'flag_refund', 'flag_technical', 'flag_cancellation',
    'flag_product', 'flag_billing'
]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_num = scaler.fit_transform(train_df[NUMERIC_FEATS])
X_test_num  = scaler.transform(test_df[NUMERIC_FEATS])

with open(os.path.join(SRC_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
    



from scipy.sparse import csr_matrix

X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_num)])
X_test_final  = hstack([X_test_tfidf,  csr_matrix(X_test_num)])

y_train = train_df['label'].values
y_test  = test_df['label'].values

print(f'X_train : {X_train_final.shape}')
print(f'X_test  : {X_test_final.shape}')
print(f'y_train : {y_train.shape}')
print(f'y_test  : {y_test.shape}')

save_npz(os.path.join(OUTPUT_DIR, 'X_train.npz'), X_train_final)
save_npz(os.path.join(OUTPUT_DIR, 'X_test.npz'),  X_test_final)

np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y_test)

print('Saved to', OUTPUT_DIR)
print(os.listdir(OUTPUT_DIR))