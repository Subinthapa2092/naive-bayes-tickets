"""Inference pipeline for support ticket classification."""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import nltk
from scipy.sparse import csr_matrix, hstack
from sklearn.utils.validation import check_is_fitted

_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from preprocessing import extract_numeric_features, preprocess_text

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path.name}. Run: python src/feature_engineering.py'
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


class TicketClassifier:
    """Load trained artifacts and classify support tickets."""

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODELS_DIR
        self._load_artifacts()

    def _load_artifacts(self):
        self.model = _load_pickle(self.model_dir / 'naive_bayes_model.pkl')
        self.label_encoder = _load_pickle(self.model_dir / 'label_encoder.pkl')
        self.vectorizer = _load_pickle(self.model_dir / 'tfidf_vectorizer.pkl')
        self.scaler = _load_pickle(self.model_dir / 'scaler.pkl')
        check_is_fitted(self.scaler)
        self.class_names = self.label_encoder.classes_

    def predict(self, raw_description: str) -> Dict[str, Any]:
        cleaned = preprocess_text(raw_description)
        X_tfidf = self.vectorizer.transform([cleaned])
        X_numeric = extract_numeric_features(raw_description, cleaned)
        X_numeric_scaled = self.scaler.transform(X_numeric)
        X_combined = hstack([X_tfidf, csr_matrix(X_numeric_scaled)])

        class_id = self.model.predict(X_combined)[0]
        class_proba = self.model.predict_proba(X_combined)[0]
        predicted_class = self.label_encoder.inverse_transform([class_id])[0]

        return {
            'predicted_class': predicted_class,
            'class_id': int(class_id),
            'confidence': float(class_proba[class_id]),
            'probabilities': {
                self.class_names[i]: float(class_proba[i])
                for i in range(len(self.class_names))
            },
            'cleaned_description': cleaned,
        }

    def predict_batch(self, descriptions: list) -> list:
        return [self.predict(text) for text in descriptions]

    def get_top_predictions(self, raw_description: str, top_k: int = 3) -> list:
        result = self.predict(raw_description)
        return sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:top_k]


def load_classifier(model_dir: str = None) -> TicketClassifier:
    return TicketClassifier(model_dir)
