"""
Train Multinomial Naive Bayes on feature matrices and save the model.

Usage:
    python src/train.py
"""

import logging
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings('ignore')


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_FEAT_DIR = PROJ_ROOT / 'data' / 'features'
MODELS_DIR = PROJ_ROOT / 'models'
FIGURES_DIR = PROJ_ROOT / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = PROJ_ROOT / 'training.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),  # Add this to save logs to a file
        logging.StreamHandler()         # Keep this to see logs in the terminal
    ],
)
logger = logging.getLogger(__name__)


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path.name}. Run: python src/feature_engineering.py'
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_features_and_artifacts():
    logger.info('Loading feature matrices...')
    X_train = load_npz(DATA_FEAT_DIR / 'X_train.npz')
    X_test = load_npz(DATA_FEAT_DIR / 'X_test.npz')
    y_train = np.load(DATA_FEAT_DIR / 'y_train.npy')
    y_test = np.load(DATA_FEAT_DIR / 'y_test.npy')

    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'X_test shape:  {X_test.shape}')

    label_encoder = _load_pickle(MODELS_DIR / 'label_encoder.pkl')
    class_names = label_encoder.classes_
    logger.info(f'Classes: {list(class_names)}')

    return X_train, X_test, y_train, y_test, class_names


def train_model_with_tuning(X_train, y_train):
    logger.info('\n' + '=' * 70)
    logger.info('HYPERPARAMETER TUNING')
    logger.info('=' * 70)

    param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
    grid_search = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    logger.info(f'\nBest alpha: {grid_search.best_params_["alpha"]}')
    logger.info(f'Best CV F1-score (weighted): {grid_search.best_score_:.4f}')

    cv_results = pd.DataFrame(grid_search.cv_results_)[
        ['param_alpha', 'mean_test_score', 'std_test_score']
    ]
    cv_results.columns = ['Alpha', 'Mean CV F1-Score', 'Std CV F1-Score']
    logger.info(f'\n{cv_results.to_string(index=False)}\n')

    return grid_search.best_estimator_


def evaluate_model(model, X_train, y_train, X_test, y_test, class_names):
    logger.info('\n' + '=' * 70)
    logger.info('MODEL EVALUATION')
    logger.info('=' * 70)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    logger.info(f'\nTrain Accuracy: {train_acc:.4f}')
    logger.info(f'Test Accuracy:  {test_acc:.4f}')
    logger.info('\n' + classification_report(y_test, y_pred_test, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred_test)
    for i, name in enumerate(class_names):
        total = cm[i, :].sum()
        correct = cm[i, i]
        logger.info(f'{name}: {correct}/{total} ({100 * correct / total:.1f}%)')

    return cm, test_acc


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out = FIGURES_DIR / 'confusion_matrix.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'Confusion matrix saved: {out}')


def save_model(model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / 'naive_bayes_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Model saved: {path}')


def main():
    logger.info('\n' + '=' * 70)
    logger.info('MULTINOMIAL NAIVE BAYES TRAINING')
    logger.info('=' * 70)

    try:
        X_train, X_test, y_train, y_test, class_names = load_features_and_artifacts()
        model = train_model_with_tuning(X_train, y_train)
        cm, test_acc = evaluate_model(model, X_train, y_train, X_test, y_test, class_names)
        plot_confusion_matrix(cm, class_names)
        save_model(model)
        logger.info(f'\nFinal test accuracy: {test_acc:.4f}')
    except Exception as e:
        logger.error(f'Training failed: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
