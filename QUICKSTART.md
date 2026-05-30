# Quick Start

## 1. Install

```powershell
cd naive-bayes-tickets
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Build features and train

```powershell
python src/feature_engineering.py
python src/train.py
```

## 3. Predict

```powershell
python predict.py "I need a refund"
python predict.py "Cancel my subscription" --format detailed
python predict.py --file tickets.txt
```

## 4. Python API

```python
from src.model import TicketClassifier

clf = TicketClassifier()
print(clf.predict("Billing question about my invoice"))
```

## 5. Outputs

| Output | Location |
|--------|----------|
| Preprocessing artifacts | `models/*.pkl` |
| Feature matrices | `data/features/` |
| Confusion matrix | `reports/figures/confusion_matrix.png` |

## Retrain

After changing `data/raw/customer_support_tickets.csv`:

```powershell
python src/feature_engineering.py
python src/train.py
```

## Troubleshooting

| Issue | Action |
|-------|--------|
| `Missing *.pkl` | Run `python src/feature_engineering.py` then `python src/train.py` |
| NLTK errors | First run downloads data automatically |
| Low accuracy (~20%) | See README — dataset labels are not predictable from text |
