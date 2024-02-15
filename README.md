## Setup

```
python -m venv .venv
pip install pandas uciml_repo
pip install -U scikit-learn
.venv\Scripts\Activate (whatever based on terminal)
python main.py
```

## adult.data Shape description

adult.data contains
  - ids       - None
  - features  - variables (X)
  - targets   - y target
  - original  - original DataFrame
  - headers   - DataFrame.Index (column headers)
