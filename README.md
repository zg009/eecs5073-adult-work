## Setup

```
python -m venv .venv
.venv\Scripts\Activate (whatever based on terminal)
pip install -r requirements.txt
python main.py
```

## adult.data Shape description

adult.data contains
  - ids       - None
  - features  - variables (X)
  - targets   - y target
  - original  - original DataFrame
  - headers   - DataFrame.Index (column headers)

## Results

output.txt        contains all output using python main.py > output.txt
decision_tree.txt contains dendrogram of decision tree classifier