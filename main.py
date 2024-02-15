from ucimlrepo import fetch_ucirepo 
import pandas as pd  
from sklearn.model_selection import train_test_split

# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
df = pd.DataFrame(adult.data.original)
  
df = df.dropna() #remove all rows with garbage
assert(len(df.columns[df.isnull().any()].tolist()) == 0)
print(df.dtypes)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)