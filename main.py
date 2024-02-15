from ucimlrepo import fetch_ucirepo 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
df = pd.DataFrame(adult.data.original)

df = df.dropna() #remove all rows with garbage
assert(len(df.columns[df.isnull().any()].tolist()) == 0) # assert no more na values
# print(df.dtypes) # inspect

target = df['income'] # Y value
features = df[df.columns.difference(['income'])] # X values

# select the category classes from features
cat_classes = features.select_dtypes(include=[object])

# do i need a labelencoder or can i just raw it?
# le = LabelEncoder()
# features_2 = features.apply(le.fit_transform)
# print(features_2.head())
enc = OneHotEncoder()
enc.fit(cat_classes)
# print(enc.categories_)
category_names = [category for arr in enc.categories_ for category in arr]
print(category_names)
onehotlabels = enc.transform(cat_classes).toarray()
new_df = pd.DataFrame(onehotlabels, )
# print(new_df.head)
# new_target = pd.DataFrame(onehotlabels)
# print(new_target.columns)
# tts for later
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)