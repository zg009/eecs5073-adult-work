from ucimlrepo import fetch_ucirepo 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

def sklearn_encoder(features: pd.DataFrame):
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
    
# factor this out
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna() # remove all rows with garbage
    assert(len(df.columns[df.isnull().any()].tolist()) == 0) # assert no more na values
    target = df['income'] # Y value
    # whoever included target column values with a period in them
    # needs to be in prison
    target = target.replace(r'\.$', value="", regex=True)
    features = df[df.columns.difference(['income'])] # X values
    # probably have to fix this encoding system
    dummified = pd.get_dummies(features)
    # print(dummified.columns)
    return (dummified, target)

# fetch dataset 
adult = fetch_ucirepo(id=2) 

df = pd.DataFrame(adult.data.original)

dummified, target = preprocessing(df)

# tts for later
X_train, X_test, y_train, y_test = train_test_split(dummified, target, test_size=0.33, random_state=42)
# decision tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
# this stinks, probably needs tweaking
tree_rules = export_text(decision_tree, feature_names=list(dummified.columns))
# print(tree_rules)

# predicted y
y_hat = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print(accuracy)

# gnb model
# this should probably be categorical NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_hat = gnb.predict(X_test)
accuracy = accuracy_score(y_test, gnb_hat)
print(accuracy)

# time for step 2
print(df.head())