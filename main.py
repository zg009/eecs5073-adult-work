from ucimlrepo import fetch_ucirepo 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

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
def preprocessing(df: pd.DataFrame):
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

def decision_tree(X_train, X_test, y_train, y_test, dummified):
    # decision tree
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    # this stinks, probably needs tweaking
    tree_rules = export_text(decision_tree, feature_names=list(dummified.columns))
    # print(tree_rules)
    # predicted y
    y_hat = decision_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)
    return accuracy

def naive_bayes(X_train, X_test, y_train, y_test):
    # gnb model
    # this should probably be categorical NB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_hat = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, gnb_hat)
    # print(accuracy)
    return accuracy

from sklearn.cluster import KMeans
def k_means(clusters, X_train, algorithm="lloyd") -> KMeans:
    kmeans = KMeans(n_clusters=clusters, random_state=42, algorithm=algorithm)
    kmeans.fit(X_train)
    return kmeans

def all_kmeans():
    kmeans = k_means(3, X_train)
    print(kmeans.cluster_centers)
    kmeans = k_means(5, X_train)
    print(kmeans.cluster_centers)
    kmeans = k_means(10, X_train)
    print(kmeans.cluster_centers)
    # elkan algorithm
    kmeans = k_means(3, X_train, algorithm="elkan")
    print(kmeans.cluster_centers)
    kmeans = k_means(5, X_train, algorithm="elkan")
    print(kmeans.cluster_centers)
    kmeans = k_means(10, X_train, algorithm="elkan")
    print(kmeans.cluster_centers)    

def svm_classifier(X_train, X_test, y_train, y_test):
    from sklearn import svm
    svm = svm.SVC()
    svm.fit(X_train, y_train)
    y_hat = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)
    # print(accuracy)
    return accuracy
    

# fetch dataset 
adult = fetch_ucirepo(id=2) 

df = pd.DataFrame(adult.data.original)

# dummified, target = preprocessing(df)

# tts for later
# X_train, X_test, y_train, y_test = train_test_split(dummified, target, test_size=0.33, random_state=42)

# dt_acc = decision_tree(X_train, X_test, y_train, y_test, dummified)
# nb_acc = naive_bayes(X_train, X_test, y_train, y_test)

# time for step 2
# remove all the '?'
# think this removes all the '?'
df = df[~df.isin(['?']).any(axis=1)]
# have to call his beforehand in case pandas decides to do more stupid magic
# i love this terrible language and librarieS!!!
df = df.dropna() # remove all rows with garbage
assert(len(df.columns[df.isnull().any()].tolist()) == 0) # assert no more na values
# find the numeric columns
numeric_columns = df.select_dtypes(include=['int64'])

# cycle through them and get teh mean then split into 0 or 1 based on greater or less
# then reassign to original df
for col in numeric_columns:
    # garbage python library means i have to import numpy to do this since it doesnt natively work
    # with series objects! :D
    mean = numeric_columns[col].mean()
    numeric_columns[col] = np.where(numeric_columns[col] > mean, 0, 1)
    df[col] = numeric_columns[col]

# actually split it
features, target = preprocessing(df)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

    
from sklearn.neural_network import MLPClassifier
# iterations dont matter here, leaving it at 400 and fuck off
mlp = MLPClassifier(random_state=42, max_iter=400)
mlp.fit(X_train, y_train)
y_hat = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print(accuracy)
