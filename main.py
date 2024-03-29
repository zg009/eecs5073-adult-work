from ucimlrepo import fetch_ucirepo 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, r2_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
from sklearn.neural_network import MLPClassifier


# Encodes catergory classes with One Hot Encoder
def category_encode(features: pd.DataFrame):

    # Select the category classes from features
    cat_classes = features.select_dtypes(include=[object])
    
    # Setup Hot Encoder
    enc = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    enc.fit(cat_classes)
    
    encoded_cat_data = enc.fit_transform(cat_classes)
    new_df = pd.DataFrame(encoded_cat_data, )

    return new_df

# Encode numerical classes using mean value to binary attribute
def numeric_binary_transform(numeric_columns: pd.DataFrame):
    encoded_numeric_col = numeric_columns
    for col in numeric_columns:
        mean = numeric_columns[col].mean()
        encoded_numeric_col[col] = np.where(numeric_columns[col] > mean, 0, 1)

    return encoded_numeric_col


# factor this out
def preprocessing(df: pd.DataFrame):

    df = df.dropna() # remove all rows with garbage
    assert(len(df.columns[df.isnull().any()].tolist()) == 0) # assert no more na values

    # Removes periods in income    
    target = df['income'] # Y value
    target = target.replace(r'\.$', value="", regex=True)

    # X values. Removes income from data
    features = df[df.columns.difference(['income'])] 

    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['int64'])
    # Transform to binary attribute
    encoded_numeric_df = numeric_binary_transform(numeric_columns)

    # Encoding category classes
    encoded_cat_df = category_encode(features)

    # Combine encoded numeric classes and encoded catergory classes
    combined_dataframe = pd.concat([encoded_numeric_df, encoded_cat_df], axis=1)

    return (combined_dataframe, target)

def calc_performance(y_test, y_hat):

    # Get accuracy score
    accuracy = accuracy_score(y_test, y_hat)

    # Generate confusion matrix to get TP and FP
    tree_cm = confusion_matrix(y_test, y_hat)
    TP = tree_cm[1][1]
    FN = tree_cm[1][0]
    FP = tree_cm[0][1]

    # Calculate precision
    precision = float(TP / (TP + FP))
    # Calculate recall
    recall = float(TP / (TP + FN))
            
    # Calculate F1
    f1 = float((2 * precision * recall) / (precision + recall))

    print(f'Accuracy: {accuracy}\n TP: {TP}\n FP: {FP}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}\n')

    return accuracy, TP, FP, precision, recall, f1

# Tree classifier. Returns accuracy, TP rate, FP rate, precision, recall, and F1
def decision_tree(X_train, X_test, y_train, y_test, dummified):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    # this stinks, probably needs tweaking
    tree_rules = export_text(decision_tree, feature_names=list(dummified.columns))
    print(tree_rules)
    # predicted y
    y_hat = decision_tree.predict(X_test)   

    # Get performance statistics
    statistics = calc_performance(y_test, y_hat)

    return statistics

def naive_bayes(model, X_train, X_test, y_train, y_test):
    # gnb model
    gnb = model
    gnb.fit(X_train, y_train)
    gnb_hat = gnb.predict(X_test)
    # Get performance statistics
    statistics = calc_performance(y_test, gnb_hat)
    return statistics


def k_means(clusters, X_train, algorithm="lloyd") -> KMeans:
    kmeans = KMeans(n_clusters=clusters, algorithm=algorithm)
    kmeans.fit(X_train)
    return kmeans

def all_kmeans(X_train):
    kmeans = k_means(3, X_train)
    print(kmeans.cluster_centers_) # shape is [3, 104]

    kmeans = k_means(5, X_train)
    print(kmeans.cluster_centers_) # shape is [5, 104]

    kmeans = k_means(10, X_train)
    # print(kmeans.cluster_centers_)

    # elkan algorithm
    kmeans = k_means(3, X_train, algorithm="elkan")
    print(kmeans.cluster_centers_)
    
    kmeans = k_means(5, X_train, algorithm="elkan")
    print(kmeans.cluster_centers_)
    
    kmeans = k_means(10, X_train, algorithm="elkan")
    print(kmeans.cluster_centers_)
    
def svm_classifier(model, X_train, X_test, y_train, y_test):
    svm = model
    svm.fit(X_train, y_train)
    y_hat = svm.predict(X_test)
    statistics = calc_performance(y_test, y_hat)
    return statistics

# CNN
def mlp(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    statistics = calc_performance(y_test, y_hat)
    return statistics

# fetch dataset 
adult = fetch_ucirepo(id=2) 

df = pd.DataFrame(adult.data.original)

# remove all the '?'
df = df[~df.isin(['?']).any(axis=1)]
df = df.dropna() # remove all rows with missing values
assert(len(df.columns[df.isnull().any()].tolist()) == 0) # assert no more na values

# find the numeric columns: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
numeric_columns = df.select_dtypes(include=['int64'])

# actually split it. Remove target column (income) from data.
features, target = preprocessing(df)
# print("df:\n", df)
# print("Features:\n", features.dtypes)
# print("Target:\n", target)

#Split training and test data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5)

# Decision Tree
print("Performing decision tree classifer:")
statistics = decision_tree(X_train, X_test, y_train, y_test, features)

# Naive Bayes
# print("Performing gaussian naive bayes classifer:")
statistics = naive_bayes(GaussianNB(), X_train, X_test, y_train, y_test)

# # probably the same because k is 2 ~> Categorical reduces to bernoulli...
print("Performing bernoulli naive bayes classifer:")
statistics = naive_bayes(BernoulliNB(), X_train, X_test, y_train, y_test)

print("Performing categorical naive bayes classifer:")
statistics = naive_bayes(CategoricalNB(min_categories=features.nunique()), X_train, X_test, y_train, y_test)

# best performing model in naive bayes
print("Performing multinomial naive bayes classifer:")
statistics = naive_bayes(MultinomialNB(), X_train, X_test, y_train, y_test)

print("Performing complement naive bayes classifer:")
statistics = naive_bayes(ComplementNB(), X_train, X_test, y_train, y_test)

# SVM
q = 42
# linear seems to be the best performed, with rbf second
print("Performing SVM SVC classifier with linear kernel:")
statistics = svm_classifier(svm.SVC(kernel='linear', random_state=q), X_train, X_test, y_train, y_test)

print("Performing SVM SVC classifier with polynomial kernel:")
statistics = svm_classifier(svm.SVC(kernel='poly', random_state=q), X_train, X_test, y_train, y_test)

print("Performing SVM SVC classifier with sigmoid kernel:")
statistics = svm_classifier(svm.SVC(kernel='sigmoid', random_state=q), X_train, X_test, y_train, y_test)

print("Performing SVM SVC classifier with rbf kernel(default):")
statistics = svm_classifier(svm.SVC(kernel='rbf', random_state=q), X_train, X_test, y_train, y_test)

# do not set nu > 0.44 or < 0.3. Optimal seems to be somewhere between 0.37 and 0.42, so settled on 0.4
# best nusvc but worse than regular svc
print("Performing SVM NuSVC classifier with linear kernel:")
statistics = svm_classifier(svm.NuSVC(nu=0.4, kernel='linear', random_state=q), X_train, X_test, y_train, y_test)

print("Performing SVM NuSVC classifier with polynomial kernel:")
statistics = svm_classifier(svm.NuSVC(nu=0.4, kernel='poly', random_state=q), X_train, X_test, y_train, y_test)

print("Performing SVM NuSVC classifier with sigmoid kernel:")
statistics = svm_classifier(svm.NuSVC(nu=0.4, kernel='sigmoid', random_state=q), X_train, X_test, y_train, y_test)

print("Performing SVM NuSVC classifier with rbf kernel(default):")
statistics = svm_classifier(svm.NuSVC(nu=0.4, kernel='rbf', random_state=q), X_train, X_test, y_train, y_test)

# idk how to quantify kmeans, maybe find the rows that were grouped together?
# Kmeans
print("Performing Kmeans classifer:")
all_kmeans(X_train)

# # MLP
print('Performing MLP neural network:')
# model = MLPClassifier(activation='identity', solver='adam')
# mlp(model, X_train, X_test, y_train, y_test)
print('activation-relu solver-adam iter-600:')
model = MLPClassifier(activation='relu', solver='adam', max_iter=600)
mlp(model, X_train, X_test, y_train, y_test)
print('activation-tanh solver-adam iter-600:')
model = MLPClassifier(activation='tanh', solver='adam', max_iter=600)
mlp(model, X_train, X_test, y_train, y_test)
print('activation-logistic solver-adam iter-600:')
model = MLPClassifier(activation='logistic', solver='adam', max_iter=600)
mlp(model, X_train, X_test, y_train, y_test)

# model = MLPClassifier(activation='identity', solver='lbfgs')
# mlp(model, X_train, X_test, y_train, y_test)
print('activation-relu solver-lbfgs iter-600:')
model = MLPClassifier(activation='relu', solver='lbfgs', max_iter=600)
mlp(model, X_train, X_test, y_train, y_test)
print('activation-tanh solver-lbfgs iter-600:')
model = MLPClassifier(activation='tanh', solver='lbfgs', max_iter=600)
mlp(model, X_train, X_test, y_train, y_test)
print('activation-logistic solver-lbfgs iter-600:')
model = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=600)
mlp(model, X_train, X_test, y_train, y_test)

# model = MLPClassifier(activation='identity', solver='sgd')
# mlp(model, X_train, X_test, y_train, y_test)
print('activation-relu solver-sgd iter-600 adaptive learning:')
model = MLPClassifier(activation='relu', solver='sgd', max_iter=600, learning_rate='adaptive')
mlp(model, X_train, X_test, y_train, y_test)
print('activation-tanh solver-sgd iter-600 adaptive learning:')
model = MLPClassifier(activation='tanh', solver='sgd', max_iter=600, learning_rate='adaptive')
mlp(model, X_train, X_test, y_train, y_test)

print('activation-logistic solver-sgd iter-600 adaptive learning:')
model = MLPClassifier(activation='logistic', solver='sgd', max_iter=600, learning_rate='adaptive')
mlp(model, X_train, X_test, y_train, y_test)


