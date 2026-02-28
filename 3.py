from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

y_pred = gnb_classifier.predict(X_test)
y_pred_prob = gnb_classifier.predict_proba(X_test)
print(f"Dokładność GNB: {(accuracy_score(y_test, y_pred)*100):.2f}%")
print(y_test)
print(y_pred)
print(y_pred_prob)