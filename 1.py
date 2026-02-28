from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(
    criterion='entropy', # 'entropy'
    max_depth=None,
    min_samples_split=2,
    random_state=42)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred))

# plt.figure(figsize=(15, 10))
# tree.plot_tree(dt_classifier, feature_names=data.feature_names,
#                class_names=data.target_names, filled=True)
# plt.show()