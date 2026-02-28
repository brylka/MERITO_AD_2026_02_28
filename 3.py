from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

y_pred = gnb_classifier.predict(X_test)
y_pred_prob = gnb_classifier.predict_proba(X_test)
print(f"Dokładność GNB: {(accuracy_score(y_test, y_pred)*100):.2f}%")
# print(y_test)
# print(y_pred)
# print(y_pred_prob)

print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{data.target_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Losowy klasyfikator')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Bayes')
plt.title('Krzywa ROC - Naive Bayes')
plt.legend()
plt.grid(True)
plt.show()