import joblib
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_digits()
X, y = data.data, data.target

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=None,
    min_samples_split=2,
    random_state=42)
model.fit(X, y)

joblib.dump(model, 'model.joblib')



# print(data.images[3])
# plt.imshow(data.images[3], cmap='gray')
# plt.show()