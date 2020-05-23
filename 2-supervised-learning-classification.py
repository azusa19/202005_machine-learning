import mglearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# C = 0.001
# logreg = LogisticRegression(C=C, max_iter=100000).fit(X_train, y_train)
# print(logreg.score(X_train, y_train))
# print(logreg.score(X_test, y_test))

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, max_iter=100000, penalty="l1", solver='liblinear').fit(X_train, y_train)
    print("Training score with C={:.3f} : {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend(loc=3)
plt.show()
