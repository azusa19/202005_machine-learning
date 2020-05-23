import mglearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

from sklearn.linear_model import Ridge
alpha = 0.1
ridge = Ridge(alpha=alpha).fit(X_train, y_train)
plt.plot(ridge.coef_, 'o', label='Ridge alpha=1')


from sklearn.linear_model import Lasso

for a, dot in zip([1, 0.01, 0.0001], ['s', '^', 'v']):
    alpha = a
    lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_train, y_train)
    # print(lasso.score(X_train, y_train))
    # print(lasso.score(X_test, y_test))
    # print(np.sum(lasso.coef_ != 0))
    plt.plot(lasso.coef_, dot, label="Lasso alpha={}".format(a))

plt.legend(ncol=2, loc=(0, 1.02))
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()
