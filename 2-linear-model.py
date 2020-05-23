import mglearn
import matplotlib.pyplot as plt
import numpy as np

# mglearn.plots.plot_linear_regression_wave()
# plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

from sklearn.linear_model import Ridge
alpha = 1
ridge = Ridge(alpha=alpha).fit(X_train, y_train)
plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
# print(ridge.score(X_train, y_train))
# print(ridge.score(X_test, y_test))

alpha = 10
ridge = Ridge(alpha=alpha).fit(X_train, y_train)
plt.plot(ridge.coef_, '^', label='Ridge alpha=10')

alpha = 0.1
ridge = Ridge(alpha=alpha).fit(X_train, y_train)
plt.plot(ridge.coef_, 'v', label='Ridge alpha=0.1')

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()
