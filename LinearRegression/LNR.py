import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Generate 'random' data
N = 1000
np.random.seed(0)

X = (2.5 * np.random.randn(N) + 1.5)   # Array of N values with mean = 1.5, stddev = 2.5
res = 0.5 * np.random.randn(N)       # Generate N residual terms
y = 2 + 0.3 * X + res                  # Actual values of Y

X = X.reshape(N, 1)

# Split data and label into 2 sets: train set, test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

reg = LinearRegression().fit(X_train, y_train)

# reg.score(X_test, y_test)
coef = reg.coef_
intercept = reg.intercept_

y_pred = reg.predict(X_test)

########### Draw ###################
"""
    TODO:
        Draw points (X_train, y_train) (color: blue)
        Draw points (X_test, y_test) (color: red)
        Draw the line based on coef and intercept (color: green)
"""