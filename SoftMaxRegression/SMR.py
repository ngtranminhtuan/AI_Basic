import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

# Generate random points surround 3 centroids: [2,2], [8,3], [3,6]
means = [[2, 2], [5, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

N = 7500
C = 3

X_0 = np.random.multivariate_normal(means[0], cov, N)
X_1 = np.random.multivariate_normal(means[1], cov, N)
X_2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X_0, X_1, X_2), axis=0).T

# extended data: append a colum of number 1 to the left
X = np.concatenate((np.ones((1, C * N)), X), axis=0).T
y = np.array([[0] * N, [1] * N, [2] * N]).reshape(C * N, )

# split train set and test set from data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

##########################################################################################################


# For a multi_class problem, if multi_class is set to
# be “multinomial” the softmax function is used to
# find the predicted probability of each class
logreg = linear_model.LogisticRegression(C=1e5,
                                         solver='lbfgs', multi_class='multinomial')

# train
logreg.fit(X_train, y_train)

# test
y_pred = logreg.predict(X_test)

# evaluate
print("Accuracy: %.2f %%" % (100 * accuracy_score(y_test, y_pred.tolist())))

#################### DRAW ############################################

"""
    TODO:
        Graph1: Visualize X_test colored by corresponding label in y_test
        Graph2: Visualize X_test colored by corresponding label in y_pred
"""


def draw_multiple_points(x_number_list, y_number_list, color, pointWeight):
    plt.scatter(x_number_list, y_number_list, s=pointWeight, c=color)


# X_class0 = X_test[y_test == 0, :]
# X_class1 = X_test[y_test == 1, :]
# X_class2 = X_test[y_test == 2, :]
# draw_multiple_points(X_class0[:, 1], X_class0[:, 2], 'red', 10)
# draw_multiple_points(X_class1[:, 1], X_class1[:, 2], 'green', 10)
# draw_multiple_points(X_class2[:, 1], X_class2[:, 2], 'blue', 10)

X_class0 = X_test[y_pred == 0, :]
X_class1 = X_test[y_pred == 1, :]
X_class2 = X_test[y_pred == 2, :]

draw_multiple_points(X_class0[:, 1], X_class0[:, 2], 'red', 10)
draw_multiple_points(X_class1[:, 1], X_class1[:, 2], 'green', 10)
draw_multiple_points(X_class2[:, 1], X_class2[:, 2], 'blue', 10)

plt.show()

