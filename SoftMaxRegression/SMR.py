import numpy as np 
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.metrics import accuracy_score

# Generate random points surround 3 centroids: [2,2], [8,3], [3,6]
means = [[2, 2], [5, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

N = 7500
C = 3

X_0 = np.random.multivariate_normal(means[0], cov, N)
X_1 = np.random.multivariate_normal(means[1], cov, N)
X_2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X_0, X_1, X_2), axis = 0).T


# extended data: append a colum of number 1 to the left
X = np.concatenate((np.ones((1, C*N)), X), axis = 0).T
y = np.array([[0]*N,[1]*N,[2]*N]).reshape(C*N,)

# split train set and test set from data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

##########################################################################################################


# For a multi_class problem, if multi_class is set to 
# be “multinomial” the softmax function is used to 
# find the predicted probability of each class
logreg = linear_model.LogisticRegression(C=1e5, 
        solver = 'lbfgs', multi_class = 'multinomial')

# train
logreg.fit(X_train, y_train)

# test
y_pred = logreg.predict(X_test)

#evaluate
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred.tolist())))

#################### DRAW ############################################

"""
    TODO:
        Graph1: Visualize X_test colored by corresponding label in y_test
        Graph2: Visualize X_test colored by corresponding label in y_pred
"""