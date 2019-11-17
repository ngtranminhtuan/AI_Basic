import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans


# Draw multiple points.
def draw_multiple_points(x_number_list, y_number_list, color, pointWeight):
    # x axis value list.
    # x_number_list = [1, 4, 9, 16, 25]

    # y axis value list.
    # y_number_list = [1, 2, 3, 4, 5]

    # Draw point based on above x, y axis values.
    plt.scatter(x_number_list, y_number_list, s=pointWeight, c=color)

    # # Set chart title.
    # plt.title("Extract Number Root ")
    #
    # # Set x, y label text.
    # plt.xlabel("Number")
    # plt.ylabel("Extract Root of Number")
    # plt.show()


# K-MEANS

# Đây là tâm của các phân phối điểm
means = [[2, 2], [8, 8], [8, 2]]

# Đây là độ phân tản
cov = [[1, 0], [0, 1]]

#  Số điểm
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

original_label = np.asarray([0] * N + [1] * N + [2] * N).T

################################################
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)

center = kmeans.cluster_centers_

print("#########################")
# shape tương tự như hàm count hya lenght
print(pred_label.shape)
X_class0 = X[pred_label == 0, :]
X_class1 = X[pred_label == 1, :]
X_class2 = X[pred_label == 2, :]

# In ra tất cả những phần tử với cột 0 và cột 1
# Trong python hỗ trợ cấu trúc bên dưới
draw_multiple_points(X_class0[:, 0], X_class0[:, 1], 'red', 10)
draw_multiple_points(X_class1[:, 0], X_class1[:, 1], 'green', 10)
draw_multiple_points(X_class2[:, 0], X_class2[:, 1], 'blue', 10)
draw_multiple_points(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 'black', 500)
plt.show()
