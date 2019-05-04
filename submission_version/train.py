import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from joblib import dump, load

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def read_data():
    file = open("final_project_point_cloud.fuse", "r")
    data = [f.split() for f in file]
    file.close()
    for d in data:
        for i in range(4):
            d[i] = float(d[i])
    print("Data loaded.")
    return data

def main():
    data = read_data()
    x, y, z, intensity = np.array(data).T
    label = [0] * len(x)
    z_range = max(z) - min(z)
    median = np.median(z)
    min_z = median - 0.05 * z_range
    max_z = median + 0.05 * z_range
    horizon_data = []
    for i in range(len(data)):
        if min_z < data[i][2] < max_z:
            horizon_data.append(data[i] + [i])
    horizon_x, horizon_y, horizon_z, horizon_intensity, index = np.array(horizon_data).T
    lane_mark = []
    for d in horizon_data:
        if d[3] >= 57 and d[3] <= 79:
            lane_mark.append(d)
    lane_x, lane_y, lane_z, lane_intensity, index = np.array(lane_mark).T
    print("Data preprocessing done.")
    kmeans = KMeans(max_iter = 10000, n_clusters = 4)
    data_for_fit = np.array(lane_mark)[:,:3]
    kmeans.fit(data_for_fit)
    print("Kmeans clustering done.")
    labels = kmeans.labels_
    lane_0, lane_1, lane_2, lane_3 = [], [], [], []
    for i in range(len(labels)):
        if labels[i] == 0:
            lane_0.append(lane_mark[i])
        elif labels[i] == 1:
            lane_1.append(lane_mark[i])
        elif labels[i] == 2:
            lane_2.append(lane_mark[i])
        else:
            lane_3.append(lane_mark[i])
    lane_0_x, lane_0_y, lane_0_z, lane_0_intensity, lane_0_index = np.array(lane_0).T
    lane_1_x, lane_1_y, lane_1_z, lane_1_intensity, lane_1_index= np.array(lane_1).T
    lane_2_x, lane_2_y, lane_2_z, lane_2_intensity, lane_2_index = np.array(lane_2).T
    lane_3_x, lane_3_y, lane_3_z, lane_3_intensity, lane_3_index = np.array(lane_3).T
    lane0RANSAC = linear_model.RANSACRegressor(base_estimator = linear_model.LinearRegression())
    lane0RANSAC.fit(lane_0_x.reshape(-1, 1), lane_0_y.reshape(-1, 1))
    lane_0_inlier_mask = list(lane0RANSAC.inlier_mask_)
    length_0 = len(lane_0)
    i = 0
    while i < length_0:
        if not lane_0_inlier_mask[i]:
            lane_0.pop(i)
            lane_0_inlier_mask.pop(i)
            length_0 -= 1
            i -= 1
        i += 1
    lane1RANSAC = linear_model.RANSACRegressor(base_estimator = linear_model.LinearRegression())
    lane1RANSAC.fit(lane_1_x.reshape(-1, 1), lane_1_y.reshape(-1, 1))
    lane_1_inlier_mask = list(lane1RANSAC.inlier_mask_)
    length_1 = len(lane_1)
    i = 0
    while i < length_1:
        if not lane_1_inlier_mask[i]:
            lane_1.pop(i)
            lane_1_inlier_mask.pop(i)
            length_1 -= 1
            i -= 1
        i += 1
    lane2RANSAC = linear_model.RANSACRegressor(base_estimator = linear_model.LinearRegression())
    lane2RANSAC.fit(lane_2_x.reshape(-1, 1), lane_2_y.reshape(-1, 1))
    lane_2_inlier_mask = list(lane2RANSAC.inlier_mask_)
    length_2 = len(lane_2)
    i = 0
    while i < length_2:
        if not lane_2_inlier_mask[i]:
            lane_2.pop(i)
            lane_2_inlier_mask.pop(i)
            length_2 -= 1
            i -= 1
        i += 1
    lane3RANSAC = linear_model.RANSACRegressor(base_estimator = linear_model.LinearRegression())
    lane3RANSAC.fit(lane_3_x.reshape(-1, 1), lane_3_y.reshape(-1, 1))
    lane_3_inlier_mask = list(lane3RANSAC.inlier_mask_)
    length_3 = len(lane_3)
    i = 0
    while i < length_3:
        if not lane_3_inlier_mask[i]:
            lane_3.pop(i)
            lane_3_inlier_mask.pop(i)
            length_3 -= 1
            i -= 1
        i += 1
    whole_lane_mark =  lane_0 + lane_1 + lane_2 + lane_3
    whole_lane_mark_x, whole_lane_mark_y, whole_lane_mark_z, whole_lane_mark_intensity, whole_lane_mark_index = np.array(whole_lane_mark).T
    print("RANSAC model fitting done.")
    print("Outliers cleaned.")
    for mark in whole_lane_mark:
        label[mark[4]] = 1
    parameters = {'latitude': x, 'longitude':y, 'altitude':z, 'intensity':intensity}
    X = pd.DataFrame(data = parameters)
    labels = {'label':label}
    Y = pd.DataFrame(data = labels)
    train, test = [], []
    for i in range(len(x)):
        if x[i] + y[i] > 56.93189317:
            train.append(i)
        else:
            test.append(i)
    train, test = pd.Int64Index(data = train), pd.Int64Index(data = test)
    x_train, x_test, y_train, y_test = X.iloc[train], X.iloc[test], Y.iloc[train], Y.iloc[test]
    clf = SVC(C = 30, kernel = "rbf")
    print("SVM model created.")
    clf.fit(x_train, y_train)
    print("SVM model trained.")
    y_predict = clf.predict(x_test)
    P = 0
    TP = 0
    FP = 0
    FN = 0
    N = 0
    i = 0
    for index, row in y_test.iterrows():
        if row['label'] == y_predict[i] == 1:
            TP += 1
        elif row['label'] == 1 and y_predict[i] == 0:
            FN += 1
        elif row['label'] == 0 and y_predict[i] == 1:
            FP += 1
        if row['label'] == 1:
            P += 1
        if row['label'] == 0:
            N += 1
        i += 1
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = (2 * precision * recall) / (precision + recall)
    print("Model score f1=" + str(f1))
    dump(clf, 'model.joblib') 
    print("SVM model saved.")
    return
    
if __name__ == '__main__':
    main()