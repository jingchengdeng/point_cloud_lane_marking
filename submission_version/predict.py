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
    clf = load('model.joblib')
    print("Model loaded.")
    lat, lon, alt, intensity = np.array(data).T
    parameters = {'latitude': lat, 'longitude': lon, 'altitude': alt, 'intensity': intensity}
    X = pd.DataFrame(data = parameters)
    print("Data preprocessing done.")
    y_predict = clf.predict(X)
    print("Predict done.")
    fig = plt.figure(figsize = (50, 30))
    plt.plot(X[y_predict == False]['latitude'], X[y_predict == False]['longitude'], ',', color = "yellow")
    plt.plot(X[y_predict == True]['latitude'], X[y_predict == True]['longitude'], ',', color = "red")
    plt.savefig('result2D.png')
    print("2D result image saved.")
    fig = plt.figure(figsize = (50, 30))
    ax = plt.axes(projection = '3d')
    ax.plot3D(X[y_predict == False]['latitude'], X[y_predict == False]['longitude'], X[y_predict == False]['altitude'], ',', color = "yellow")
    ax.plot3D(X[y_predict == True]['latitude'], X[y_predict == True]['longitude'], X[y_predict == True]['altitude'], ',', color = "red")
    plt.savefig('result3D.png')
    print("3D result image saved.")
    return

if __name__ == '__main__':
    main()