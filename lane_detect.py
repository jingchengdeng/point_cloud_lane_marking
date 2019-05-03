import numpy as np
import matplotlib.pyplot as plt
from open3d import *
from mpl_toolkits import mplot3d


def load_data():
    file = open('final_project_data/final_project_point_cloud.fuse', 'r')

    data = []

    for f in file:
        fs = f.split(" ")
        data.append([float(fs[0]), float(fs[1]), float(fs[2]), float(fs[3][:-1])])

    file.close()
    print('Done! load file')
    return data


def road_serface_detect(data, z):
    min_alt = np.median(z) - 2
    max_alt = np.median(z) + 0.3
    road_data = []
    for d in data:
        if min_alt <= d[2] <= max_alt:
            road_data.append(d)
    print(len(road_data))
    return road_data


def lane_detect(data):
    lane_data = []
    for d in data:
        if d[3] >= 70:
            lane_data.append(d[:-1])
    return lane_data


if __name__ == '__main__':
    data = load_data()
    data = np.array(data)
    x, y, z, intensity = data.T
    road_data = road_serface_detect(data, z)
    road_data = np.array(road_data)
    road_x, road_y, road_z, r_intensity = road_data.T

    fig = plt.figure(figsize=(50, 29.4))
    ax = plt.axes(projection='3d')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim3d(200, 234)
    ax.plot(x, y, z, ',')
    plt.savefig('test99.png')


    fig = plt.figure(figsize=(50, 29.4))
    ax = plt.axes(projection='3d')
    rangex = max(road_x) - min(road_x)
    rangey = max(road_y) - min(road_y)
    ax.set_xlim(min(road_x), max(road_x) - 0.1*rangex)
    ax.set_ylim(min(road_y)+0.4*rangey, max(road_y) - 0.8*rangey)
    ax.set_zlim3d(200, 234)
    ax.plot(road_x, road_y, road_z, ',')
    plt.savefig('test2.png')

    lane_data = lane_detect(data)
    lane_data = np.array(lane_data)
    lane_x, lane_y, lane_z = lane_data.T

    fig = plt.figure(figsize=(50, 29.4))
    ax = fig.add_subplot(111)
    ax.plot(road_x, road_y, ',')
    ax.plot(lane_x, lane_y, ',', color='red')
    plt.savefig('output2.png')

