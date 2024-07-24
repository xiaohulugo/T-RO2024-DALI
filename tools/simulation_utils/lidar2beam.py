import glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def fit_beam(points_list, center):
    #np.savetxt('/home/lxh/Documents/LidarBeam/npy/kitti0.txt', points_list[0][:,1:4], delimiter=',', fmt='%.4f')   # X is an array

    points_all = np.concatenate(points_list, axis=0)
    points_all = points_all[:,1:4]
    points_all[:,0] -= center[0]
    points_all[:,1] -= center[1]
    points_all[:,2] -= center[2]
    l_xy = (points_all[:,0]**2 + points_all[:,1]**2)**(0.5)
    l_z = points_all[:,2]
    vertical_angle = np.arctan2(l_z, l_xy)  
    horizontal_angle = np.arctan2(points_all[:,0], points_all[:,1])  
    #hist = np.histogram(vertical_angle, bins=150)   
    hist = np.histogram(horizontal_angle, bins=720)   
    freq = hist[0]
    peaks, _ = find_peaks(freq)
    plt.plot(freq)
    plt.plot(peaks, freq[peaks], "x")    
    plt.show()

    #plt.hist(elevation_sort,bins=200)
    #plt.show()    
    

if __name__ == "__main__":
    data_center = np.array([0,0,1.8])
    list_data = []
    for file in glob.glob("/home/lxh/Documents/CarModel/npy/nuscenes*.npy"):
        points = np.load(file)
        list_data.append(points)
    fit_beam(list_data,data_center)