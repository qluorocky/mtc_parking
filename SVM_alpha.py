# -*- coding: utf-8 -*-
'''
Created on Fri Feb 19 10:24:42 2016

@author: qi
'''

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
'''
import data as global variables
'''
raw_data = pd.read_csv('leftfront2.csv', usecols = ['time','X', 'Y', 'vehicle_East', 'vehicle_North', 'vehicle_East_0', 'vehicle_North_0']);
raw_data = np.array(raw_data)

'''
pre-process data, return time step
'''
def preprocess(data):
    time_dur = (max(data[:,0]) - min(data[:,0]))/ 100.0;  #time duration in s
    time_step = min(data[:,0]) * np.ones(np.floor(time_dur / 4.0 ));
    if time_dur > 4:
        time_step = time_step + 400.0 * np.arange(np.floor(time_dur / 4.0)); 
    else:
        time_step = time_step + 400.0 * np.arange(1)
    return time_step
    
'''
loop through batch dataset to simulate dynamic sensor input
'''
def dynamic_read(data, start_t, end_t):
    reading = np.where((data[:,0] >= start_t) & (data[:,0] < end_t))
    new_reading = [];
    for i in range(len(reading)):
        ind = reading[i]
        for j in ind:
            vehicle_pos = data[j, 3:4]
            sensor_pos = data[j, 1:2] + data[j, 5:6]        
            distance = np.dot((vehicle_pos - sensor_pos).T, (vehicle_pos - sensor_pos)) **0.5
            if (distance <= 3.0) & (distance > 0.1):
                new_reading.append(j)
    new_reading = np.array(new_reading)
    if new_reading.shape[0] > 1:
        new_reading = new_reading
    else:
        new_reading = -1
    return data[new_reading,:]

'''
Mean Shift Clustering
'''
def meanshift(raw_data, t):
   # Compute clustering with MeanShift
    # The following bandwidth can be automatically detected using
    #data = [ [(raw_data[i, 1]+raw_data[i, 5]), (raw_data[i, 2]+raw_data[i,6])] for i in range(raw_data.shape[0]) ]
    data = np.zeros((raw_data.shape[0],2))
    X = raw_data[:,1] + raw_data[:,5]
    Y = raw_data[:,2] + raw_data[:,6]
    #X = raw_data[:,1] ; Y = raw_data[:,2];
    data = np.transpose(np.concatenate((np.mat(X),np.mat(Y)), axis=0))
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_) 
    # Plot result
    plt.figure(t)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.axis('equal')
    plt.show()    
    #return labels, cluster_centers, n_clusters_


'''
Gaussian Mixture Clustering
'''


'''
SVM with linear kernel
'''


'''
Plot Results
'''
def plot1(train, t):
    X = train[:, 1] + train[:, 5]
    Y = train[:, 2] + train[:, 6]
    plt.scatter(X, Y)
    plt.title('plot %d: reading from %d second to %d second' %(t+1, 4*t, 4*(t+1)))
    plt.axis('equal')
    plt.show()

def plot2(data, labels, cluster_centers, n_clusters):
    from itertools import cycle
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.show()
'''

'''
    
def main():
    time_step = preprocess(raw_data)    # a vector of timesteps in this trip
    for t in range(len(time_step)-1):    
        train_data = dynamic_read(raw_data, time_step[t], time_step[t+1])
        try:
            if train_data.shape[0] > 5:
                #plot1(train_data, t) 
                meanshift(train_data, t)
             
        except:
            if train_data.shape[0] < 5:
                print "too few points"
        

if __name__ == '__main__':
    main()

    # comment
