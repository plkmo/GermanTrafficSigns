# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:20:29 2019

@author: WT
"""

#from paths import path
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt
import pickle


def readTrafficSigns_train(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    areas = [] # area around traffic sign
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.__next__() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(int(row[7])) # the 8th column is the label
            areas.append((int(row[3]),int(row[4]),int(row[5]),int(row[6])))
        gtFile.close()
    return images, labels, areas

def readTrafficSigns_test(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    areas = [] # area around traffic sign
    prefix = rootpath + '/'  # subdirectory for class
    gtFile = open(prefix + 'GT-final_test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.__next__() # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        labels.append(int(row[7])) # the 8th column is the label
        areas.append((int(row[3]),int(row[4]),int(row[5]),int(row[6])))
    gtFile.close()
    return images, labels, areas

## Crop images to ROI
def crop_images(images, areas, resize=False, size=(50,50)):
    cropped_images = []
    for image, area in zip(images, areas):
        image = Image.fromarray(image, "RGB")
        image = image.crop(area)
        if resize:
            image = image.resize(size=size)
        cropped_images.append(image)
    return cropped_images

def save_as_pickle(filename, data):
    completeName = os.path.join("C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

if __name__ == "__main__":
    train_data_path =  "C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
    test_data_path =  "C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    # train
    images, labels, areas = readTrafficSigns_train(train_data_path)
    images = crop_images(images, areas, resize=True)
    save_as_pickle("train_images.pkl", images)
    save_as_pickle("train_images_labels.pkl", labels)
    
    # test
    images, labels, areas = readTrafficSigns_test(test_data_path)
    images = crop_images(images, areas, resize=True)
    save_as_pickle("test_images.pkl", images)
    save_as_pickle("test_images_labels.pkl", labels)