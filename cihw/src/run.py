import matplotlib.pyplot as plt
import re

import numpy as np
from sklearn import svm
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.io import imread, ImageCollection

# function to plot the graphs and images 
def plot_graphs(image_list,rescaled_hog_image_list,hist_list,subx,suby):    
    fig, ax = plt.subplots(subx, suby, figsize=(8, 4))
    fig1, ax1 = plt.subplots(subx, suby, figsize=(8,4))
    fig2, ax2 = plt.subplots(subx, suby, figsize=(8,4))
    
    img_idx = 0
    for x in range(subx):
        for y in range(suby):
            ax[x][y].axis('off')
            ax[x][y].imshow(image_list[img_idx], 
                            cmap=plt.cm.gray)
            img_idx+=1
    
    img_idx = 0
    for x in range(subx):
        for y in range(suby):
            ax1[x][y].axis('off')
            ax1[x][y].imshow(rescaled_hog_image_list[img_idx], 
                             cmap=plt.cm.gray)
            img_idx+=1
    
    img_idx = 0
    for x in range(subx):
        for y in range(suby):
            #ax2[x][y].axis('off')
            width = 0.7 * (hist_list[img_idx][1][1]-hist_list[img_idx][1][2]) 
            ax2[x][y].bar(hist_list[img_idx][1],
                          hist_list[img_idx][0],
                          align='center',
                          width=width)
            img_idx+=1

    plt.show() 

def get_target_list(image_list):
    file_list = image_list.files
    target_list = []
    for file_name in file_list:
        m = re.search('[a-zA-Z0-9./]*_([0-9]*)k.PNG',file_name)
        ill_val = int(m.group(1))
        target_list.append(ill_val)
    return target_list

if __name__=="__main__":

    #loading the train dataset
    train_image_list = ImageCollection('../data/train/*')
    train_rescaled_hog_image_list = []
    train_hog_histograms_list = []
    train_hist_list= []
    #calculating the features for train set
    for image in train_image_list:
        gray_image = color.rgb2gray(image)
        fd, hog_image = hog(gray_image, 
                            orientations=10, 
                            pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), 
                            visualise=True)
        train_hog_histograms_list.append(exposure.histogram(hog_image, nbins=256))
        train_rescaled_hog_image_list.append(exposure.rescale_intensity(hog_image, 
                                                                  in_range=(0, 0.02)))
        train_hist_list.append(exposure.histogram(gray_image, nbins=256))

    #plot_graphs(train_image_list,train_rescaled_hog_image_list,train_hist_list,5,7)

    #loading the test dataset
    test_image_list = ImageCollection('../data/test/*')
    test_rescaled_hog_image_list = []
    test_hog_histograms_list = []
    test_hist_list= []
    #calculating the features for test set
    for image in test_image_list:
        gray_image = color.rgb2gray(image)
        fd, hog_image = hog(gray_image, 
                            orientations=10, 
                            pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), 
                            visualise=True)
        test_hog_histograms_list.append(exposure.histogram(hog_image, nbins=256))
        test_rescaled_hog_image_list.append(exposure.rescale_intensity(hog_image, 
                                                                  in_range=(0, 0.02)))
        test_hist_list.append(exposure.histogram(gray_image, nbins=256))

    #plot_graphs(test_image_list,test_rescaled_hog_image_list,test_hist_list,2,3)

    #Applying regression and generating models
    clf = svm.SVR() 
    train_target_list = get_target_list(train_image_list)
    #extracting only the histogram from the tuples with histogram and centers lists
    train_hog_histograms_list = [x[0]  for x in train_hog_histograms_list]
    clf.fit(train_hog_histograms_list,train_target_list)

    test_hog_histograms_list = [x[0]  for x in test_hog_histograms_list]
    print clf.predict(test_hog_histograms_list)

    train_hist_list = [x[0]  for x in train_hist_list]
    clf.fit(train_hist_list,train_target_list)

    test_hist_list = [x[0]  for x in test_hist_list]
    print clf.predict(test_hist_list)
