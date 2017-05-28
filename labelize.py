# -*- coding: utf-8 -*-
import glob
import scipy as sp
from PIL import Image
import os, sys
from resizeimage import resizeimage
import cv2
import numpy as np

def labelize():
    txt = open('labeledData.csv','a') 
    '''
    for file in glob.iglob('./realism_output/*.jpeg'):  
        print 1
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.shape = (1, 784)
        img = np.concatenate((img, np.array([[0]])), axis = 1)
        np.savetxt(txt, img, fmt='%d', delimiter=",")
    '''

    for file in glob.iglob('./unabstract_ouput/*.jpeg'):  
        print 1
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.shape = (1, 784)
        img = img/255.0
        img = np.concatenate((img, np.array([[1]])), axis = 1)
        np.savetxt(txt, img, fmt='%f', delimiter=",")

    '''
    for file in glob.iglob('./expressionism_output/*.jpeg'):  
        print 1
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.shape = (1, 784)
        img = np.concatenate((img, np.array([[1]])), axis = 1)
        np.savetxt(txt, img, fmt='%d', delimiter=",")
    '''

    for file in glob.iglob('./abstract_output/*.jpeg'):  
        print 1
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.shape = (1, 784)
        img = img/255.0
        img = np.concatenate((img, np.array([[0]])), axis = 1)
        np.savetxt(txt, img, fmt='%f', delimiter=",")

    txt.close()

labelize()