#!/usr/bin/env python
# coding: utf-8

# # Setups, Installations and Imports

# In[1]:




# In[2]:



# In[3]:


import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import io
import itertools
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from tqdm.notebook import tqdm_notebook
from ipywidgets import IntProgress
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold
import urllib.request
import shutil
import tarfile
from scipy.io import loadmat
from keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import mnist


# # Download and Prepare Datasets

# #### CIFAR-10 - split dataset to 2 sub datasets. each with its own run

# In[30]:


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    data = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)    
    x1, x2, y1, y2 = train_test_split(data, labels, train_size=0.5, test_size=0.5, random_state= 3)    
    num_labels = 10
    img_shape=32
    return x1, x2, y1, y2, num_labels, img_shape


# #### CIFAR-100 - split dataset to 2 sub datasets. each with its own run

# In[33]:


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
    data = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)   
    x1, x2, y1, y2 = train_test_split(data, labels, train_size=0.5, test_size=0.5, random_state= 3)    
    num_labels = 20
    img_shape=32
    return x1, x2, y1, y2, num_labels, img_shape


# #### MNIST - split dataset to 2 sub datasets. each with its own run

# In[34]:


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)   
    x1, x2, y1, y2 = train_test_split(data, labels, train_size=0.5, test_size=0.5, random_state=3)    
    num_labels = 10
    img_shape=28
    color = 1
    return x1, x2, y1, y2, num_labels, img_shape, color


# #### 102 flowers - split dataset to 2 sub datasets. each with its own run

# In[6]:


def extract(tar_url, extract_path='.'):
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
            
def get_categories():
    mat = loadmat('imagelabels.mat')
    _categories = mat['labels'][0]
    return _categories

def encode_categories(categories):
    padded_zero_arr = []
    for cat in categories:      
        padded_zero_arr.append(cat-1)
    padded_zero_arr_np = np.asarray(padded_zero_arr)
    return padded_zero_arr_np

def split_data(categories):
    file_names = os.listdir('jpg/')
    file_names.sort()
    x_train, x_test, y_train, y_test = train_test_split(file_names, categories, train_size=0.5, test_size=0.5, random_state=3)    
    return x_train, x_test, y_train, y_test

def _preprocess_images(arr_of_names):
    total_images_preprocessed = []
    for i, f in enumerate(arr_of_names):
        # f = fileName[i]
        # print(f)
        imagePath = "jpg/" + f
        img = load_img(imagePath, target_size=(224, 224, 3))
        # convert the image pixels to a numpy array
        img_arr = img_to_array(img)        
        total_images_preprocessed.append(img_arr)

    return np.asarray(total_images_preprocessed)

def load_flowers102():
    # uncomment below line if you get SSL error
    # ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', '102flowers.tgz')
    urllib.request.urlretrieve('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat', 'imagelabels.mat')
    extract('102flowers.tgz')
    categories = get_categories()
    encoded_categories = encode_categories(categories)
    x_train, x_test, y_train, y_test = split_data(encoded_categories)
    
    y_train = np.asarray(y_train)
    y_train = y_train.reshape((-1, 1))
    y_test =  np.asarray(y_test)
    y_test = y_test.reshape((-1, 1))

    x_train = _preprocess_images(x_train)
    x_test = _preprocess_images(x_test)

    num_labels = 102
    img_shape=224
    return x_train, x_test, y_train, y_test, num_labels, img_shape


# #### Kaggle general methods

# In[4]:


def load_data(path, target_image_size):       
    images = []
    labels = []
    labels_count = 0
    for i in os.listdir(path):
        imageList = os.listdir(path + "/" + i)
        for j in imageList:          
            img = load_img(path + "/" + i + "/" + j, target_size=(target_image_size, target_image_size, 3))
            # convert the image pixels to a numpy array
            img_arr = img_to_array(img)        
            images.append(img_arr)
            labels.append(labels_count)
            
        labels_count = labels_count+1
                
    #print(labels)
    return (images, labels)


# #### install Kaggle API

# In[5]:


get_ipython().system('pip install kaggle')
get_ipython().system('mkdir /root/.kaggle')
get_ipython().system('touch /root/.kaggle/kaggle.json')
api_token = {"username":"","key":""}
# put your personal Kaggle API token key
# for  getting your Kaggle API token, go to your Kaggle->Account->Create new API Token. it generates a JSON file with your token that should be placed here
get_ipython().system('echo \'{"username":"YOUR_ID","key":"YOUR_KEY"}\' > ~/.kaggle/kaggle.json')
get_ipython().system('cat ~/.kaggle/kaggle.json')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# #### 10 monkeys (Kaggle) - split dataset to 2 sub datasets. each with its own run

# In[6]:


def load_monkeys10():
    shutil.rmtree('./datasets/Kaggle', ignore_errors=True)
    get_ipython().system('mkdir ./datasets')
    get_ipython().system('mkdir ./datasets/Kaggle')
    # download the dataset from Kaggle and unzip it
    get_ipython().system('kaggle datasets download -d slothkong/10-monkey-species -p ./datasets/Kaggle')
    get_ipython().system('unzip ./datasets/Kaggle/*.zip  -d ./datasets/Kaggle')
    get_ipython().system('ls ./datasets/Kaggle')

    img_shape=128

    data1, labels1 = load_data("/content/datasets/Kaggle/validation/validation", img_shape)
    data2, labels2 = load_data("/content/datasets/Kaggle/training/training", img_shape)
    data = data1+data2
    labels = labels1+labels2
    data = np.asarray(data)
    labels = np.asarray(labels)
    x1, x2, y1, y2 = train_test_split(data, labels, train_size=0.5, test_size=0.5, random_state=3)    
    num_labels = 10
    return x1, x2, y1, y2, num_labels, img_shape


# #### Intel images (6 labeled images from Kaggle) - split dataset to 2 sub datasets. each with its own run

# In[7]:


def load_intel():
    shutil.rmtree('./datasets/Kaggle', ignore_errors=True)
    get_ipython().system('mkdir ./datasets')
    get_ipython().system('mkdir ./datasets/Kaggle')
    # download the dataset from Kaggle and unzip it
    get_ipython().system('kaggle datasets download -d puneet6060/intel-image-classification -p ./datasets/Kaggle')
    get_ipython().system('unzip ./datasets/Kaggle/*.zip  -d ./datasets/Kaggle')
    get_ipython().system('ls ./datasets/Kaggle')

    img_shape=128
    data1, labels1 = load_data("/content/datasets/Kaggle/seg_test/seg_test", img_shape)
    data2, labels2 = load_data("/content/datasets/Kaggle/seg_train/seg_train", img_shape)
    data = data1+data2
    labels = labels1+labels2
    data = np.asarray(data)
    labels = np.asarray(labels)
    x1, x2, y1, y2 = train_test_split(data, labels, train_size=0.5, test_size=0.5, random_state=3)    
    num_labels = 10
    return x1, x2, y1, y2, num_labels, img_shape


# #### test datasets creations

# In[9]:


#load_cifar10()
#load_cifar100()
#load_mnist()
#load_flowers102()
#load_monkeys10()
#load_intel()

