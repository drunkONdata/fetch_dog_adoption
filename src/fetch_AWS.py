import pandas as pd
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
from collections import deque
#import requests 
#import json
#from IPython.display import display, Image
import urllib.request
#from PIL.ExifTags import TAGS
import PIL.Image
import PIL
from PIL import Image, ImageFile
import time
import requests
from io import BytesIO
import io
import boto3
import os
#import glob
#import sys
#import multiprocessing
ImageFile.LOAD_TRUNCATED_IMAGES = True


def fetch_run(length):
    image_path_list = pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
    #feature_matrix = np.zeros((len(image_path_list),4096))
    #start = len(glob.glob1('../data/feature_vec/','*.jpg'))
    model = initialize_neural_network()
    
    for idx,img in enumerate(image_path_list[length:66404]):
        #feature_matrix[idx] = vectorize_image(img, model)
        vectorize_image(img, model)
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #outputs = pool.map(lambda l: vectorize_image(l,model), image_path_list.tolist())
    #outputs = pool.map(vectorize_image, image_path_list)
    print('All dogs vectorized!')

def initialize_neural_network():
    model = vgg16.VGG16(include_top = True, weights = 'imagenet')
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    return model


def vectorize_image(image_name, model):
    start = time.time() 
    URL ='https://s3-us-west-2.amazonaws.com/hole-in-a-bucket/data/'+image_name
    with urllib.request.urlopen(URL) as url_open:
            f = io.BytesIO(url_open.read())
    '''
    filepath = '/data/'+image_name
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='hole-in-a-bucket', Key=filepath)
    data = obj['Body'].read()
    f = BytesIO(data)
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
    '''
    dog = load_img(f, target_size=(224, 224))
    image_batch = np.expand_dims(img_to_array(dog), axis=0)  
    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    np.save('../data/feature_vec_corrupt/feature_vec_'+image_name.split('.')[0], predictions)            

    end = time.time()
    error_log = deque()
    #Error Log for files that take longer than 5 seconds
    if end-start > 5:
        error_log.append(image_name.split('.')[0])
    print('Features vectorized for '+image_name+'   Time: '+str(end-start))
    return error_log

def create_file_list():
    vector_list = [f for f in os.listdir('../data/feature_vec/') if f.endswith('.npy')]
    fetch_image_names = pd.Series(vector_list)
    fetch_image_names.to_pickle('../data/fetch_vector_list.pkl', compression='gzip')
    #return vector_list

def create_feature_matrix():
    '''
    Merge all individual image vectors into one feature matrix.
    '''
    start = time.time()
    image_path_list = [f for f in os.listdir('../data/feature_vec/') if f.endswith('.npy')]
    #pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
    feature_matrix = np.zeros((len(image_path_list),4096))

    for idx,img_name in enumerate(image_path_list):
        feature_matrix[idx] = np.load('../data/feature_vec/'+img_name)
        #feature_matrix[idx] = np.load('../data/feature_vec/feature_vec_'+img_name.split('.')[0]+'.npy')
        if idx%1000 == 0:
            print(str(idx)+' vectors merged to feature matrix.')
    
    np.save('../data/feature_matrix/fetch_feature_matrix', feature_matrix)
    end = time.time()
    print(len(image_path_list)+' image vectors merged. Features matrix created. Time: '+str(end-start))