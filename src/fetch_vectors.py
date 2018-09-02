import pandas as pd
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
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
#import glob
#import sys
#import multiprocessing
ImageFile.LOAD_TRUNCATED_IMAGES = True


def run_durka(length):
    image_path_list = pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
    #feature_matrix = np.zeros((len(image_path_list),4096))
    #start = len(glob.glob1('../data/feature_vec/','*.jpg'))
    model = initialize_neural_network()
    #length:len(image_path_list)

    for idx,img in enumerate(image_path_list[length:66404]):
        #feature_matrix[idx] = vectorize_image(img, model)
        vectorize_image(img, model)
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #outputs = pool.map(lambda l: vectorize_image(l,model), image_path_list.tolist())
    #outputs = pool.map(vectorize_image, image_path_list)
    #np.save('data/feature_matrix/fetch_feature_matrix', outputs)
    #np.save('../data/feature_matrix/fetch_feature_matrix', feature_matrix)
    print('All dogs vectorized!')
    #print('All dogs vectorized! Feature matrix created & saved.')

def initialize_neural_network():
    model = vgg16.VGG16(include_top = True, weights = 'imagenet')
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    return model

def vectorize_image(image_name, model):
    start = time.time()
    #model = initialize_neural_network()
   
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
    #Implement error checking for (end-start)>10secs?
    print('Features vectorized for '+image_name+'   Time: '+str(end-start))
    #sys.stdout.flush()
    #return predictions

def create_file_list():
    return os.listdir('../data/feature_vec/')

def create_feature_matrix():
    start = time.time()
    image_path_list = os.listdir('../data/feature_vec/')
    #pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
    feature_matrix = np.zeros((len(image_path_list),4096))

    for idx,img_name in enumerate(image_path_list):
        feature_matrix[idx] = np.load('../data/feature_vec/feature_vec_'+img_name.split('.')[0]+'.npy')
        if idx%1000 == 0:
            print(str(idx)+' vectors merged to feature matrix.')
       
    end = time.time()
    print('Features matrix created. Time: '+str(end-start))
