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
#import sys
#import multiprocessing
ImageFile.LOAD_TRUNCATED_IMAGES = True


def run_durka():
    image_path_list = pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
    feature_matrix = np.zeros((10,4096))
    model = initialize_neural_network()

    for idx,img in enumerate(image_path_list[0:10]):
        feature_matrix[idx] = vectorize_image(img, model)
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #outputs = pool.map(lambda l: vectorize_image(l,model), image_path_list.tolist())
    #outputs = pool.map(vectorize_image, image_path_list)
    #np.save('data/feature_matrix/fetch_feature_matrix', outputs)
    np.save('../data/feature_matrix/fetch_feature_matrix', feature_matrix)
    print('All dogs vectorized! Feature matrix created & saved.')

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
 
    dog = load_img(f, target_size=(224, 224))
    image_batch = np.expand_dims(img_to_array(dog), axis=0)  
    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    np.save('../data/feature_vec/feature_vec_'+image_name.split('.')[0], predictions)            
           
    end = time.time()
    print('Features vectorized for '+image_name+'   Time: '+str(end-start))
    #sys.stdout.flush()
    return predictions

'''
def create_feature_matrix():
    start = time.time()
    
    image_path_list = pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')

    for idx,img_name in enumerate(image_path_list[0:10]):
        feature_matrix[idx] = np.load('../data/feature_vec/feature_vec_'+img_name.split('.')[0]+'.npy')
       
    end = time.time()
''' 
    
