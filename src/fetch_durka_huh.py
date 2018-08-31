import pandas as pd
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
import urllib.request
import PIL.Image
import PIL
from PIL import Image, ImageFile
import time
import requests
from io import BytesIO
import io
ImageFile.LOAD_TRUNCATED_IMAGES = True

class feature_extraction():
    
    def __init__(self):
        # Initialize VGG16
        model = vgg16.VGG16(include_top = True, weights = 'imagenet')
        model.layers.pop()
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        
        self.model = model
        self.feature_matrix = np.zeros((140741,4096))
        self.image_path_list = pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
      
    def open_image(self, img_name):
        # Open image file given image file name as string      
        URL ='https://s3-us-west-2.amazonaws.com/hole-in-a-bucket/data/'+img_name
        with urllib.request.urlopen(URL) as url_open:
            f = io.BytesIO(url_open.read())
        return f
       
    def image_processor(self, img_file):
        # Processes image prior to entering VGG16
        dog = load_img(open_image(img_file), target_size=(224, 224))
        image_batch = np.expand_dims(img_to_array(dog), axis=0)  
        return vgg16.preprocess_input(image_batch.copy())
    
    def create_features(self, num_files=len(self.image_path_list)):
        # Create feature vectors for each image, save as .npy
        for idx, img_name in enumerate(self.image_path_list.tolist()[0:num_files]):
            self.feature_matrix[idx] = self.model.predict(image_processor(open_image(img_name)))
            np.save('/data/fetch_feature_vector_'+str(idx), self.feature_matrix[idx])
        return durka

    def merge_features(self, )




        start = time.time()
    #combined_df, combined_imgs = load_RG_data()
    #num_images = len(glob.glob1('/Users/bil2ab/galvanize/RG5kimages/','*.jpg'))
    image_path_list = pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
    feature_matrix = np.zeros((140741,4096))
    
    model = vgg16.VGG16(include_top = True, weights = 'imagenet')
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]

    for idx,img_name in enumerate(image_path_list.tolist()):    
        URL ='https://s3-us-west-2.amazonaws.com/hole-in-a-bucket/data/'+img_name
        
        with urllib.request.urlopen(URL) as url_open:
            f = io.BytesIO(url_open.read())

        #img = Image.open(f)       
        dog = load_img(f, target_size=(224, 224))
        image_batch = np.expand_dims(img_to_array(dog), axis=0)  
        processed_image = vgg16.preprocess_input(image_batch.copy())
        feature_matrix[idx] = model.predict(processed_image)
        print('File Processed: '+str(idx))
    #Save list of feature arrays as compressed numpy data file
    #doggie = np.asarray(feature_array_list)
    #np.save('/data/fetch_feature_matrix', feature_matrix)
    np.savez_compressed('data/fetch_feature_matrix', feature_matrix)
    
    end = time.time()
    print('Total Time: '+str(end-start))
    print('All dog features vectorized!')
    return feature_matrix





def durka():
    
    start = time.time()
    #combined_df, combined_imgs = load_RG_data()
    #num_images = len(glob.glob1('/Users/bil2ab/galvanize/RG5kimages/','*.jpg'))
    image_path_list = pd.read_pickle('../data/fetch_img_urls.pkl', compression='gzip')
    feature_matrix = np.zeros((140741,4096))
    
    model = vgg16.VGG16(include_top = True, weights = 'imagenet')
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]

    for idx,img_name in enumerate(image_path_list.tolist()):    
        URL ='https://s3-us-west-2.amazonaws.com/hole-in-a-bucket/data/'+img_name
        
        with urllib.request.urlopen(URL) as url_open:
            f = io.BytesIO(url_open.read())

        #img = Image.open(f)       
        dog = load_img(f, target_size=(224, 224))
        image_batch = np.expand_dims(img_to_array(dog), axis=0)  
        processed_image = vgg16.preprocess_input(image_batch.copy())
        feature_matrix[idx] = model.predict(processed_image)
        print('File Processed: '+str(idx))
    #Save list of feature arrays as compressed numpy data file
    #doggie = np.asarray(feature_array_list)
    #np.save('/data/fetch_feature_matrix', feature_matrix)
    np.savez_compressed('data/fetch_feature_matrix', feature_matrix)
    
    end = time.time()
    print('Total Time: '+str(end-start))
    print('All dog features vectorized!')
    return feature_matrix

    
