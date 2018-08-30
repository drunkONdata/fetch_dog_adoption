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
import time
import requests
from io import BytesIO
import io


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

        img = Image.open(f)       
        dog = load_img(img, target_size=(224, 224))
        image_batch = np.expand_dims(img_to_array(dog), axis=0)  
        processed_image = vgg16.preprocess_input(image_batch.copy())
        feature_matrix[idx] = model.predict(processed_image)
      
    #Save list of feature arrays as compressed numpy data file
    #doggie = np.asarray(feature_array_list)
    #np.save('/data/fetch_feature_matrix', feature_matrix)
    np.savez_compressed('data/fetch_feature_matrix', feature_matrix)
    
    end = time.time()
    print('Total Time: '+str(end-start))
    print('All dog features vectorized!')
    return feature_matrix

    
