import pandas as pd
import numpy as np

import keras
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf

import requests 
import json
from IPython.display import display, Image
import urllib.request
from PIL.ExifTags import TAGS
#import PIL.Image
import PIL
from PIL import Image, ImageFile
import time

from io import BytesIO
import io
import boto3
import os
#import glob
#import sys
#import multiprocessing
ImageFile.LOAD_TRUNCATED_IMAGES = True
from geopy.geocoders import Nominatim

import scipy.stats as scs

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
    '''
    Initializes VGG16 model in Keras without the softmax classification layer and last fully connected layer. 
    '''
    model = vgg16.VGG16(include_top = True, weights = 'imagenet')
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    return model


def vectorize_image(image_name, model):
    '''
    Take collection of dog images and vectorize each image to a 1D NumPy array. 
    INPUT: List, Pandas Series, some iterable of filepaths to dog images (strings)
    OUTPUT: Returns Numpy data file
    '''
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
    #return predictions


def create_file_list():
    '''
    Create a list of all vectorized images in .npy format, create Pandas Series & pickle. 
    '''
    vector_list = [f for f in os.listdir('../data/feature_vec/') if f.endswith('.npy')]
    fetch_image_names = pd.Series(vector_list)
    fetch_image_names.to_pickle('../data/fetch_vector_list.pkl', compression='gzip')
    np.save('../data/feature_vector_list', vector_list)
    return vector_list


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
    print('Features matrix created. Time: '+str(end-start))


def extract_image_url(pd_series):
    '''
    Extracts image URLs from the pictures column in the RescueGroups database.
    INPUT: Pandas Series where each item is a list of dictionaries of dictionaries??
    OUTPUT: Pandas dataframe with animalID and imageURL
    '''
    large_image_urls = []
    animalIDs = []
        
    for lst in pd_series:
        for dct in lst:
            large_image_urls.append(dct['largeUrl'])
                
    for url in large_image_urls:
        animalIDs.append(url.split('/')[-2])
    
    return pd.DataFrame({'animalID': animalIDs,'ImageUrl': large_image_urls})


def extract_df(filepath):
    '''
    Extracts orgId, animalID, name breed and animalLocation from RescueGroups JSON and adds imageURLs
    INPUT: JSON filepath, string
    OUTPUT: Pandas dataframes
    '''
    df = pd.read_json(filepath, lines=True)
    images = extract_image_url(df.pictures)
    df1 = df[['orgID','animalID','name','animalLocation']]
    # NOTE: You loose images with this concat
    result = pd.concat([df1, images.ImageUrl], axis=1, join_axes=[df1.index])
    # Return combined dataframe and original image source dataframe
    return result, images


def download_images(urls, length=25):
    '''
    Downloads all images from Rescue Groups S3 bucket 
    INPUT: Pandas Series of URLs
    OUTPUT: Images stored in data directory.
    '''
    for image_url in list(urls)[0:length]:
        image_name = image_url.split('/')[-1]
        r = requests.get(image_url, allow_redirects = True)
        open('data/images/'+image_name, 'wb').write(r.content)


def load_RG_data():
    '''
    Load data from RescueGroup JSONs into Pandas dataframes and merge to single dataframe 
    INPUT: None
    OUTPUT: Returns 2 dataframes, one of image URLs and other of other info
    '''
    df0, image0 = extract_df('/Users/bil2ab/galvanize/RG_JSON/h9DH7711_newpets_1.json')
    df1, image1 = extract_df('/Users/bil2ab/galvanize/RG_JSON/h9DH7711_pets_1.json')
    df2, image2 = extract_df('/Users/bil2ab/galvanize/RG_JSON/h9DH7711_pets_2.json')
    df3, image3 = extract_df('/Users/bil2ab/galvanize/RG_JSON/h9DH7711_pets_3.json')
    df4, image4 = extract_df('/Users/bil2ab/galvanize/RG_JSON/h9DH7711_pets_4.json')
    df5, image5 = extract_df('/Users/bil2ab/galvanize/RG_JSON/h9DH7711_pets_5.json')

    combined_df = df0.append([df1, df2, df3, df4, df5])
    combined_imgs = image0.append([image1, image2, image3, image4, image5])
    combined_df = combined_df.reset_index(drop=True)
    combined_imgs = combined_imgs.reset_index(drop=True)

    total_records = [df0.shape[0], df1.shape[0], df2.shape[0], df3.shape[0], df4.shape[0], df5.shape[0]]
    image_records = [image0.shape[0], image1.shape[0], image2.shape[0], image3.shape[0], image4.shape[0], image5.shape[0]]
    print('Total Records: ',sum(total_records))
    print('Total Images: ',sum(image_records))
    
    return combined_df, combined_imgs


def zip_lookup(zip_code):
    '''
    Find city and state from zip code 
    INPUT: zip code
    OUTPUT: Returns city and state
    '''
    geolocator = Nominatim()
    location = geolocator.geocode(zip_code)
    city = location.address.split(',')[0].strip()
    state = location.address.split(',')[1].strip()
    return city, state


def gps_lookup(gps):
    '''
    Find city and state from GPS coordinates. 
    INPUT: zip code
    OUTPUT: Returns city and state
    '''
    geolocator = Nominatim()
    location = geolocator.geocode(gps)
    city = location.address.split(',')[0].strip()
    state = location.address.split(',')[1].strip()
    return city, state


def rotate_image(filepath):
    '''
    Rotates images from cellphones if needed by checking exif data, prior to processing. 
    '''
    image=Image.open(filepath)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
            exif=dict(image._getexif().items())
    
        if exif[orientation] == 3:
            print('ROTATING 180')
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            print('ROTATING 270')
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            print('ROTATING 90')
            image=image.rotate(90, expand=True)
        image.save(filepath)
        image.close()
    except (AttributeError, KeyError, IndexError):
    # cases: image don't have getexif   
        pass
    return(image)





def similarity(user_image, feature_array_collection):
    '''
    Calculate cosine similarity between user submitted image and entire adoptable dog corpus 
    INPUT: User submitted image in form of feature vector (NumPy Array, 1D x 4096 features)
    OUTPUT: Returns list of cosine similarity scores between user submitted image and entire adoptable dog corpus
    '''
    results = []
    for feature_array in feature_array_collection:
        results.append(distance.cosine(user_image.flatten(),feature_array.flatten()))
    #print('Max Similarity Score = ' +str(max(results))
    #similar_images=pd.DataFrame({'imgfile':images,'simscore':sims})
    return results
    

def top_matches(results, imageURLs, num_top_matches):
    '''
    Creates zipped list of image files and similarity scores. 
    INPUT: Similarity scores (list), imageURLs (Pandas Series), top_matches (int)
    OUTPUT: Returns similarity scores and images urls (Pandas Dataframe)
    '''
    zipped_dogs = list(zip(imageURLs.tolist(),results))
    sorted_zipped_dogs = sorted(zipped_dogs, key = lambda t: t[1])
    #num_top_matches=10
    return sorted_zipped_dogs[0:num_top_matches]
    #return pd.DataFrame({'Image_URLs':sorted_zipped_dogs[0],'Similarity_Score':sorted_zipped_dogs[1]})


def find_matches(pred, collection_features, images):
    pred = pred.flatten()
    nimages = len(collection_features)
    #vectorize cosine similarity
    #sims= inner(pred,collection_features)/norm(pred)/norm(collection_features,axis=1)
    sims = []
    for i in range(0,nimages):
        sims.append(distance.cosine(pred.flatten(),collection_features[i].flatten()))
    print('max sim = ' +str(max(sims)))
    similar_images=pd.DataFrame({'imgfile':images,'simscore':sims})
    return(similar_images)


def display_top_matches(top_dogs):
    
    for image, score in top_dogs:
        plt.imshow(load_img('/Users/bil2ab/galvanize/RG5kimages/'+image.split('/')[-1]))
        plt.show()
        print(1-score)