import pandas as pd
import numpy as np
import scipy.stats as scs

import keras
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation
import tensorflow as tf
import requests 

import json
from IPython.display import display, Image
import urllib.request
from PIL.ExifTags import TAGS
import PIL.Image
import time
from geopy.geocoders import Nominatim
import zipfile


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
    #https://drive.google.com/open?id=1Q16HK3A93_6C-D_96-AUwwKWWf4ruBav
    #zip_images = zipfile.ZipFile('/data/images.zip', 'r')
    #zip_images.extractall('/data/')
    #zip_images.close()
    
    df0, image0 = extract_df('/data/h9DH7711_newpets_1.json')
    df1, image1 = extract_df('/data/h9DH7711_pets_1.json')
    df2, image2 = extract_df('/data/h9DH7711_pets_2.json')
    df3, image3 = extract_df('/data/h9DH7711_pets_3.json')
    df4, image4 = extract_df('/data/h9DH7711_pets_4.json')
    df5, image5 = extract_df('/data/h9DH7711_pets_5.json')

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
    Find city and state from zip code. 
    INPUT: zip code
    OUTPUT: Returns city and state
    '''
    geolocator = Nominatim()
    location = geolocator.geocode(zip_code)
    city = location.address.split(',')[0].strip()
    state = location.address.split(',')[1].strip()
    return city, state


def rotate_image(filepath):
    
    '''Phones rotate images by changing exif data, 
    but we really need to rotate them for processing'''
    
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


def vectorize_dog_images(image_path_list, length=25):
    '''
    Take collection of dog images and vectorize each image to a 1D NumPy array. 
    INPUT: List, Pandas Series, some iterable of filepaths to dog images (strings)
    OUTPUT: Returns Numpy data file
    '''
    start = time.time()
    feature_array_list = []
    #image_path_list formerly combined_df.ImageUrl[0:4750]
    for url in image_path_list[0:length]:
        image_path = 'data/images/'+url.split('/')[-1]
        dog = load_img(image_path, target_size=(224, 224))
        numpy_image = img_to_array(dog)
        image_batch = np.expand_dims(numpy_image, axis=0)  
        processed_image = vgg16.preprocess_input(image_batch.copy())
        feature_array = model.predict(processed_image)
        feature_array_list.append(feature_array)
        #doggie = np.asarray(feature_array_list)
        #np.save('data/RG_features', doggie)
    end = time.time()
    total_time = end-start
    print('Total Time: '+str(total_time))
    print('All dog features vectorized!')
    return feature_array_list


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

def durka():
   
    start = time.time()
    combined_df, combined_imgs = load_RG_data()
    #num_images = len(glob.glob1('/Users/bil2ab/galvanize/RG5kimages/','*.jpg'))
    image_path_list = combined_imgs.ImageUrl
    #Pickle image urls
    image_path_list.to_pickle('/data/fetch_img_urls.pkl', compression='gzip')
    feature_matrix = np.zeros((140741,4096))
    
    model = vgg16.VGG16(include_top = True, weights = 'imagenet')
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]

    for idx,url in enumerate(image_path_list.tolist()[0:140741]):
        dog = load_img('/data/images/'+url.split('/')[-1], target_size=(224, 224))
        image_batch = np.expand_dims(img_to_array(dog), axis=0)  
        processed_image = vgg16.preprocess_input(image_batch.copy())
        feature_matrix[idx] = model.predict(processed_image)
    
    #Pickle image urls
    #image_path_list.to_pickle('/data/fetch_img_urls.pkl', compression='gzip')
    
    #Save list of feature arrays as numpy data file
    #doggie = np.asarray(feature_array_list)
    np.save('/data/fetch_feature_matrix', feature_matrix)
    
    end = time.time()
    print('Total Time: '+str(end-start))
    print('All dog features vectorized!')
    return feature_matrix