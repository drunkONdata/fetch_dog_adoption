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
    Find city and state from zip code. 
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