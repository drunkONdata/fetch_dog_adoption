#Miscellaneous Functions for Fetch! Dog Adoption, not utilized

from scipy.spatial import distance
import pandas as pd
from numpy import inner
from numpy.linalg import norm


def cosine_similarity(user_predict, adoptable_dogs, images):
    '''
    Calculating cosine similarity between user submitted picture and adoptable dogs collection.
    INPUT: predict = features from user submitted image, 
           adoptable_dogs = list of features in the collection, 
           images = list of image filepaths associated with adoptable_dogs features 
    OUTPUT: Pandas dataframe with image filepath and similarity score
    '''   
    sim_score = []
    for idx in range(0, len(adoptable_dogs)):
        sim_score.append(distance.cosine(user_predict.flatten(), adoptable_dogs[idx].flatten()))
    print('Maximum SimScore: '+str(max(sim_score)))
    return pd.DataFrame({'imgFile':images, 'SimScore':sim_score})

def bray_curtis_dist(user_predict, adoptable_dogs, images):
    '''
    Calculating Bray-Curtis distance between two 1D arrays and return similarity score
    '''   
    sim_score = []
    for idx in range(0, len(adoptable_dogs)):
        sim_score.append(distance.braycurtis(user_predict.flatten(), adoptable_dogs[idx].flatten()))
    print('Maximum SimScore: '+str(max(sim_score)))
    return pd.DataFrame({'imgFile':images, 'SimScore':sim_score})

def canberra_dist(user_predict, adoptable_dogs, images):
    '''
    Calculating Canberra distance between two 1D arrays and return similiarty score
    '''   
    sim_score = []
    for idx in range(0, len(adoptable_dogs)):
        sim_score.append(distance.canberra(user_predict.flatten(), adoptable_dogs[idx].flatten()))
    print('Maximum SimScore: '+str(max(sim_score)))
    return pd.DataFrame({'imgFile':images, 'SimScore':sim_score})

def cheb_dist(user_predict, adoptable_dogs, images):
    '''
    Calculating Chepyshev distance between two 1D arrays and return similarity score
    '''   
    sim_score = []
    for idx in range(0, len(adoptable_dogs)):
        sim_score.append(distance.chebyshev(user_predict.flatten(), adoptable_dogs[idx].flatten()))
    print('Maximum SimScore: '+str(max(sim_score)))
    return pd.DataFrame({'imgFile':images, 'SimScore':sim_score})

def manhattan_dist(user_predict, adoptable_dogs, images):
    '''
    Calculating Manhattan distance between two 1D arrays and return similarity score
    '''   
    sim_score = []
    for idx in range(0, len(adoptable_dogs)):
        sim_score.append(distance.cityblock(user_predict.flatten(), adoptable_dogs[idx].flatten()))
    print('Maximum SimScore: '+str(max(sim_score)))
    return pd.DataFrame({'imgFile':images, 'SimScore':sim_score})

def euclidean_dist(user_predict, adoptable_dogs, images):
    '''
    Calculating Euclidean distance between two 1D arrays and return similarity score
    '''   
    sim_score = []
    for idx in range(0, len(adoptable_dogs)):
        sim_score.append(distance.euclidean(user_predict.flatten(), adoptable_dogs[idx].flatten()))
    print('Maximum SimScore: '+str(max(sim_score)))
    return pd.DataFrame({'imgFile':images, 'SimScore':sim_score})

def bool_conv(array):
    '''
    Converting array to boolean, with average value threshold.
    '''   
    avg = sum(array.flatten())/len(array.flatten())
    return array > avg

def binary_conv(array):
    '''
    Converting array to binary, with average value threshold. 
    '''   
    avg = sum(array.flatten())/len(array.flatten())
    return np.where(array > avg, 1, 0)



    ####

import requests
import pymongo

mc = pymongo.MongoClient()
scraper_db = mc['scraper']
sites = scraper_db['sites']
sites.delete_many({})


def retrieve_site(url:str) -> bytes:
    for site in sites.find():
        if site['url'] == url:
            return site['data']

        
def store_image(url:str) -> bytes:
    data = retrieve_site(url)
    if data:
        return data
    response = requests.get(url)
    data = response.content
    sites.insert_one({'url': url,'data': data})
    return data