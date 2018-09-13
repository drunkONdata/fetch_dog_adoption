import os
import flask
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import werkzeug


import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import ExifTags, Image
from keras.applications import VGG16
from keras.preprocessing import image as keras_image

from scipy.spatial import distance

from numpy import inner
from numpy.linalg import norm

#from src.fetch_data_pipeline import rotate_image, similarity, top_matches
#from src.fetch_data_processing import initialize_neural_network
#from src.fetch_web import vectorize_image, similarity

# Set up filepath to store user submitted photo
UPLOAD_FOLDER = '/Users/bil2ab/galvanize/fetch_dog_adoption/web/static/temp/upload'
#DATA_FOLDER = '/Users/bil2ab/galvanize/fetch_dog_adoption/web/static/temp/data'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['DATA_FOLDER'] = DATA_FOLDER

# Load Data
vector_list = pd.read_pickle('fetch_vector_list.pkl', compression='gzip')
feature_matrix = np.load('fetch_feature_matrix.npy')
#collection_features = np.load(os.path.join(app.config['DATA_FOLDER'], 'fetch_feature_matrix.npy'))
#files_and_titles=pd.read_csv(os.path.join(app.config['DATA_FOLDER'], 'img_urls.csv'))                           

def initialize_neural_network():
    model = VGG16(include_top = True, weights = 'imagenet')
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    return model

# Initialize neural network with classification layer and fully connected layer dropped
model = initialize_neural_network()

# Verify file extension of user submitted photo    
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def top_matches(dog_vector, feature_matrix, images): 
    pred = dog_vector.flatten()
    sims = []
    
    for i in range(0,len(feature_matrix)):
        sims.append(distance.cosine(pred.flatten(), feature_matrix[i].flatten()))
    
    return pd.DataFrame({'imgfile':images, 'simscore':sims})
  

def rotate_image(filepath):     
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


global graph
graph = tf.get_default_graph()

@app.route('/',  methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')
    
    if flask.request.method == 'POST':
        # No file in post submission
        if 'file' not in flask.request.files:
            print('No file!') #flash
            return redirect(flask.request.url)
        file = flask.request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No image selected!') #flash
            return redirect(request.url)
        if file and allowed_file(file.filename):
            img_file = request.files.get(file)
            # Secure image
            #img_name = werkzeug.utils.secure_filename(img_file.filename)
            # Store user image in temp folder
            img_url = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
            file.save(img_url)
            # Rotate cellphone image (if needed)
            img_file = rotate_image(img_url)
            print('Image Rotated')
            # Process image for model input
            #img = fetch_web.vectorize_image(img_name, model) #change to img_file when uncommenting rotate function
            # Calculate similarity
            #results = fetch_web.similarity(vector_list,dog_vector)
                          
            #load image for processing through the model
            img = kimage.load_img(img_url, target_size=(224, 224))
            img = kimage.img_to_array(img)
            img = np.expand_dims(img, axis=0)  
                        
            global graph
            with graph.as_default():
                pred=model.predict(img)
            matches = top_matches(pred, feature_matrix, vector_list['imgfile'])
            
            results = vector_list.set_index('imgfile', drop=False).join(matches.set_index('imgfile'))
            results.sort_values(by='simscore', ascending=True, inplace=True)

            original_url = img_file #formerly img_name
            return flask.render_template('results.html', matches=results, original=original_url)
        flask.flash('Upload only image files')

        
        return flask.redirect(flask.request.url)


'''
def vectorize_image(image, model):
    dog = load_img(image, target_size=(224, 224))
    image_batch = np.expand_dims(img_to_array(dog), axis=0)  
    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    #np.save('../data/user_files/feature_vec_'+image_name.split('.')[0], predictions)            
    return predictions

def similarity(vector_list,predictions):
    labels = []
    for vector in vector_list:
        labels.append(vector[12:].split('.')[0]+'.jpg')  
    score = distance.cdist(predictions_a, feature_matrix, 'cosine').tolist()
    if len(labels) != len(score[0]):
        print('Length mismatch!')
    sorted_scores = sorted(list(zip(labels,score[0])), key = lambda t: t[1])
    return sorted_scores
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=False)