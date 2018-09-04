import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import ExifTags, Image
from keras.applications import VGG16
from keras.preprocessing import image as keras_image

from scipy.spatial import distance

from numpy import inner
from numpy.linalg import norm

#from artmagic import app
#from artmagic.models.similarity import find_matches
from src.fetch_data_pipeline import rotate_image, similarity, top_matches
from src.fetch_data_processing import initialize_neural_network
from src.fetch_web import vectorize_image, similarity

# Set up filepath to store user submitted photo
UPLOAD_FOLDER = '/Users/bil2ab/galvanize/fetch_dog_adoption/web/static/temp/upload'
#DATA_FOLDER = '/Users/bil2ab/galvanize/fetch_dog_adoption/web/static/temp/data'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['DATA_FOLDER'] = DATA_FOLDER

# Load Data
feature_matrix = np.load('/data/fetch_feature_matrix.npy')
vector_list = pd.read_pickle('/data/fetch_vector_list.pkl', compression='gzip')
#collection_features = np.load(os.path.join(app.config['DATA_FOLDER'], 'fetch_feature_matrix.npy'))
#files_and_titles=pd.read_csv(os.path.join(app.config['DATA_FOLDER'], 'img_urls.csv'))                           

# Initialize neural network with classification layer and fully connected layer dropped
model = fetch_vectors.initialize_neural_network()

# Verify file extension of user submitted photo    
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#Art Magic MVP find_matches
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
            img_name = secure_filename(img_file.filename)
            # Store user image in temp folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            # Rotate cellphone image (if needed)
            img_file = rotate_image(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            print('Image Rotated')
            # Process image for model input
            dog_vector = fetch_web.vectorize_image(img_file, model)
            # Calculate similarity
            results = fetch_web.similarity(vector_list,dog_vector)

                  
            
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
        
       
            
            
            
            
              
            
            

            


            #original = keras_image.load_img(imgurl, target_size=(224, 224))
            #np_img = keras_image.img_to_array(original)
            #img_batch = np.expand_dims(np_img, axis=0)
            #processed_image = vgg16.preprocess_input(img_batch.copy())           
            
            global graph
            #graph = tf.get_default_graph()
            with graph.as_default():
                #pred = model.predict(processed_image)
            results = similarity(pred, collection_features)
            top_10_matches = top_matches(results, files_and_titles[1], 10) #files_and_titles['imgfile'])
            
            #showresults=files_and_titles.set_index('imgfile',drop=False).join(matches.set_index('imgfile'))
            #showresults.sort_values(by='simscore',ascending=True,inplace=True)

            original_url = img_name
            return flask.render_template('results.html', matches = results[0:11], original = top_10_matches[0])
        flask.flash('Upload only image files')

        
        return flask.redirect(flask.request.url)

#app.secret_key = 'adam'

#I was getting an error because the model was losing track of the graph
#defining graph here lets me keep track of it later as things move around
#

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=False)