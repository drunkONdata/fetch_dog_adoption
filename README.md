# Fetch! the dog adoption app
A dog adoption application, where a user submits a picture of a dog and curated list of similar looking dogs that are available for adoption nearby. 

## Table of Contents
1. [Motivation](#motivation)
2. [Product](#product)
3. [Gathering and Cleaning Data](#gathering-and-cleaning-data)
4. [Data Preparation](#data-preparation)
5. [Modeling](#modeling)
6. [Usage](#usage)
7. [Future Work](#future-work)
8. [References](#references)
9. [License](#license)

## Motivation
According to the ASPCA, over 1.7 million pets are euthanized per year. A good potion of which are healthy enough to be 
be rehomed. Besides the moral case, costs of euthansia range between $500-$700 per pet according to the American Veterinary 
Medical Association (AVMA). This is a potential loss of $1.9 billion dollars for taxpayers who foot the bill for these costs.
Fetch! is on a mission to make it easier for people to find their canine best friends.

## Product
The interface for Fetch allows a user to upload an image of a dog and recieve Top 10 most similar looking dogs available for 
adoption. By default, location is not enabled for privacy reasons. Once turned ON, location is determined via IP address or 
by GPS coordinates extracted from the image's EXIF data. The hope is that Fetch will increase the number of adoptions by 
streamlining the adoption process. 

The matches are determined by a cosine similarity distance metric between a vectorized user submitted image and a feature 
matrix of all vectorized images of adopted dogs. Images are vectorized by a modified VGG16 convoluted neural network and 
ImageNet weights loaded. 

## Gathering & Cleaning Data

Data was gathered from RescueGroups.org as JSON files. 

* Repository Name
* Owner
* Name With Owner
* Disk Usage
* Project's Url,
* SSH Url,
* Fork Count
* Star Count
* Watcher Count

After the repository metadata is collected I use [GitPython](https://github.com/gitpython-developers/GitPython) to clone
the repository to an Amazon AWS EC2 instance. Non-Python files are deleted to save space, since the RNN is only trained on
and predicts Python code.

In total, the image corpus consisted of 14.3Gb of data totaling 146,185 images of 48,784 dogs.

## Data Preparation
The RNN is trained on a sequence of 100 characters. Training was performed on an [g2.8xlarge AWS EC2 instance](https://aws.amazon.com/ec2/instance-types/),
with 4 NVIDIA GRID K520 GPUs. This allowed me to train the RNN on batches of 512 of these sequences at a time.

During training, batches are generated as follows:

1. Read the next .py file from the collection of GitHub repositories
2. Encode the each character as a one-hot encoded vector of length 100, since I limited the allowable characters to 
   python's strings.printable list of 100 characters. For example, the character 'a' is encoded as a 100 dimensional vector
   with all 0's with the exception of the 11th component which is 1.
3. 100 characters are assembled into a numpy array, resulting in a feature matrix of shape 100 x 100.
4. The target for this 100 x 100 feature matrix is the encoded 101st character
5. Because of memory limitations. A batch of 512 sequences and targets are prepared for each epoch.

## Modeling

The modified VGG16 model consists of an input layer were a 224x224 image is inputted 4 fully connected layers of 512 LSTM units each. Between each LSTM layer are layers of 512 dropout nodes
with a dropout probability of 0.2 for regularization, to prevent overfitting. Finally there is a layer of 100 nodes, one for each character in the vocabulary. Finally there is a softmax node to create a probability distribution of the characters. 
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
=================================================================
Total params: 117,479,232
Trainable params: 117,479,232
Non-trainable params: 0
_________________________________________________________________
```



## Usage

Clone this repository with the following command:
```
git clone https://github.com/drunkONdata/fetch_dog_adoption.git
```
The repository has the following file structure. 
```
.
├── LICENSE
├── pycc.sh
├── pycodecomplete
│   ├── ml
│   │   ├── code_generation.py
│   │   ├── make_model.py
│   │   ├── process_text.py
│   │   └── rnn.py
│   ├── scraping
│   │   ├── __init__.py
│   │   └── scrape_github.py
│   ├── tests
│   └── trained-models
├── README.md
├── requirements.txt
├── setup.py
└── webapp
    ├── app.py
    ├── config.py
    ├── requirements.txt
    ├── static
    │   ├── css
    │   │   └── main.css
    │   ├── favicon.ico
    │   └── js
    │       ├── app.js
    │       └── jquery-3.3.1.js
    └── templates
        └── template.html
```
Recreating the corpus from GitHub requires a [GitHub personal access token](https://developer.github.com/v3/auth/#basic-authentication). Create a token and save the token file to your local computer.  

Run the following command to generate the corpus with 1000 Python repos:
```
python ./pycodecomplete/scraping/scrape_github.py -f /path/to/github/token /cloned/repo/destination/path 1000
```

Once the script has completed cloning the repos, deleting unnecessary files and cleaning the .py file, you can start training a new RNN model with the following command:
```
python ./pycodecomplete/ml/make_model.py /path/to/save/pickled/models /path/to/cloned/repos 100 512 1 4 512 20 6000 1
```
The arguments are:
1. Path to save serialized RNN models. A trained model is saves after the completion of each epoch.
2. Path to the cloned GitHub repositories from which to train the model on
3. Sequence length (100 character long sequence)
4. Number of layers (4 layers of LSTM nodes)
5. Number of nodes per layer (512 nodes per layer)
6. Number of Epochs to train
7. Number of steps per Epoch (Should be Number of total characters in the corpus divided by batch size)
8. Max Queue Size (Number of batches to queue in RAM)

Additionally you can continue training an existing model with the addition of the -m option:
```
-m /path/to/existing/model
```
and train on a number of GPUs with the -g option:
```
-g 4
```
for a computer with 4 GPUs

Finally once a model is complete you can start the flask app with the command:
```
./pycc.sh
```

## Future Work


## Tech Stack

NumPy, SciPy, Pandas, AWS, S3, EC2, VisPy, Flask, Keras, TensorFlow, ImageNet
<img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width="250">
<img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" width="250">
<img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/3428/media/scikit-learn-logo.png" width="250">
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Logo_of_NumPy.svg/1200px-Logo_of_NumPy.svg.png" width="250">
<img src="https://pandas.pydata.org/_static/pandas_logo.png" width="250">
<img src="https://cdn-images-1.medium.com/max/1600/1*AD9ZSLXKAhZ-_WomszsmPg.png" width="250">
<img src="https://camo.githubusercontent.com/630f51296667710aa4dd5959ec5cbc9c03bd48ac/687474703a2f2f7777772e6168612e696f2f6173736574732f6769746875622e37343333363932636162626661313332663334616462303334653739303966612e706e67" width="250">
<img src="https://cdn-images-1.medium.com/max/2000/1*49DDRZhUWvVnH-QNHuSUSw.png" width="250">
<img src="http://flask.pocoo.org/static/logo/flask.png" width="250">

## References


## License
MIT License

Copyright (c) 2018 Abhi Banerjee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
