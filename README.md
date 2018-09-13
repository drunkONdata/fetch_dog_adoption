# Fetch! Dog Adoption
Fetch is on a mission to make it easier for people to find their canine best friend. Utilizing a modified neural network, the Fetch app 
searches for dogs to adopt that closely resemble a user submitted photo. By tapping into existing records exceeding 150k+ images 
of 45k+ dogs, Fetch provides users a curated list of the top 25 most similar dogs for adoption. 

## Table of Contents
1. [Motivation](#motivation)
2. [Product Details](#product)
3. [Data Preparation](#data-preparation)
4. [Model](#model)
5. [Usage](#usage)
6. [Future Work](#future-work)
7. [License](#license)

## Motivation
According to the ASPCA, over 1.7 million pets are euthanized per year. A good portion of which are healthy enough to be 
be rehomed. Besides the moral case, costs of euthansia range between $500-$700 per pet according to the American Veterinary 
Medical Association (AVMA). This is a potential loss of $1.2 billion dollars for taxpayers who foot the bill for these costs.

## Product Details
Fetch allows a user to upload an image of a dog and recieve a list of the most similar looking dogs available for adoption. By default, 
location is not enabled for privacy reasons. Once turned 'ON', location is determined via IP address or by GPS coordinates 
extracted from the image's EXIF data. The hope is that Fetch will increase the number of adoptions by streamlining access 
to adoption information. 

The matches are determined by a cosine similarity between a vectorized user submitted image and a feature matrix of all 
vectorized images of adopted dogs. Images are vectorized by a modified VGG16 convoluted neural network with ImageNet weights 
loaded. Cosine similarity is the distance metric of choice after recieving 380+ responses to a user validation survey which 
showed a 81.7% preference for it as apposed to eucleadian, manhattan, bray-curtis, chebyshev and hamming distance metrics.

## Data Preparation
Data was gathered from RescueGroups.org as JSON files. In total, the image dataset consisted of 14.3Gb of data totaling 
152,185 images of 48,784 dogs. Features of interest for our application included:

* AnimalID
* OrgID
* Name
* Breed
* animalLocation
* pictureUrl

## Model
The modified VGG16 model consists of an input layer were a 224x224x3 image is recieved, 13 Conv2D layers, 5 MaxPooling2D layers 
and a dense and flatten layer left at the end. The softmax classification layer and a fully connected layer in the original 
model was dropped for our purposes. This enabled us to vectorize a dog image down to a 1D array with a length of 4096 'features'. 

```
Fetch! Model Summary:
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
├

```

## Future Work
- Feature extraction optimization strategies:
    + Scale Invarient Feature Transform (SIFT) module from OpenCV
    + Gradient-weighted Class Activation Mapping (GradCAM) to identify most used features across all images
    + Add latent layer to neural network and train on targeted domain with dropout & guided back propagation
    + PCA and discriminative dimensionality reduction
    + Naive clustering by finding centroids of each class in original image space
- Investigate Faiss from Facebook Open Source for similarity modeling alternative
- Optimize image database performance for future scaling
- Investigate alternative distance metrics with further user validation
- Investigate performance with other models: InceptionV3, ResNet50, NASNet, MobileNet
- Improve web and smartphone app experience with React Native


## References
1. ****
2. 

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
