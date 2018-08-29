

ef vectorize_dog_images(image_path_list, length=25):
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