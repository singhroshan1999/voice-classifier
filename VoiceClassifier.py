# importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
INP_CONST = 882

def bytes_from_file(filename, chunksize=8192):
    """ Read RAW Audio file and return generator function
        Parameter(s): filename --> name of file to read
                      chunksize --> number of bytes to read at a time
        Returns: generator object
    """
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break
def create_audio_data(l):
    """Reads list of RAW Audio file(s) and returns DataFrame"""
    datArr = []
    for i in l:
        t_arr = np.array([x for x in bytes_from_file(i)])
        datArr.append(t_arr[:t_arr.size - t_arr.size%INP_CONST])
    datArr = np.concatenate(datArr)
    datArr = datArr.reshape((datArr.size//INP_CONST,INP_CONST),order = 'C')
    return pd.DataFrame(datArr),datArr.shape[0]

def create_y_sequence(n_data,div):
    """Generate label for trainig"""
    onehotencoder = OneHotEncoder(categorical_features = [0])
    n = n_data//div  # check for float
    l = []
    for i in range(div):
        l.append(np.ones(n)*i)
    fet = pd.DataFrame(np.concatenate(l))
    return onehotencoder.fit_transform(fet).toarray()

def train_data(x,y,units,batch_size = 100,epochs = 100,verbose = 0):
    """Train Artifitial Neural Networks
    Parameter(s): x --> RAW audio data
                  y --> Label
                  units --> number of output of ANN
                  batch_size --> size of batch (default = 100)
                  epoch --> epoch (default = 100)
                  verbose --> verbosity (default = 0) 0 --> none | 1 --> only epochs with progressbar | 2 --> only epoch
    Return: tuple (classifier,sc) --> classifier --> keras model object
                                      sc --> StandardScalar object
    """
    sc = StandardScaler()
    x = sc.fit_transform(x)
    # initializing ANN
    classifier = Sequential()
    # adding input and hidden layers
    classifier.add(Dense(units = 50,use_bias = True, kernel_initializer = 'random_normal', activation = 'relu', input_dim = INP_CONST))  # first hidded layer
    classifier.add(Dense(units = 40,use_bias = True, kernel_initializer = 'random_normal', activation = 'relu'))  # second hidded layer
    classifier.add(Dense(units = 30,use_bias = True, kernel_initializer = 'random_normal', activation = 'relu'))  # second hidded layer
    classifier.add(Dense(units = 20,use_bias = True, kernel_initializer = 'random_normal', activation = 'relu'))  # second hidded layer
    classifier.add(Dense(units = units,use_bias = True, kernel_initializer = 'random_normal', activation = 'sigmoid'))  # output layer
    # compiling ANN
    classifier.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(x, y, batch_size = batch_size, epochs = epochs,verbose = verbose)
    return classifier,sc
    
def train(file_list,name_list = [],verbose = 0,epochs = 100):
    """Train every file in file_list with name of each voice
       Parameter(s): file_list --> list of file as on disk
                     name_list --> list of corrorsponing name of voice
       Return(s): (classifier,sc,name_list) --> classifier --> keras ANN model
                                                sc --> StandardScalar object
                                                name_list --> name_list
    """
    x,size = create_audio_data(file_list)
    y = create_y_sequence(size,5)
    classifier = train_data(x,y,len(file_list),verbose = verbose,epochs = epochs)
    return classifier[0],classifier[1],name_list

def predict(filename,classifier,scalar):
    """predict voice of given file
       Parameter(s): filename --> as on disk
                     classifier --> train() classifier
                     scalar --> train() sc
       Return(s): Datafram of predicted values and size tuple"""
    df,size = create_audio_data([filename])
    predicted_value = classifier.predict(scalar.transform(df))
    return pd.DataFrame(predicted_value),size

def predict_names(filename,train):
    """returns percent predicted values with name of voice
       Parameter(s): filename --> as on disk
                     train --> tuple returned by train()
       Return(s): Dataframe of percent prediction and name"""
    predicted,sum = predict(filename,train[0],train[1])
    return pd.concat([pd.Series(train[2]),pd.Series([x/(sum) for x in  predicted.sum()])],axis = 1)