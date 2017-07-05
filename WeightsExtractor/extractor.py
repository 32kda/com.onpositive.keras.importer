'''
Created on Mar 24, 2017

@author: Sviridov
'''
# Create first network with Keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy
import h5py
import os
from keras.layers.recurrent import LSTM

def train():
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.add(LSTM(1000));
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X, Y)
    model.save_weights("sample_weights.h5")
    model.save("model.h5")
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model
    
def save_to_txt(name):
    layers = []
    f = h5py.File(name,"r")
    for i, key in enumerate(list(f.keys())):
        layer_weights = {}
        for value in f[key].values():
            for childValue in value.values():
                print(childValue)
                splitted = childValue.name.split('/');
                layer_weights[splitted[-2] + "_" + splitted[-1]] = numpy.array(childValue)        
                print (childValue.name, numpy.array(childValue).shape)
        layers.append(layer_weights)
    
    if not os.path.exists("weights"):
        os.makedirs("weights")
    
    for i, l in enumerate(layers):
        for key in l.keys():
            #numpy.savetxt(os.path.join("weights",  str(i) + "_" + key +".txt"), l[key])
            numpy.savetxt(os.path.join("weights", key +".txt"), l[key])
            
def test():
    model = load_model("model.h5")
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:,0:8]
    predicted = model.predict(X)
    if not os.path.exists("predicted"):
        os.makedirs("predicted")
    numpy.savetxt(os.path.join("predicted",  "python_result.txt"), predicted)

save_to_txt("weights2.h5")