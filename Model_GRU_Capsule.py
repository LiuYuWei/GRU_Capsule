from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import os
import time
import gc
import re
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential, Model, model_from_json, load_model

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"

#define the variable
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
VALIDATION_SPLIT = 0.1

from Layer_Capsule import *
        
## Keras Model-capsule

def Keras_Model_GRU_Capsule(max_features,embedding_matrix,Y_num,embed_size=300,maxlen=MAX_SEQUENCE_LENGTH):
    
    #Input layer
    inp = Input(shape=(maxlen,))
    #Embedding Layer
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.2)(x)
    x = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, 
                                kernel_initializer=glorot_normal(seed=12300), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)
    x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
    x = Flatten()(x)
    x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12300))(x)
    x = Dropout(0.12)(x)
    x = BatchNormalization()(x)
    #output layer
    x = Dense(Y_num, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    #Adam = RMSProp+Momentum
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
    return model