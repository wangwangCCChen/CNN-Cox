from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,Dropout,Input,Flatten,concatenate
from tensorflow.keras.regularizers import l2

def cnncox(conv1=128,conv1_size=(1, 10),conv2=32,conv2_size=(10, 1),dense=64,input_shape = (10, 10, 1)):
    input_img = Input(input_shape)
    
    tower_1 = Conv2D(conv1, conv1_size, activation='relu')(input_img)
    tower_1 = MaxPooling2D(1, 2)(tower_1)
    tower_1 = Flatten()(tower_1)

    tower_2 = Conv2D(conv2, conv2_size, activation='relu')(input_img)
    tower_2 = MaxPooling2D(1, 2)(tower_2)
    tower_2 = Flatten()(tower_2)

    output = concatenate([tower_1, tower_2], axis=1)
    out1 = Dense(dense, activation='relu')(output)
    last_layer = Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(out1)
    model = Model(inputs=[input_img], outputs=last_layer)
    return model

def dcnncox(conv1=128,conv1_size=(1, 10),dense=64,input_shape = (10, 10, 1)):

    input_img = Input(input_shape)
    
    tower = Conv2D(conv1, conv1_size, activation='relu')(input_img)
    tower1 = MaxPooling2D(1, 2)(tower)
    tower2 = Flatten()(tower1)

    out = Dense(dense, activation='relu')(tower2)
    last_layer = Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(out)
    model = Model(inputs=[input_img], outputs=last_layer)
    return model


def nncox(dense1=64,dense2=64,input_shape = (100,)):

    input_d = Input(input_shape)
    
    tower1 = Dense(dense1,activation='relu',kernel_initializer='glorot_uniform')(input_d)
    tower2 = Dense(dense2,activation='relu',kernel_initializer='glorot_uniform')(tower1)
    tower3 = Dropout(0.5)(tower2)
    last_layer = Dense(1,activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01))(tower3)
    model = Model(inputs=[input_d], outputs=last_layer)
    return model

def cnncoxclin(conv1=128,conv1_size=(1, 10),conv2=32,conv2_size=(10, 1),dense=64,input_shape = (10, 10, 1),input2_shape = (3,)):

    input_img = Input(input1_shape)
    input_clin = Input(input2_shape)
    
    tower_1 = Conv2D(conv1, conv1_size, activation='relu')(input_img)
    tower_1 = MaxPooling2D(1, 2)(tower_1)
    tower_1 = Flatten()(tower_1)

    tower_2 = Conv2D(conv2, conv2_size, activation='relu')(input_img)
    tower_2 = MaxPooling2D(1, 2)(tower_2)
    tower_2 = Flatten()(tower_2)

    output = concatenate([tower_1, tower_2], axis=1)
    out1 = Dense(dense, activation='relu')(output)
    out2 = concatenate([out1, input_clin], axis=1)
    last_layer = Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(out2)
    model = Model(inputs=[input_img,input_clin], outputs=last_layer)

    return model

def dcnncoxclin(conv1=128,conv1_size=(1, 10),dense=64,input_shape = (10, 10, 1),input2_shape = (3,)):

    input_img = Input(input_shape)
    input_clin = Input(input2_shape)
    
    tower = Conv2D(conv1, conv1_size, activation='relu')(input_img)
    tower1 = MaxPooling2D(1, 2)(tower)
    tower2 = Flatten()(tower1)
    out = Dense(dense, activation='relu')(tower2)
    out1 = concatenate([out, input_clin], axis=1)
    last_layer = Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(out1)
    model = Model(inputs=[input_img,input_clin], outputs=last_layer)
  
    return model


def nncoxclin(dense1=64,dense2=64,input_shape = (100,),input2_shape = (3,)):

 
    input_d = Input(input_shape)
    input_clin = Input(input2_shape)
    
    tower1 = Dense(dense1,activation='relu',kernel_initializer='glorot_uniform')(input_d)
    tower2 = Dense(dense2,activation='relu',kernel_initializer='glorot_uniform')(tower1)
    tower3 = Dropout(0.5)(tower2)
    out = concatenate([tower3, input_clin], axis=1)
    last_layer = Dense(1,activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01))(out)
    model = Model(inputs=[input_d,input_clin], outputs=last_layer)

    return model