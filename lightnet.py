
from keras.layers import Conv2D,Conv1D,BatchNormalization,ReLU,Dense,DepthwiseConv2D,Input\
                                                        ,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D,Reshape,concatenate,multiply,Dropout,Lambda
from keras.models import Model,load_model,save_model
from keras.initializers import HeNormal
from tensorflow.keras import backend
from keras.regularizers import L2
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from math import log2
drive.mount('/content/drive')



(xdata,ydata),(_,_)= cifar10.load_data()
xdata = xdata[15000:20000].astype('float32')/255.0


xdata = np.array([tf.image.resize(img,(224,224)) for img in xdata])


ydata = to_categorical(ydata[15000:20000],10)

data_gen = ImageDataGenerator(rotation_range=20,zoom_range=0.1,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15\
                              ,horizontal_flip=True,vertical_flip=True,validation_split=0.2)   #data agumentation

train_gen = data_gen.flow(xdata,ydata,shuffle=True,batch_size=128,subset='training') #train set

valid_gen = data_gen.flow(xdata,ydata,shuffle=False,batch_size=128,subset='validation') # validation set

def depthwise_sep_conv(input,filters,stride,attention=False):
    # depth-wise-convolution
    x= DepthwiseConv2D(kernel_size=3,strides=stride,padding='same',depthwise_initializer=HeNormal())(input)
    x = BatchNormalization(axis=-1,momentum=0.99)(x)
    x = ReLU()(x)
    # point-wise-convolution
    y = Conv2D(filters,kernel_size=(1,1),strides=1,padding='same', kernel_initializer=HeNormal())(x)
    y = BatchNormalization(axis=-1,momentum=0.99)(y)
    y = ReLU()(y)

    if attention is True: # Effi-channel-attention
        beta = 1
        gamma = 2
        input_channels = backend.int_shape(y)[-1] # get the channels from the input
        avg_pooling = GlobalAveragePooling2D()(y)
        avg_pooling = Reshape((1,input_channels))(avg_pooling)
        k_size = int(abs((log2(input_channels)/gamma)+(beta/gamma))) #adaptive kernal
        k = k_size if k_size%2 else k_size+1
        conv1d = Conv1D(filters=input_channels,kernel_size=k,padding='same',activation='sigmoid', kernel_initializer=HeNormal())(avg_pooling)
        attention_vector = Reshape((1,input_channels))(conv1d)
        return multiply([y,attention_vector])
    else:
        return y






def model():
    layer_0 = Input((224,224,3))
    layer_1 = depthwise_sep_conv(layer_0,32,2)
    layer_2 = depthwise_sep_conv(layer_1,64,2)
    layer_3 = depthwise_sep_conv(layer_2,128,2,True)
    layer_4 = depthwise_sep_conv(layer_3,256,2,True)
    layer_5 = depthwise_sep_conv(layer_4,512,2,True)
    layer_6 =GlobalAveragePooling2D()(layer_5)
    #dense = Dense(64,activation=ReLU(),kernel_regularizer=L2(0.1), kernel_initializer=HeNormal())(layer_6)
    dense = Dropout(0.5)(layer_6)
    oplayer = Dense(10,activation='softmax',kernel_regularizer=L2(0.01), kernel_initializer=HeNormal())(dense)
    model = Model(inputs=layer_0, outputs=oplayer)
    return model


model().summary(show_trainable=True)

model = model()

#reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,cooldown=2, min_lr=0.0000001)
path = '/content/drive/MyDrive/light5.weights.h5'
ckpt = ModelCheckpoint(filepath=path,save_weights_only=True)
try:
    model = load_model('/content/drive/MyDrive/light5.h5',custom_objects={'ReLU': ReLU})
    model.load_weights(path)
except Exception as e:
    print("Error loading model:", e)
    exit()
model.compile(optimizer=Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
csv_log = CSVLogger('/content/drive/MyDrive/light5.csv',append=True)
mixed_pre_policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(mixed_pre_policy)

history = model.fit(train_gen,validation_data=valid_gen,epochs=40,callbacks=[ckpt,csv_log])
model.save('/content/drive/MyDrive/light5.h5')
print(model.history.history.keys())

plt.plot(model.history.history["accuracy"], label="accuracy")
plt.plot(model.history.history["val_accuracy"], label="val_acc")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





plt.plot(model.history.history["loss"], label="loss")
plt.plot(model.history.history["val_loss"], label="val_loss")

log = '/content/drive/MyDrive/light5.csv'
import pandas as pd
df = pd.read_csv(log)
df.head()

plt.plot(df["accuracy"], label="accuracy")
plt.plot(df["val_accuracy"], label="val_acc")
plt.plot(df["loss"], label="loss")
plt.plot(df["val_loss"], label="val_loss")



