#%%
#Imports
import gzip
import pickle
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import gzip
import pickle
#%matplotlib inline
import numpy as np
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Activation
import pandas as pd
import random
import time
import keras.backend as K
from keras import losses
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import numpy.ma as ma

#%% mask
def masked_mse(y_true,y_pred):
    nanval = 0
    isMask = K.equal(y_true,nanval)
    isMask = 1 - K.cast(isMask,dtype=K.floatx())
    y_true = y_true*isMask
    y_pred = y_pred*isMask
    return (losses.mean_squared_error(y_true,y_pred))

#Initialisation
fname = "base_hiver_2008.pklgz"
with gzip.open(fname,"rb") as fp:
    dictio = pickle.load(fp,encoding='latin1')
    
imind=0
SSTMW = dictio['SSTMW']
plt.imshow(SSTMW[imind], cmap='jet')
plt.colorbar()
plt.show()

#%%
#newimages extracts 4 images from each initial image
def newimages(images):
    tmp = images
    image1 = tmp[0:64,0:64]
    image2 = tmp[tmp.shape[0]-65:tmp.shape[0]-1,0:64]
    image3 = tmp[0:64,tmp.shape[1]-65:tmp.shape[1]-1]
    image4 = tmp[tmp.shape[0]-65:tmp.shape[0]-1,tmp.shape[1]-65:tmp.shape[1]-1]
    images_new = [image1,image2,image3,image4]
    return images_new

#Function make_hole returns the images given with a hole inside
def make_hole(image, size, position):
    img=image.copy()
    #moy=np.zeros(img.shape[0])
    #for i in range(image.shape[0]):
    #    moy[i]=image[i].mean()
    for j in range(size):
        for k in range(size):
            img[position[0]+j,position[1]+k]=0;
    return img;

#%%
#Function to create the model
def get_model():
    inputs = Input(shape=(64, 64, 1))
    conv_1 = Conv2D(25, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    conv_2 = Conv2D(15, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    conv_3 = Conv2D(10, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    deconv_1 = Conv2DTranspose(10, (3, 3), strides=(2, 2), padding='same')(pl_3)
    dact_1 = Activation('relu')(deconv_1)
    merge_1 = concatenate([dact_1, act_3], axis=3) 
    deconv_2 = Conv2DTranspose(15, (3, 3), strides=(2, 2), padding='same')(merge_1)
    dact_2 = Activation('relu')(deconv_2)
    merge_2 = concatenate([dact_2, act_2], axis=3)
    deconv_3 = Conv2DTranspose(25, (3, 3), strides=(2, 2), padding='same')(merge_2)
    dact_3 = Activation('relu')(deconv_3)
    merge_3 = concatenate([dact_3, inputs], axis=3)
    final = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(merge_3)
    dact_4 = Activation('relu')(final)
    
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    
    model = Model(inputs=[inputs], outputs=[dact_4])
    model.compile(loss='mean_squared_error', optimizer=adadelta)
    return model

def get_discriminator():
    inputs = Input(shape=(64,64,2))
    conv_1 = Conv2D(25, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    conv_2 = Conv2D(15, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    conv_3 = Conv2D(10, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    deconv_1 = Conv2DTranspose(10, (3, 3), strides=(2, 2), padding='same')(pl_3)
    dact_1 = Activation('relu')(deconv_1)
    merge_1 = concatenate([dact_1, act_3], axis=3) 
    deconv_2 = Conv2DTranspose(15, (3, 3), strides=(2, 2), padding='same')(merge_1)
    dact_2 = Activation('relu')(deconv_2)
    merge_2 = concatenate([dact_2, act_2], axis=3)
    deconv_3 = Conv2DTranspose(25, (3, 3), strides=(2, 2), padding='same')(merge_2)
    dact_3 = Activation('relu')(deconv_3)
    merge_3 = concatenate([dact_3, inputs], axis=3)
    final = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(merge_3)
    dact_4 = Activation('relu')(final)
    
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    
    model = Model(inputs=[inputs], outputs=[dact_4])
    model.compile(loss='mean_squared_error', optimizer=adadelta)
    return model

#%%
#Creating the data
Images_1=newimages(SSTMW)
Images_1 = [ newimages(SSTMW[i]) for i in range(0,SSTMW.shape[0]) ]
Images_1 = np.asarray(Images_1)
Images_1 = Images_1.reshape(86*4,64,64)

means=np.zeros(Images_1.shape[0])
stds=np.zeros(Images_1.shape[0])
for i in range(Images_1.shape[0]):
    means[i]=Images_1[i].mean()
    stds[i]=Images_1[i].std()
    for j in range(64):
        for k in range(64):
            Images_1[i,j,k]=(Images_1[i,j,k]-means[i])/stds[i]+3

Holes=Images_1.copy()
Holes = make_hole(Images_1, 10, (10,10))
 
xtrain, xtest, ytrain, ytest = train_test_split(Holes, Images_1, test_size=0.2)
xtrain = np.reshape(xtrain, (xtrain.shape[0], 64, 64,1))
xtest = np.reshape(xtest, (xtest.shape[0], 64, 64,1))
ytrain = np.reshape(ytrain, (ytrain.shape[0], 64, 64,1))
ytest = np.reshape(ytest, (ytest.shape[0], 64, 64,1))

#%%
#Test with no hole
model = get_model()
model.fit(ytrain, ytrain, epochs=40,batch_size=10,shuffle=True)

Outputs = model.predict(ytest)
Outputs = Outputs.reshape(Outputs.shape[0],64,64)

plt.figure()
plt.imshow(xtest[1,:,:,0],vmin=-2.5+3, vmax=2.5+3, cmap='jet')
plt.colorbar()

plt.figure()
plt.imshow(Outputs[1])
plt.colorbar()

#%%
#Test with one hole
model = get_model()
model.fit(xtrain, ytrain, epochs=40,batch_size=10,shuffle=True)

Outputs = model.predict(xtest)
Outputs = Outputs.reshape(Outputs.shape[0],64,64)

plt.figure()
plt.imshow(xtest[1,:,:,0])
plt.colorbar()

plt.figure()
plt.imshow(Outputs[1])
plt.colorbar()

plt.figure()
plt.imshow(ytest[1,:,:,0])
plt.colorbar()

#%%
#Function to do multiple holes in the image
#Needs upgrade for different holes
import matplotlib as m
cdict = {
  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
}

cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

def multipleholes(image, sizes, number):
    zero=np.zeros((344*3,64,64))
    for a in range(3):
        for j in range(image.shape[0]):
            zero[j+a*344]=image[j].copy()
            for i in range(number):
                size=random.randint(sizes[0],sizes[1])
                zero[j+a*344] = make_hole(zero[j+a*344], size, (random.randint(0,63-size),random.randint(0,63-size)))
    return zero
Multiple=multipleholes(Images_1, [1,10], 100)
plt.figure()
plt.imshow(Multiple[0],vmin=-2.5+3, vmax=2.5+3, cmap='jet')#cmap=plt.get_cmap('nipy_spectral'))
plt.colorbar()
plt.figure()
plt.imshow(Multiple[344],vmin=-2.5+3, vmax=2.5+3, cmap='jet')#cmap=plt.get_cmap('nipy_spectral'))
plt.colorbar()
plt.figure()
plt.imshow(Multiple[344*2],vmin=-2.5+3, vmax=2.5+3, cmap='jet')#cmap=plt.get_cmap('nipy_spectral'))
plt.colorbar()

def imgx3(images):
    img=np.zeros((344*3,64,64))
    for i in range(3):
        for j in range(344):
            img[j+i*344]=images[j]
    return img
Images=imgx3(Images_1)

#%%Tests
import pylab
Multiple_test=multipleholes(Images_1, [1,10], 100)
plt.figure()
plt.imshow(Multiple_test[0]*stds[0]+means[0]-3, cmap='seismic',vmin=19,vmax=23)
plt.xlabel('p')
plt.ylabel('p')
cbar=plt.colorbar()
cbar.set_label('Temperature', rotation=270)

plt.figure()
plt.imshow(Multiple_test[344]*stds[0]+means[0]-3, cmap='seismic',vmin=19,vmax=23)
plt.colorbar()

plt.figure()
plt.imshow(Multiple_test[344*2]*stds[0]+means[0]-3, cmap='seismic',vmin=19,vmax=23)
plt.colorbar()

#%%
#Test with multiple holes
x_train, xtest, y_train, ytest = train_test_split(Multiple, Images, test_size=0.2)
xtrain, xval, ytrain, yval = train_test_split(x_train, y_train, test_size=0.2)
xtrain = np.reshape(xtrain, (xtrain.shape[0], 64, 64,1))
xtest = np.reshape(xtest, (xtest.shape[0], 64, 64,1))
xval = np.reshape(xval, (xval.shape[0], 64, 64,1))
ytrain = np.reshape(ytrain, (ytrain.shape[0], 64, 64,1))
ytest = np.reshape(ytest, (ytest.shape[0], 64, 64,1))
yval = np.reshape(yval, (yval.shape[0], 64, 64,1))

model = get_model()
model.fit(xtrain, ytrain, epochs=40,batch_size=16,shuffle=True,validation_data=(xval, yval))

Outputs = model.predict(xtest)
Outputs = Outputs.reshape(Outputs.shape[0],64,64)

plt.figure()
plt.imshow(xtest[7,:,:,0])
plt.colorbar()

plt.figure()
plt.imshow(Outputs[7,:,:,0])
plt.colorbar()

plt.figure()
plt.imshow(ytest[7,:,:,0])
plt.colorbar()

#%%
#Discriminator model
def get_discriminator():
    inputs = Input(shape=(64, 64, 2))
    conv_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    conv_2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    conv_3 = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    fc=Flatten()(pl_3)
    fc_2=Dense(40)(fc)
    act_4=Activation('relu')(fc_2)
    fc_3=Dense(25)(act_4)
    act_5=Activation('relu')(fc_3)
    fc_4=Dense(10)(act_5)
    act_6=Activation('relu')(fc_4)
    fc_5=Dense(1)(act_6)
    act_7=Activation('sigmoid')(fc_5)
    
    rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    model = Model(inputs=[inputs], outputs=[act_7])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%%
#Test of the discriminator model
Input_test=np.ones((Outputs.shape[0],64,64,2))
Output_wished=np.ones(Outputs.shape[0])
ytest=np.reshape(ytest, (ytest.shape[0],64,64))
for i in range(Outputs.shape[0]):
    a=random.randint(0,1)
    if (a==1):
        Input_test[i,:,:,0]=Outputs[i].copy()
        Input_test[i,:,:,1]=ytest[i].copy()
    else:
        Input_test[i,:,:,1]=Outputs[i].copy()
        Input_test[i,:,:,0]=ytest[i].copy()
    Output_wished[i]=a

x_train2, xtest2, y_train2, ytest2 = train_test_split(Input_test, Output_wished, test_size=0.2)
xtrain2, xval2, ytrain2, yval2 = train_test_split(x_train2, y_train2, test_size=0.2)
xtrain2 = np.reshape(xtrain2, (xtrain2.shape[0], 64, 64, 2))
xtest2 = np.reshape(xtest2, (xtest2.shape[0], 64, 64, 2))
xval2 = np.reshape(xval2, (xval2.shape[0], 64, 64, 2))
ytrain2 = np.reshape(ytrain2, (ytrain2.shape[0]))
ytest2 = np.reshape(ytest2, (ytest2.shape[0]))
yval2 = np.reshape(yval2, (yval2.shape[0]))
       
    
model2 = get_discriminator()
model2.fit(xtrain2, ytrain2, epochs=40,batch_size=8,shuffle=True,validation_data=(xval2, yval2))
#model2.save("discriminator.h5")
#model3=load_model("discriminator.h5")

#%% Tests on masks
def nearby_hole(y_true,y_pred,distance):
    mask=np.zeros((64,64))
    for i in range(64):
        for j in range(64):
            if (y_true[i,j]==0):
                for i2 in range(i-distance,i+distance+1):
                    for j2 in range(j-distance,j+distance+1):
                        if (i2>=0 and i2<y_true.shape[0] and j2>=0 and j2<y_true.shape[1]):
                            if(y_true[i2,j2]!=0):
                                mask[i2,j2]=1
                            else:
                                mask[i2,j2]=0.4
    return (mask)

def masked_mse(y_true,y_pred):
    nanval = 0
    isMask = nearby_hole(y_true,y_pred,4)
    isMask = 1 - K.cast(isMask,dtype=K.floatx())
    y_true = y_true*isMask
    y_pred = y_pred*isMask
    return (losses.mean_squared_error(y_true,y_pred))
M=nearby_hole(Multiple[0],Images[0],distance=2)
x = ma.masked_array(Images[0], mask=M, hard_mask=True)

# loss function
def penalized_loss(x,weight_hole=20,weight_ol=6):
    def loss(y_true, y_pred):
        zeronan = 0
        isMask = K.equal(x,zeronan) # mask in the region with hole (integer)
        isMask_square = K.cast(isMask,dtype=K.floatx()) # mask for the pixel where the hole is
        isMask_out = 1 - isMask_square # mask for the pixels not considering the hole
        loss_square = K.mean(K.square(y_true*isMask_square-y_pred*isMask_square)) # inside the hole
        loss_out = K.mean(K.square(y_true*isMask_out-y_pred*isMask_out)) # outside without hole
        return loss_square*weight_hole + loss_out #  loss outputs sum
    return loss

#%% Main GAN
# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
# dwgan


import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Input, merge
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras_adversarial.legacy import l1l2
import keras.backend as K
import pandas as pd
import numpy as np
from keras_adversarial.image_grid_callback import ImageGridCallback

from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.layers import LeakyReLU, Activation
import os


def model_generator(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1l2(1e-7, 0)):
    inputs = Input(shape=(input_shape[0], input_shape[1], 1))
    model=Model(inputs,inputs,name="generator")
    return model
    #return Sequential([
     #   Dense(hidden_dim, name="generator_h1", input_dim=latent_dim, W_regularizer=reg()),
     #   LeakyReLU(0.2),
     #   Dense(hidden_dim, name="generator_h2", W_regularizer=reg()),
     #   LeakyReLU(0.2),
     #   Dense(np.prod(input_shape), name="generator_x_flat", W_regularizer=reg()),
     #   Activation('sigmoid'),
     #   Reshape(input_shape, name="generator_x")],
     #   name="generator")


def model_encoder(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1l2(1e-7, 0)):
    inputs = Input(shape=(64, 64, 1))
    conv_1 = Conv2D(25, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    conv_2 = Conv2D(15, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    conv_3 = Conv2D(10, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    deconv_1 = Conv2DTranspose(10, (3, 3), strides=(2, 2), padding='same')(pl_3)
    dact_1 = Activation('relu')(deconv_1)
    merge_1 = concatenate([dact_1, act_3], axis=3) 
    deconv_2 = Conv2DTranspose(15, (3, 3), strides=(2, 2), padding='same')(merge_1)
    dact_2 = Activation('relu')(deconv_2)
    merge_2 = concatenate([dact_2, act_2], axis=3)
    deconv_3 = Conv2DTranspose(25, (3, 3), strides=(2, 2), padding='same')(merge_2)
    dact_3 = Activation('relu')(deconv_3)
    merge_3 = concatenate([dact_3, inputs], axis=3)
    final = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(merge_3)
    dact_4 = Activation('relu')(final)
    
    model = Model(inputs=[inputs], outputs=dact_4, name="encoder")
    return model
#def model_encoder(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1l2(1e-7, 0)):
#    x = Input(input_shape, name="x")
#    h = Flatten()(x)
#    h = Dense(hidden_dim, name="encoder_h1", W_regularizer=reg())(h)
#    h = LeakyReLU(0.2)(h)
#    h = Dense(hidden_dim, name="encoder_h2", W_regularizer=reg())(h)
#    h = LeakyReLU(0.2)(h)
#    mu = Dense(latent_dim, name="encoder_mu", W_regularizer=reg())(h)
#    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", W_regularizer=reg())(h)
#    z = merge([mu, log_sigma_sq], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
#              output_shape=lambda p: p[0])
#    return Model(x, z, name="encoder")

def model_discriminator(latent_dim, output_dim=2, hidden_dim=512,reg=lambda: l1l2(1e-7, 1e-7)):
    inputs = Input(shape=(64, 64, 2))
    conv_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    conv_2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    conv_3 = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    fc=Flatten()(pl_3)
    fc_2=Dense(40)(fc)
    act_4=Activation('relu')(fc_2)
    fc_3=Dense(25)(act_4)
    act_5=Activation('relu')(fc_3)
    fc_4=Dense(10)(act_5)
    act_6=Activation('relu')(fc_4)
    fc_5=Dense(1)(act_6)
    act_7=Activation('sigmoid')(fc_5)
    
    model = Model(inputs=[inputs], outputs=[act_7])
    return model

#def model_discriminator(latent_dim, output_dim=1, hidden_dim=512,
#                        reg=lambda: l1l2(1e-7, 1e-7)):
#    z = Input((latent_dim,))
#    h = z
#    h = Dense(hidden_dim, name="discriminator_h1", W_regularizer=reg())(h)
#    h = LeakyReLU(0.2)(h)
#    h = Dense(hidden_dim, name="discriminator_h2", W_regularizer=reg())(h)
#    h = LeakyReLU(0.2)(h)
#    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())(h)
#    return Model(z, y)


def example_aae(path, adversarial_optimizer):
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = (64, 64)

    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape)
    # autoencoder (x -> x')
    autoencoder = model_encoder(latent_dim, input_shape)
    #autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (z -> y)
    discriminator = model_discriminator(latent_dim)

    # assemple AAE
    x = autoencoder.inputs[0]
    #z = encoder(x)
    xpred = autoencoder(x)
    #zreal = normal_latent_sampling((latent_dim,))(x)
    yreal = discriminator(concatenate([x,xpred],axis=3))
    #yfake = discriminator(z)
    aae = Model(x, fix_names([xpred, yreal], ["xpred", "yreal"]))

    # print summary of models
    generator.summary()
    encoder.summary()
    discriminator.summary()
    autoencoder.summary()

    adversarial_optimizer = AdversarialOptimizerSimultaneous()
    model = AdversarialModel(base_model=aae,
                         player_params=[autoencoder.trainable_weights, discriminator.trainable_weights],
                         player_names=["autoencoder", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=['adam', Adam(1e-3, decay=1e-3)],
                              loss={"yreal": "binary_crossentropy",
                              "xpred": masked_mse},
                              player_compile_kwargs=[{"loss_weights": {"yreal": 1e-2, "xpred": 1}}] * 2)




    History=model.fit(x=xtrain, y=y, validation_data=(xval, yval),epochs=100, batch_size=15)

    Outputs = model.predict(xtest)
    print(Outputs[0].shape)
    Outputs = Outputs[0].reshape(Outputs[0].shape[0],64,64)

    plt.figure()
    plt.imshow(xtest[1,:,:,0])
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(Outputs[1,:,:])
    plt.colorbar()
    plt.show()

    #plt.figure()
    #plt.imshow(ytest[0,1,:,:])
    #plt.colorbar()
    #plt.show()
    return (History, Outputs)
    
    
def main():
    (Hist, Out)=example_aae("output/aae", AdversarialOptimizerSimultaneous())
    return (Hist,Out)


#if __name__ == "__main__":
#    main()

#%%
Means=np.zeros(344*3)
Stds=np.zeros(344*3)
for i in range(3):
    for j in range(344):
        Means[i*344+j]=means[j]
        Stds[i*344+j]=stds[j]
        

# load mnist data
x_train, xtest, y_train, ytest, m_train, mtest, std_train, stdtest = train_test_split(Multiple, Images, Means, Stds, test_size=0.2)
xtrain, xval, ytrain, yval, mtrain, mval, stdtrain, stdval = train_test_split(x_train, y_train, m_train, std_train, test_size=0.2)
    
xtrain = np.reshape(xtrain, (xtrain.shape[0], 64, 64,1))
xtest = np.reshape(xtest, (xtest.shape[0], 64, 64,1))
ytrain = np.reshape(ytrain, (ytrain.shape[0], 64, 64,1))
ytest = np.reshape(ytest, (ytest.shape[0], 64, 64,1))
xval = np.reshape(xval, (xval.shape[0], 64, 64,1))
yval = np.reshape(yval, (yval.shape[0], 64, 64,1))

n = xtrain.shape[0]
y = [ytrain, np.ones((n, 1)), ytrain, np.ones((n, 1))]
nval = xval.shape[0]
yval = [yval, np.ones((nval, 1)), yval, np.zeros((nval, 1))]

#%%
(Hist, Out)=main()

#%% Plots
a=6

plt.figure()
plt.imshow(xtest[a,:,:,0]*stdtest[a]+mtest[a]-3,cmap='seismic',vmin=18,vmax=20.7)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(ytest[a,:,:,0]*stdtest[a]+mtest[a]-3,cmap='seismic')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Out[a,:,:]*stdtest[a]+mtest[a]-3,cmap='seismic',vmin=18.5,vmax=21)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()

Outcorr=Out.copy()
for i in range(64):
    for j in range(64):
        if (xtest[a,i,j,0]!=0):
            Outcorr[a,i,j]=ytest[a,i,j,0]

plt.figure()
plt.imshow(Outcorr[a,:,:]*stdtest[a]+mtest[a]-3,cmap='seismic',vmin=18.5,vmax=21)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()


#%%% Errors
a=7

Outcorr=Out.copy()
for i in range(64):
    for j in range(64):
        if (xtest[a,i,j,0]!=0):
            Outcorr[a,i,j]=ytest[a,i,j,0]
            
plt.figure()
plt.imshow(Outcorr[a,:,:]*stdtest[a]-ytest[a,:,:,0]*stdtest[a],cmap='seismic',vmin=-0.4,vmax=0.4)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Out[a,:,:]*stdtest[a]-ytest[a,:,:,0]*stdtest[a],cmap='seismic',vmin=-0.4,vmax=0.4)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()

plt.figure(figsize=(6,6)) 
plt.scatter(Outcorr[a,:,:]*stdtest[a]+mtest[a]-3,ytest[a,:,:,0]*stdtest[a]+mtest[a]-3,s=10)
plt.xlabel('real temperature')
plt.ylabel('predicted temperature')
plt.plot([18,20.5],[18,20.5])
plt.show()

plt.figure(figsize=(6,6)) 
plt.scatter(Out[a,:,:]*stdtest[a]+mtest[a]-3,ytest[a,:,:,0]*stdtest[a]+mtest[a]-3,s=10)
plt.xlabel('real temperature')
plt.ylabel('predicted temperature')
plt.plot([18,20.5],[18,20.5])
plt.show()

#%%
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
Rmse=rmse(Outcorr,ytest[:,:,:,0])

Sum=0
n=0
for a in range(207):
    for i in range(64):
        for j in range(64):
            if(xtest[a,i,j,0]!=0):
                Sum=Sum+(Out[a,i,j]-ytest[a,i,j,0])**2
                n=n+1
Rmse=np.sqrt(Sum/n)