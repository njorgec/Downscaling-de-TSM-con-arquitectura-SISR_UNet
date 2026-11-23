# model.py
"""
Definición de la arquitectura de la red neuronal U-Net personalizada.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, concatenate, UpSampling2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow import keras

def build_unet(seed: int = 42) -> keras.Model:
    """
    Construye y compila el modelo U-Net.
    
    Args:
        seed (int): Semilla para la inicialización de pesos.
        
    Returns:
        keras.Model: Modelo U-Net listo para entrenar.
    """
    kernel_init = tf.keras.initializers.HeNormal(seed=seed)
    
    inputT = Input(shape=(33, 53, 1), name="T")
    inputU = Input(shape=(99, 106, 1), name="U")
    inputV = Input(shape=(99, 106, 1), name="V")
    M_input = Input(shape=(297, 318, 1), name="Mask")
    
    conv1 = Conv2D(32, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(inputT)
    upT = UpSampling2D(size=(9,6), interpolation='bilinear')(inputT)
    
    conv1U = Conv2D(32, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(inputU)
    conv1V = Conv2D(32, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(inputV)
    
    up0 = UpSampling2D(size=(3,1), interpolation='bilinear')(conv1)
    conv2 = Conv2D(32, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up0)
    
    up1 = UpSampling2D(size=(1,2), interpolation='bilinear')(conv2)
    merge2 = concatenate([up1, conv1U, conv1V])
    conv3 = Conv2D(32, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(merge2)
    
    up2 = UpSampling2D(size=(3,1), interpolation='bilinear')(conv3)
    conv4 = Conv2D(32, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up2)
    
    up3 = UpSampling2D(size=(1,3), interpolation='bilinear')(conv4)
    conv5 = Conv2D(32, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up3)
    
    conv5T = concatenate([conv5, upT])
    conv10 = Conv2D(16, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(conv5T)
    
    # Bloque de contracción 1
    pool1 = MaxPooling2D(pool_size=(3, 1))(conv10)
    conv11 = Conv2D(32, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(pool1)
    pool2 = MaxPooling2D(pool_size=(1, 3))(conv11)
    conv12 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(pool2)
    pool3 = MaxPooling2D(pool_size=(3, 1))(conv12)
    conv13 = Conv2D(128, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(pool3)
    pool4 = MaxPooling2D(pool_size=(1, 2))(conv13)
    conv14 = Conv2D(256, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(pool4)
    
    # Bloque de expansión 1
    up4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(conv14)
    up4 = Conv2D(128, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up4)
    merge4 = concatenate([up4, conv13])
    conv15 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(merge4)
    up5 = UpSampling2D(size=(3, 1), interpolation='bilinear')(conv15)
    up5 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up5)
    merge5 = concatenate([up5, conv12])
    conv16 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(merge5)
    up6 = UpSampling2D(size=(1, 3), interpolation='bilinear')(conv16)
    up6 = Conv2D(32, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up6)
    merge6 = concatenate([up6 , conv11])
    
    conv17 = Conv2D(16, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init,kernel_regularizer=regularizers.l2(1e-5))(merge6)
    
    up7 = UpSampling2D(size=(3, 1), interpolation='bilinear')(conv17)
    up7 = Conv2D(16, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up7)
    merge7 = concatenate([up7, conv10, upT])
    
    conv20 = Conv2D(16, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(merge7)
    
    # Bloque de contracción 2
    pool21 = MaxPooling2D(pool_size=(3, 1))(conv20)
    conv21 = Conv2D(32, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(pool21)
    pool22 = MaxPooling2D(pool_size=(1, 3))(conv21)
    conv22 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(pool22)
    pool23 = MaxPooling2D(pool_size=(3, 1))(conv22)
    conv23 = Conv2D(128, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(pool23)
    pool24 = MaxPooling2D(pool_size=(1, 2))(conv23)
    conv24 = Conv2D(256, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(pool24)
    
    # Bloque de expansión 2
    up24 = UpSampling2D(size=(1, 2), interpolation='bilinear')(conv24)
    up24 = Conv2D(128, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up24)
    merge24 = concatenate([up24, conv23])
    conv25 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(1e-5))(merge24)
    up25 = UpSampling2D(size=(3, 1), interpolation='bilinear')(conv25)
    up25 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up25)
    merge25 = concatenate([up25, conv22])
    conv26 = Conv2D(64, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(merge25)
    up26 = UpSampling2D(size=(1, 3), interpolation='bilinear')(conv26)
    up26 = Conv2D(32, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up26)
    merge26 = concatenate([up26 , conv21])
    
    conv27 = Conv2D(16, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init,kernel_regularizer=regularizers.l2(1e-5))(merge26)
    up27 = UpSampling2D(size=(3, 1), interpolation='bilinear')(conv27)
    up27 = Conv2D(16, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(up27)
    merge27 = concatenate([up27,conv20,upT])
    
    # Salida final
    conv18 = Conv2D(8, 3, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(merge27)
    conv19 = Conv2D(4, 2, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(conv18)
    conv19x = Conv2D(1, 1, activation='leaky_relu', padding='same', kernel_initializer=kernel_init)(conv19)
    
    masked_output = keras.layers.Multiply()([conv19x, M_input])
    
    model = keras.models.Model(inputs=[inputT, inputU, inputV, M_input], outputs=masked_output, name="UNet_Clima")
    return model