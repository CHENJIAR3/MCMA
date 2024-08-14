import tensorflow as tf
import tensorflow_addons as tfa
from utils import args
from models import InstanceNormalization_ecg12

def downblock(x0, filters, kernel_size=13,strides=1,padding='same'):
    x1 = tf.keras.layers.Conv1D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                activation='gelu',
                                padding=padding)(x0)

    x2 = tf.keras.layers.Conv1D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides, activation='linear',
                               padding=padding)(x0)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tfa.layers.GELU()(x2)

    x = tf.keras.layers.Conv1D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='gelu',
                               padding=padding)(x2)
    x = tfa.layers.InstanceNormalization(epsilon=1e-9)(x1+x)

    return x

def upblock(x0, filters,kernel_size=13,strides=1,padding='same'):
    x1 = tf.keras.layers.Conv1DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        activation='gelu',
                                        padding=padding)(x0)
    # x1 = tf.keras.layers.LayerNormalization()(x1)
    x2 = tf.keras.layers.Conv1DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        activation='linear',
                                        padding=padding)(x0)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tfa.layers.GELU()(x2)

    x = tf.keras.layers.Conv1D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=1,
                                        activation='gelu',
                                        padding=padding)(x2)
    x = tfa.layers.InstanceNormalization(epsilon=1e-9)(x1+x)
    return x

def unet3plus_block(e0,output_channels=12,pool_size=1,filters =[32,64,128,256,512],kernel_size=13,num=32,con=None):
    e1 = downblock(e0,filters[0],kernel_size=kernel_size,strides=1)
    e2 = downblock(e1,filters[1],kernel_size=kernel_size,strides=pool_size)
    e3 = downblock(e2,filters[2],kernel_size=kernel_size,strides=pool_size)
    e4 = downblock(e3,filters[3],kernel_size=kernel_size,strides=pool_size)
    e5 = downblock(e4,filters[4],kernel_size=kernel_size,strides=pool_size)
    e6 = downblock(e5,filters[5],kernel_size=kernel_size,strides=pool_size)

    d5_e6 = upblock(e6,filters[4],strides=pool_size,kernel_size=kernel_size)
    d5_e5 = downblock(e5,filters[4],kernel_size=kernel_size)
    d5 = d5_e5+d5_e6

    d4_d5 = upblock(d5,filters[3], strides=pool_size,kernel_size=kernel_size)
    d4_e4 = downblock(e4,filters[3],kernel_size=kernel_size)
    d4 = d4_d5+d4_e4

    d3_d4 = upblock(d4,filters[2], strides=pool_size,kernel_size=kernel_size)
    d3_e3 = downblock(e3, filters[2],kernel_size=kernel_size)
    d3 = d3_d4+d3_e3

    d2_d3 = upblock(d3, filters[1],  strides=pool_size,kernel_size=kernel_size)
    d2_e2 = downblock(e2, filters[1],kernel_size=kernel_size)
    d2 = d2_d3+d2_e2

    d1_d2 = upblock(d2,filters[0],strides=pool_size,kernel_size=kernel_size)
    d1_e1 = downblock(e1, filters[0],kernel_size=kernel_size)
    d1 = d1_d2+d1_e1

    d0 = downblock(d1,output_channels,kernel_size=kernel_size)
    return d0


def modelx(input_size=(1024, 12), output_channels=12, pool_size = 2,kernel_size=13):
    """ UNet3+ base model """
    filters = [16,32,64,128,256,512]
    input_layer = tf.keras.layers.Input(
        shape=input_size,name="input_layer")
    x = input_layer
    output = unet3plus_block(x,output_channels,pool_size,filters,kernel_size=kernel_size)
    return tf.keras.Model(inputs=[input_layer], outputs=[output], name='Backbone')


if __name__=='__main__':
    model=modelx()
    model.summary()
