import tensorflow as tf
import tensorflow_addons as tfa
from utils import args
from models import InstanceNormalization_ecg12

class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(RMSNorm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="alpha"
        )

        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="betas"
        )
    def call(self, inputs):
        # norm = tf.norm(inputs, axis=-1, keepdims=True)
        denominator = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + self.epsilon)
        x_norm = self.alpha*inputs / (denominator)+self.beta
        return x_norm
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


# Denoiser设置为filters = [32,64,128,256,512],    time_embed = TimeEmbedding(512)(time_layer)
def modelx(input_size=(1024, 12), output_channels=12, pool_size = 2,kernel_size=13):
    """ UNet3+ base model """
    filters = [16,32,64,128,256,512]
    input_layer = tf.keras.layers.Input(
        shape=input_size,name="input_layer")
    x = input_layer
    output = unet3plus_block(x,output_channels,pool_size,filters,kernel_size=kernel_size)
    return tf.keras.Model(inputs=[input_layer], outputs=[output], name='Backbone')
def Disc_model(input_size=(1024, 12), output_channels=12,cycles=14):
    """ UNet3+ base model """
    myinput= tf.keras.layers.Input(
        shape=input_size,name="input_layer")
    x1 = tf.keras.layers.Conv1D(filters=output_channels,
                                kernel_size=3,
                                strides=1,
                                activation='gelu',
                                padding='same')(myinput)
    for i in range(cycles):
        x1=tfa.layers.InstanceNormalization()(x1)
        x1 = tf.keras.layers.Conv1D(filters=output_channels,
                                    kernel_size=3,
                                    strides=1,
                                    activation='gelu',
                                    padding='same')(x1)
    myoutput=x1
    return tf.keras.Model(inputs=[myinput], outputs=[myoutput])
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, Signal_Len, dim, initializer='glorot_uniform', **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.Signal_Len = Signal_Len
        self.dim = dim
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.Signal_Len, output_dim=self.dim,
            embeddings_initializer=initializer,
        )
    def call(self, patches):
        positions = tf.range(start=0, limit=self.Signal_Len, delta=1,dtype=tf.float32)
        return patches + self.position_embedding(positions)
def SimpleANN(input_size=24):
    x0 = tf.keras.layers.Input(
        shape=input_size,
        name="input_layer"
    )
    x1 = tf.keras.layers.Dense(12,'sigmoid')(x0)
    x1 = x1 + tf.keras.layers.Dense(12,'sigmoid')(x1)
    x2 = x0+tf.keras.layers.Dense(24,'sigmoid')(x1)
    return tf.keras.Model(inputs=[x0], outputs=[x2], name='SimpleANN')


if __name__=='__main__':
    model=modelx()
    model.summary()
