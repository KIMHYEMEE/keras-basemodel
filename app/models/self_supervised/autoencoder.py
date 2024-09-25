import keras
import tensorflow as tf
from keras import ops
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model, Sequential

# simple autoencoder
class simple_autoencoder:
    def __init__(self, input_shape:int):
        self.input_shape = input_shape
        self.loss = 'mse'
        self.layers()
        self.model = self.modeling()

    def get_info(self):
        model_info = {'input_shape':self.input_shape,
                      'output_shape':self.input_shape,
                      'loss':self.loss}
        return model_info

    def layers(self):
        # encoder
        self.enc_h = Dense(32)
        self.enc_output = Dense(16)
        # decoder
        self.dec_h = Dense(32)
        self.dec_output = Dense(self.input_shape)

    def modeling(self):
        inputs = Input((self.input_shape,))
        h = self.enc_h(inputs)
        h = self.enc_output(h)
        h = self.dec_h(h)
        outputs = self.dec_output(h)

        model = Model(inputs,outputs)
        model.compile(loss=self.loss)

        return model
    
    def encoder(self):
        inputs = Input((self.input_shape,))
        h = self.enc_h(inputs)
        outputs = self.enc_output(h)
        model = Model(inputs,outputs)

        return model
    
# cnn autoencoder
class cnn_autoencoder:
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.loss = 'binary_crossentropy'
        self.layers()
        self.model = self.modeling()
    
    def get_info(self):
        model_info = {'input_shape':self.input_shape,
                      'output_shape':self.input_shape,
                      'loss':self.loss}
        return model_info
    
    def layers(self):
        self.enc_layers = [Conv2D(32, (3,3), activation='relu',padding='same'),
                           MaxPooling2D((2,2),padding='same'),
                           Conv2D(32, (3,3), activation='relu',padding='same'),
                           MaxPooling2D((2,2),padding='same')
                           ]
        
        self.dec_layers = [Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same'),
                           Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same'),
                           Conv2D(1, (3,3), activation='sigmoid', padding='same')
                           ]

    def modeling(self):
        inputs = Input(self.input_shape)

        # encoder
        h = self.enc_layers[0](inputs)
        for enc_layer in self.enc_layers[1:]:
            h = enc_layer(h)
        
        # decoder
        for dec_layer in self.dec_layers[:-1]:
            h = dec_layer(h)

        outputs = self.dec_layers[-1](h)

        model = Model(inputs, outputs)
        model.compile(loss=self.loss)

        return model

    def encoder(self):
        inputs = Input(self.input_shape)

        h = self.enc_layers[0](inputs)
        for enc_layer in self.enc_layers[1:-1]:
            h = enc_layer(h)
        outputs = self.enc_layers[-1](h)

        model = Model(inputs, outputs)

        return model
    
# vae
class Sampling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_log_var)[1]
        epsilon = keras.random.normal(shape=(batch,dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class vae:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.latent_dim = 2
        self.loss = 'reconstruction_kl'
        self.layers()
        self.model = self.modeling()
    
    def get_info(self):
        model_info = {'input_shape':self.model.input_shape,
                      'output_shape':self.model.output_shape,
                      'loss':self.loss}
        return model_info
    
    def layers(self):
        # encoder
        self.enc_h = [Conv2D(32, 3, activation='relu',strides=2, padding='same'),
                 Conv2D(32, 3, activation='relu',strides=2, padding='same'),
                 Flatten(),
                 Dense(16, activation='relu')
                 ]
        self.z_mean = Dense(self.latent_dim, name='z_mean')
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')
        self.z = Sampling()

        # decoder
        self.dec_h = [Dense(7*7*64, activation='relu'),
                      Reshape((7,7,64)),
                      Conv2DTranspose(64,3,activation='relu',strides=2,padding="same"),
                      Conv2DTranspose(32,3,activation='relu',strides=2,padding="same"),
                      Conv2DTranspose(1,3,activation='sigmoid',padding="same")                      
                      ]

        return
    
    def modeling(self):
        encoder = self.encoder()
        decoder = self.decoder()

        inputs = encoder.input
        z_mean = encoder.output[0]
        z_log_var = encoder.output[1]
        z = encoder.output[2]
        reconstruction = decoder(z)

        model = Model(inputs, [reconstruction,z_mean,z_log_var],name='vae')
        model.compile(loss=self.loss)

        return model
    
    def encoder(self):
        inputs = Input(shape=self.input_shape)
        x = self.enc_h[0](inputs)
        for h in self.enc_h[1:]:
            x = h(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.z([z_mean,z_log_var])

        model = Model(inputs, [z_mean, z_log_var,z], name='encoder')

        return model
    
    def decoder(self):
        inputs = Input(shape=(self.latent_dim,))
        x = self.dec_h[0](inputs)
        for h in self.dec_h[1:-1]:
            x = h(x)
        outputs = self.dec_h[-1](x)
        model = Model(inputs, outputs, name='decoder')

        return model
    
    def loss(self,y_true,y_pred):
        # y: [reconstruction, z_mean, z_log_var]
        reconstruction_loss = ops.mean(
            ops.sum(
                keras.losses.binary_crossentropy(y_true[0],y_pred[0]),
                axis=(1,2),
            )
        )
        kl_loss = -0.5 * (1 + y_pred[2] - ops.squre(y_pred[1]) - ops.exp(y_pred[2]))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        return total_loss

# VQ-VAE
class vq_vae:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.loss = ''
        self.layers()
        self.model = self.modeling()
    
    def get_info(self):

        return
    
    def layers(self):
        return
    
    def modeling(self):

        return # model
    
    def encoder(self):

        return # model
