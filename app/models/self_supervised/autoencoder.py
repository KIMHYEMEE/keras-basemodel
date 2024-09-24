from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model

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
class vae:
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
