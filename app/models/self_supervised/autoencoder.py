from keras.layers import Input, Dense
from keras.models import Model

# simple autoencoder
class simple_autoencoder:
    def __init__(self, input_shape:int):
        self.input_shape = input_shape
        self.layers()
        self.model = self.modeling()

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

        return model
    
    def encoder(self):
        inputs = Input((self.input_shape,))
        h = self.enc_h(inputs)
        outputs = self.enc_output(h)
        model = Model(inputs,outputs)

        return model