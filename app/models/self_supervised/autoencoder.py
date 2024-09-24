from keras.layers import Input, Dense
from keras.models import Model

# simple autoencoder
def simple_ae(input_shape:int):
    enc_input = Input((input_shape,))
    enc_h = Dense(32)(enc_input)
    enc_output = Dense(16)(enc_h) # dec_input
    dec_h = Dense(32)(enc_output)
    dec_output = Dense(input_shape)(dec_h)
    
    model = Model(enc_input, dec_output)
    
    return model