# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
                                        LSTM, Bidirectional,Conv1D,Dense,Lambda,
                                        BatchNormalization,Dropout,concatenate,
                                        Layer,Multiply,LayerNormalization,SeparableConv1D,
                                        Add,Concatenate,GlobalAveragePooling1D,Softmax,
                                        RNN
                                    )

class MaskedConv1D(Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None and self.padding == 'valid':
            mask = mask[:, self.kernel_size[0] // 2 * self.dilation_rate[0] * 2:]
        return mask

    def call(self, inputs, mask=None):
        
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)
    
class ResidualGatedConv1D(Layer):
    def __init__(self, 
                 filters=None, 
                 kernel_size=3, 
                 dilation_rate=1, 
                 skip_connect=True,  
                 drop_gate=None, 
                 **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.skip_connect = skip_connect
        self.drop_gate = drop_gate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        if self.filters is None:
   
            self.filters = int(input_shape[-1])
        self.conv1d = MaskedConv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = Dense(self.filters, use_bias=False)


    def call(self, inputs, mask=None):
        
        outputs = self.conv1d(inputs)
        gate = outputs[..., self.filters:]
        outputs = outputs[..., :self.filters]
    
        if self.drop_gate is not None:
            gate = K.in_train_phase(K.dropout(gate, self.drop_gate), gate)
        gate = K.sigmoid(gate)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs * (1 - gate) + outputs * gate


    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return (shape[0], shape[1], shape[2] // 2)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'skip_connect': self.skip_connect, 
            'drop_gate': self.drop_gate
        }
        base_config = super(ResidualGatedConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))