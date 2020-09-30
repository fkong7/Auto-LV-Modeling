import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  return decoder

def UNet2D(img_shape,num_class):
    
    inputs = layers.Input(shape=img_shape)
    # 256
    
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 128
    
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64
    
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32
    
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16
    
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    # 8
    
    center = conv_block(encoder4_pool, 1024)
    # center
    
    decoder4 = decoder_block(center, encoder4, 512)
    # 16
    
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    
    outputs = layers.Conv2D(num_class, (1, 1), activation='softmax', data_format="channels_last")(decoder0)
    return inputs, outputs

class UNet2DIsensee(object):
    def __init__(self, input_size, num_class=8):
        super(UNet2DIsensee, self).__init__()
        self.num_class = num_class
        self.input_size = input_size

    def build(self):
        inputs = layers.Input(self.input_size)

        output0 = self._context_module(16, inputs, strides=(1,1))
        output1 = self._context_module(32, output0, strides=(2,2))
        output2 = self._context_module(64, output1, strides=(2,2))
        output3 = self._context_module(128, output2, strides=(2,2))
        output4 = self._context_module(256, output3, strides=(2,2))
        
        decoder0 = self._decoder_block(128, [output3, output4])
        decoder1 = self._decoder_block(64, [output2, decoder0])
        decoder2 = self._decoder_block(32, [output1, decoder1])
        decoder3 = self._decoder_block_last(16, [output0, decoder2])
        output0 = layers.Conv2D(self.num_class, (1, 1))(decoder3)
        output1 = layers.Conv2D(self.num_class, (1, 1))(decoder2)
        output2_up = layers.UpSampling2D(size=(2,2))(layers.Conv2D(self.num_class, (1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling2D(size=(2,2))(output_sum)
        output_sum = layers.Add()([output_sum, output0])
        output = layers.Softmax()(output_sum)

        return models.Model(inputs=[inputs], outputs=[output])

    def _conv_block(self, num_filters, inputs, strides=(1,1)):
        output = layers.Conv2D(num_filters, (3, 3),kernel_regularizer=regularizers.l2(0.01),  padding='same', strides=strides)(inputs)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(output))
        return output

    def _context_module(self, num_filters, inputs, dropout_rate=0.3, strides=(1,1)):
        conv_0 = self._conv_block(num_filters, inputs, strides=strides)
        conv_1 = self._conv_block(num_filters, conv_0)
        dropout = layers.SpatialDropout2D(rate=dropout_rate)(conv_1)
        conv_2 = self._conv_block(num_filters, dropout)
        sum_output = layers.Add()([conv_0, conv_2])
        return sum_output
    
    def _decoder_block(self, num_filters,  inputs, strides=(2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling2D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters, concat)
        conv_3 = layers.Conv2D(num_filters, (1,1), padding='same')(conv_2)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(conv_3))
        return output

    def _decoder_block_last(self, num_filters,  inputs, strides=(2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling2D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters*2, concat)
        return conv_2

from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras import initializers, regularizers, constraints

class InstanceNormalization(layers.Layer):
    """Instance normalization layer. Taken from keras.contrib
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
