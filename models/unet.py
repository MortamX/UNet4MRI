from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Dropout
from tensorflow.keras.models import Model


class DecoderCell(Layer):
    def __init__(self, filters, dropout_rate, kernel_size):
        super(DecoderCell, self).__init__()
        self.trans_conv = Conv2DTranspose(filters, kernel_size, strides = (2, 2), padding = 'same')
        self.dropout = Dropout(dropout_rate)
        self.double_conv = DoubleConv2D(filters, kernel_size)

    def call(self, VerticalConvInput, HorizontalConvInput):
        x = self.trans_conv(VerticalConvInput)
        x = Concatenate()([x, HorizontalConvInput])
        x = self.dropout(x)
        VerticalConvOutput = self.double_conv(x)

        return VerticalConvOutput

class EncoderCell(Layer):

    def __init__(self, filters, kernel_size, dropout_rate):
        super(EncoderCell, self).__init__()
        self.double_conv = DoubleConv2D(filters, kernel_size)
        self.maxpool = MaxPool2D((2, 2))
        self.dropout = Dropout(dropout_rate)
    
    def call(self, input_tensor):
        conv_output = self.double_conv(input_tensor)
        pool_output = self.maxpool(conv_output)
        pool_output = self.dropout(pool_output)

        return conv_output, pool_output
    
class DoubleConv2D(Layer):

    def __init__(self, filters, kernel_size):
        super(DoubleConv2D, self).__init__()
        self.conv1 = Conv2D(filters = filters, kernel_size = kernel_size, kernel_initializer = 'he_normal', padding = 'same')
        self.conv2 = Conv2D(filters = filters, kernel_size = kernel_size, kernel_initializer = 'he_normal', padding = 'same')

    def call(self, input_tensor):    
        x = self.conv1(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.conv2(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

class UNet(Model):

    def __init__(self, n_filters, dropout_rate):

        super(UNet, self).__init__()

        self.EncoderCell_1 = EncoderCell(filters = n_filters * 1, kernel_size = 3, dropout_rate = dropout_rate)

        self.EncoderCell_2 = EncoderCell(filters = n_filters * 2, kernel_size = 3, dropout_rate = dropout_rate)
        
        self.EncoderCell_3 = EncoderCell(filters = n_filters * 4, kernel_size = 3, dropout_rate = dropout_rate)

        self.EncoderCell_4 = EncoderCell(filters = n_filters * 8, kernel_size = 3, dropout_rate = dropout_rate)
        
        self.transition_double_conv = DoubleConv2D(filters = n_filters * 16, kernel_size = 3)
        
        self.decoderCell_1 = DecoderCell(filters = n_filters * 8, kernel_size = 3, dropout_rate = dropout_rate)

        self.decoderCell_2 = DecoderCell(filters = n_filters * 4, kernel_size = 3, dropout_rate = dropout_rate)
        
        self.decoderCell_3 = DecoderCell(filters = n_filters * 2, kernel_size = 3, dropout_rate = dropout_rate)

        self.decoderCell_4 = DecoderCell(filters = n_filters * 1, kernel_size = 3, dropout_rate = dropout_rate)
        
        self.final_conv = Conv2D(1, (1, 1), activation='sigmoid')

    def build(self, input_shape):
        super(UNet, self).build(input_shape)
    
    def call(self, inputs):

        conv_output1, pooling_output1 = self.EncoderCell_1(inputs)

        conv_output2, pooling_output2 = self.EncoderCell_2(pooling_output1)
        
        conv_output3, pooling_output3 = self.EncoderCell_3(pooling_output2)

        conv_output4, pooling_output4 = self.EncoderCell_4(pooling_output3)

        conv_output5 = self.transition_double_conv(pooling_output4)
        
        conv_output6 = self.decoderCell_1(conv_output5, conv_output4)

        conv_output7 = self.decoderCell_2(conv_output6, conv_output3)
        
        conv_output8 = self.decoderCell_3(conv_output7, conv_output2)

        conv_output9 = self.decoderCell_4(conv_output8, conv_output1)
        
        self.final_conv(conv_output9)
        
        return self.final_conv(conv_output9)