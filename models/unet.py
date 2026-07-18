import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, ReLU, Concatenate


def UNet(input_size=(256, 256, 1), num_filters=8):
    inputs = Input(shape=input_size)

    # Encoder
    conv1 = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(num_filters, (3, 3), padding='same')(conv1)
    conv1 = ReLU()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(num_filters * 2, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(num_filters * 2, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(num_filters * 4, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(num_filters * 4, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(num_filters * 8, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(num_filters * 8, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(num_filters * 16, (3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv2D(num_filters * 16, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)

    # Decoder
    upconv4 = Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat4 = Concatenate()([upconv4, conv4])
    conv6 = Conv2D(num_filters * 8, (3, 3), padding='same')(concat4)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv2D(num_filters * 8, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)

    upconv3 = Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat3 = Concatenate()([upconv3, conv3])
    conv7 = Conv2D(num_filters * 4, (3, 3), padding='same')(concat3)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv2D(num_filters * 4, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)

    upconv2 = Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat2 = Concatenate()([upconv2, conv2])
    conv8 = Conv2D(num_filters * 2, (3, 3), padding='same')(concat2)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)
    conv8 = Conv2D(num_filters * 2, (3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)

    upconv1 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat1 = Concatenate()([upconv1, conv1])
    conv9 = Conv2D(num_filters, (3, 3), padding='same')(concat1)
    conv9 = ReLU()(conv9)
    conv9 = Conv2D(num_filters, (3, 3), padding='same')(conv9)
    conv9 = ReLU()(conv9)

    # Output Layer
    outputs = Conv2D(1, (1, 1), activation='linear')(conv9)

    model = Model(inputs, outputs)
    return model
