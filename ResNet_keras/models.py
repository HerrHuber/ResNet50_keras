# -*- coding: utf-8 -*-
# This Program

import time
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Activation, MaxPooling2D, Add, ZeroPadding2D, AveragePooling2D, Flatten, Dense
import tensorflow as tf

from tensorflow.keras import datasets, models, Input
import matplotlib.pyplot as plt


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # main path 1
    X = Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # main path 2
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # main path 3
    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # shortcut path
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # main + shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # main path 1
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # main path 2
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # main path 3
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # main + shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape, classes):
    X_input = Input(input_shape)

    #X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    model = models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def main():
    print(time.time())
    # Test ResNet50 on tensorflow tutorial dataset
    # https://www.tensorflow.org/tutorials/images/cnn
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # get first 1000 examples
    train_images = train_images[:1000, :, :, :]
    train_labels = train_labels[:1000, :]

    test_images = test_images[:1000, :, :, :]
    test_labels = test_labels[:1000, :]

    # image shape is (32, 32, 3)
    # I want to test on (224, 224, 3)
    print("shape before resize: ", train_images.shape)
    train_images = tf.image.resize(train_images, (224, 224))
    print("shape after resize: ", train_images.shape)

    print("shape before resize: ", test_images.shape)
    test_images = tf.image.resize(test_images, (224, 224))
    print("shape after resize: ", test_images.shape)

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    classes = 10
    model = ResNet50(input_shape=(224, 224, 3), classes=classes)
    #model.summary()
    last_layer = model.get_layer('activation_48').output

    X = AveragePooling2D(pool_size=(7, 7), name="avg_pool")(last_layer)
    X = Flatten()(X)
    #X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    X = Dense(classes, name='fc' + str(classes))(X)
    new_model = models.Model(model.input, X)

    new_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = new_model.fit(train_images, train_labels, epochs=1,
                        validation_data=(test_images, test_labels))

    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)
    """


if __name__ == "__main__":
    main()
#
#
#
#
#
#
#
#
#
#
