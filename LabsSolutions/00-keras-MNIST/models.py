from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Lambda, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def make_linear(input_shape, num_classes,
                normalization=None,
                use_dropout=False,
                use_L2=False):
    xi = Input(shape=input_shape)
    x  = xi
    if normalization:
        x = normalization(x)
    x  = Flatten()(x)
    y  = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[xi], outputs=[y])
    return model

def make_fc(input_shape, num_classes,
            normalization=None,
            use_dropout=False, use_L2=False):

    L2_reg = 1e-4
    input_dropout = 0.2
    dropout = 0.5
    nhidden1 = 512
    nhidden2 = 512

    xi = Input(shape=input_shape)
    x = xi
    if normalization:
        x = normalization(x)
    x  = Flatten()(x)
    if use_dropout:
        x = Dropout(input_dropout)(x)

    if use_L2:
        x = Dense(nhidden1, kernel_regularizer=regularizers.l2(L2_reg), activation='relu')(x)
    else:
        x = Dense(nhidden1, activation='relu')(x)

    if use_dropout:
        x = Dropout(dropout)(x)

    if use_L2:
        x = Dense(nhidden2, kernel_regularizer=regularizers.l2(L2_reg), activation='relu')(x)
    else:
        x = Dense(nhidden2, activation='relu')(x)

    if use_dropout:
        x = Dropout(dropout)(x)

    y = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[xi], outputs=[y])
    return model

def conv_relu_max(nfilters, x):
    x = Conv2D(filters=nfilters,
            kernel_size=5, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    return x


def make_vanilla_cnn(input_shape, num_classes,
            normalization=None,
            use_dropout=False, use_L2=False):

    L2_reg = 1e-4
    input_dropout = 0.2
    dropout = 0.5
    nhidden1 = 512
    nhidden2 = 512

    xi = Input(shape=input_shape)
    x = xi
    if normalization:
        x = normalization(x)
    if use_dropout:
        x = Dropout(input_dropout)(x)

    for i in [16, 32, 64]:
        x = conv_relu_max(i, x)

    x  = Flatten()(x)
    if use_dropout:
        x = Dropout(dropout)(x)
    if use_L2:
        x = Dense(128, kernel_regularizer=regularizers.l2(L2_reg), activation='relu')(x)
    else:
        x = Dense(128, activation='relu')(x)
    if use_dropout:
        x = Dropout(dropout)(x)
    if use_L2:
        x = Dense(64, kernel_regularizer=regularizers.l2(L2_reg), activation='relu')(x)
    else:
        x = Dense(64, activation='relu')(x)
    y = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[xi], outputs=[y])
    return model


def fancyBlock(nfilters, x, last):
    x = Conv2D(filters=nfilters,
            kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=nfilters,
            kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    if last:
        x = GlobalAveragePooling2D()(x)
    else:
        x = MaxPooling2D(pool_size=2, padding='same')(x)

    return x


def make_fancy_cnn(input_shape, num_classes,
            normalization=None,
            use_dropout=False, use_L2=False):

    L2_reg = 1e-4
    input_dropout = 0.2
    dropout = 0.5

    xi = Input(shape=input_shape)
    x = xi
    if normalization:
        x = normalization(x)
    if use_dropout:
        x = Dropout(input_dropout)(x)

    for i in [16, 32]:
        x = fancyBlock(i, x, last=False)
    x = fancyBlock(64, x, last=True)
    if use_dropout:
        x = Dropout(dropout)(x)

    y = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[xi], outputs=[y])
    return model

builders = {'linear': make_linear,
            'fc': make_fc,
            'vanilla': make_vanilla_cnn,
            'fancy': make_fancy_cnn
           }
def build_network(model, input_shape, num_classes, normalization, use_dropout, use_L2):
    if normalization is not None:
        print(normalization)
        lambda_normalization = lambda xi: Lambda(lambda image, mu, std: (image - mu) / std,
                                                 arguments={'mu': normalization[0], 
                                                            'std': normalization[1]})(xi)
    return builders[model](input_shape, num_classes, lambda_normalization, use_dropout, use_L2)

