from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, ELU
from keras import regularizers, optimizers

def k_mae_mape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    mape = 100. * K.mean(diff, axis=-1)
    mae = K.mean(K.abs(y_true - y_pred), axis=-1)
    return mape * mae

def k_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)

# For many activations, we can just pass the activation name into Activations
# For some others, we have to import them as their own standalone activation function
def get_activation_layer(activation):
    if activation == 'LeakyReLU':
        return LeakyReLU()
    if activation == 'PReLU':
        return PReLU()
    if activation == 'ELU':
        return ELU()
    if activation == 'ThresholdedReLU':
        return ThresholdedReLU()

    return Activation(activation)

def get_optimizer(name='Adadelta'):
    if name == 'SGD':
        return optimizers.SGD()#clipnorm=1.)
    if name == 'RMSprop':
        return optimizers.RMSprop()#clipnorm=1.)
    if name == 'Adagrad':
        return optimizers.Adagrad()#clipnorm=1.)
    if name == 'Adadelta':
        return optimizers.Adadelta()#clipnorm=1.)
    if name == 'Adam':
        return optimizers.Adam()#clipnorm=1.)
    if name == 'Adamax':
        return optimizers.Adamax()#clipnorm=1.)
    if name == 'Nadam':
        return optimizers.Nadam()#clipnorm=1.)

    return optimizers.Adam()#clipnorm=1.)


def compile_keras_model(network, dimensions):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout']
    kernel_initializer = network['kernel_initializer']
    model_type = network['model_type']
    num_classes = 0

    if model_type == "categorical_crossentropy":
        num_classes = network['num_classes']

    model = Sequential()

    # The hidden_layers passed to us is simply describing a shape. it does not know the num_cols we are dealing with, it is simply values of 0.5, 1, and 2, which need to be multiplied by the num_cols
    scaled_layers = []
    for layer in network['hidden_layers']:
        scaled_layers.append(max(int(dimensions * layer), 1))

    print('scaled_layers', scaled_layers)

    # Add input layers
    model.add(Dense(scaled_layers[0], kernel_initializer=kernel_initializer, input_dim=dimensions))
    model.add(get_activation_layer(activation))
    model.add(Dropout(dropout))

    # Add hidden layers
    for layer_size in scaled_layers[1:-1]:
        model.add(Dense(layer_size, kernel_initializer=kernel_initializer))
        model.add(get_activation_layer(activation))
        model.add(Dropout(dropout))


    if 'int_layer' in network:
        model.add(Dense(network['int_layer'], name="int_layer", kernel_initializer=kernel_initializer))
        model.add(get_activation_layer(activation))
        model.add(Dropout(dropout))

    # Output layer.
    if model_type == "categorical_crossentropy":
        model.add(Dense(num_classes, kernel_initializer=kernel_initializer, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer=get_optimizer(optimizer),
                      metrics=["categorical_accuracy"])
    else:
        model.add(Dense(1, kernel_initializer=kernel_initializer, activation='linear'))

        if model_type == "mape":
            model.compile(loss=k_mean_absolute_percentage_error, optimizer=get_optimizer(optimizer), metrics=['mae'])
        elif model_type == "mae_mape":
            model.compile(loss=k_mae_mape, optimizer=get_optimizer(optimizer), metrics=['mae', k_mean_absolute_percentage_error])
        else:
            model.compile(loss='mae', optimizer=get_optimizer(optimizer), metrics=[k_mean_absolute_percentage_error])

    return model
