"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import pandas as pd
import numpy as np

def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            0.15,
                                            None))
    return 100. * K.mean(diff, axis=-1)

def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1.
    return return_vals

def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    diff = np.absolute((actual_y - prediction_y) / np.clip(np.absolute(actual_y), 0.25, None))
    return 100. * np.mean(diff)

def compile_model(network, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(dropout))


    # Output layer.
    model.add(Dense(1, activation='linear'))

    model.compile(loss=sc_mean_absolute_percentage_error, optimizer=optimizer, metrics=['mae'])

    return model

def train_and_score(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.as_matrix()


    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # csv_logger = CSVLogger('./logs/actual-mape-training.log')

    input_shape = (train_x.shape[1],)


    model = compile_model(network, input_shape)

    model.fit(train_x, train_actuals,
              batch_size=network['batch_size'],
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(test_x, test_actuals),
              # callbacks=[reduce_lr, early_stopping, csv_logger])
              callbacks=[early_stopping])

    predictions = model.predict(test_x)
    score = safe_mape(test_actuals, predictions)

    print('\rNetwork results')

    for property in network:
        print(property, ':', network[property])

    if np.isnan(score):
        score = 9999

    print('Safe MAPE score:', score)
    print('-' * 20)

    return score
