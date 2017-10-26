"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, concatenate, Input
from keras.layers.embeddings import Embedding
from keras.models import Model

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

import logging

def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)

def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1.
    return return_vals

def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.exp(np.clip(np.absolute(return_vals), -7, 7)) - 1
    return_vals[neg_mask] *= -1.
    return return_vals

def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    # Ensure data shape is correct
    actual_y = actual_y.reshape(actual_y.shape[0], )
    prediction_y = prediction_y.reshape(prediction_y.shape[0], )
    # Calculate MAPE
    diff = np.absolute((actual_y - prediction_y) / np.clip(np.absolute(actual_y), 1., None))
    return 100. * np.mean(diff)

def compile_model(network, input_shape, model_type):
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

    if model_type == "mae":
        model.add(Dense(30, activation=activation, name="int_layer"))
        model.add(Dropout(dropout))

    # Output layer.
    model.add(Dense(1, activation='linear'))

    if model_type == "mape":
        model.compile(loss=sc_mean_absolute_percentage_error, optimizer=optimizer, metrics=['mae'])
    else:
        model.compile(loss='mae', optimizer=optimizer, metrics=[sc_mean_absolute_percentage_error])

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
    train_x = df_all_train_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.as_matrix()


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=36)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    input_shape = (train_x.shape[1],)


    model = compile_model(network, input_shape, "mape")

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))

    history = model.fit(train_x, train_y,
                        batch_size=network['batch_size'],
                        epochs=10000,  # using early stopping, so no real limit
                        verbose=0,
                        validation_data=(test_x, test_y),
                        callbacks=[early_stopping, csv_logger, reduce_lr, checkpointer])

    model.load_weights('weights.hdf5')
    predictions = model.predict(test_x)
    mae = mean_absolute_error(test_y, predictions)
    mape = safe_mape(test_y, predictions)

    score = mape

    print('\rResults')

    hist_epochs = len(history.history['val_loss'])

    if np.isnan(score):
        score = 9999

    print('epochs:', hist_epochs)
    print('mape:', mape)
    print('mae:', mae)
    print('-' * 20)

    logging.info('epochs: %d' % hist_epochs)
    logging.info('mape: %.4f' % mape)
    logging.info('mae: %.4f' % mae)
    logging.info('-' * 20)

    return score

def train_and_score_bagging(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    train_predictions = pd.read_pickle('data/train_predictions.pkl.gz', compression='gzip')
    test_predictions = pd.read_pickle('data/test_predictions.pkl.gz', compression='gzip')

    train_actuals = pd.read_pickle('data/train_actuals.pkl.gz', compression='gzip')
    test_actuals = pd.read_pickle('data/test_actuals.pkl.gz', compression='gzip')


    train_x = train_predictions.as_matrix()
    train_y = train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_x = test_predictions.as_matrix()
    test_y = test_actuals[0].values
    test_log_y = safe_log(test_y)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    input_shape = (train_x.shape[1],)


    model = compile_model(network, input_shape, "mape")

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))

    # history = model.fit(train_x, train_y,
    history = model.fit(train_x, train_log_y,
                        batch_size=network['batch_size'],
                        epochs=10000,  # using early stopping, so no real limit
                        verbose=0,
                        # validation_data=(test_x, test_y),
                        validation_data=(test_x, test_log_y),
                        callbacks=[early_stopping, csv_logger, checkpointer])


    print('\rResults')

    hist_epochs = len(history.history['val_loss'])
    # score = history.history['val_loss'][hist_epochs - 1]

    model.load_weights('weights.hdf5')
    predictions = model.predict(test_x)
    prediction_results = predictions.reshape(predictions.shape[0],)
    prediction_results = safe_exp(prediction_results)
    score = safe_mape(test_y, prediction_results)

    if np.isnan(score):
        score = 9999

    print('epochs:', hist_epochs)
    print('loss:', score)
    print('-' * 20)

    logging.info('epochs: %d' % hist_epochs)
    logging.info('loss: %.4f' % score)
    logging.info('-' * 20)

    return score

def train_and_score_entity_embedding(network):
    # Creating the neural network
    embeddings = []
    inputs = []
    __Enc = dict()
    __K = dict()

    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout']
    batch_size = network['batch_size']



    train_x_df = pd.read_pickle('data/pp_train_x_df.pkl.gz', compression='gzip')
    train_y_df = pd.read_pickle('data/pp_train_y_df.pkl.gz', compression='gzip')
    test_x_df = pd.read_pickle('data/pp_test_x_df.pkl.gz', compression='gzip')
    test_y_df = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')

    __Lcat = train_x_df.dtypes[train_x_df.dtypes == 'object'].index
    # __Lnum = train_x_df.dtypes[train_x_df.dtypes != 'object'].index


    for col in __Lcat:
        exp_ = np.exp(-train_x_df[col].nunique() * 0.05)
        __K[col] = np.int(5 * (1 - exp_) + 1)

    for col in __Lcat:
        d = dict()
        levels = list(train_x_df[col].unique())
        nan = False


        if np.NaN in levels:
            nan = True
            levels.remove(np.NaN)

        for enc, level in enumerate([np.NaN] * nan + sorted(levels)):
            d[level] = enc

        __Enc[col] = d

        var = Input(shape=(1,))
        inputs.append(var)

        emb = Embedding(input_dim=len(__Enc[col]),
                        output_dim=__K[col],
                        input_length=1)(var)
        emb = Reshape(target_shape=(__K[col],))(emb)

        embeddings.append(emb)

    if (len(__Lcat) > 1):
        emb_layer = concatenate(embeddings)
    else:
        emb_layer = embeddings[0]



    # Add embedding layer as input layer
    outputs = emb_layer

    # Add each dense layer
    for i in range(nb_layers):

        outputs = Dense(nb_neurons, kernel_initializer='uniform', activation=activation)(outputs)

        # Add dropout for all layers after the embedding layer
        if i > 0:
            outputs = Dropout(dropout)(outputs)


    # Add final linear output layer.
    outputs = Dense(1, kernel_initializer='normal', activation='linear')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning the weights
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, verbose=1, patience=4)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    csv_logger = CSVLogger('./logs/entity_embedding.log')

    train_x = [train_x_df[col].apply(lambda x: __Enc[col][x]).values for col in __Lcat]
    test_x = [test_x_df[col].apply(lambda x: __Enc[col][x]).values for col in __Lcat]

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))


    history = model.fit(train_x, train_y_df.values,
        validation_data=(test_x, test_y_df.values),
        epochs=500,
        verbose=0,
        batch_size=batch_size,
        callbacks=[early_stopping, csv_logger],
        )

    print('\rResults')

    hist_epochs = len(history.history['val_loss'])

    score = history.history['val_loss'][hist_epochs-1]

    if np.isnan(score):
        score = 9999

    print('epochs:', hist_epochs)
    print('loss:', score)
    print('-' * 20)

    logging.info('epochs: %d' % hist_epochs)
    logging.info('loss: %.4f' % score)
    logging.info('-' * 20)

    return score

