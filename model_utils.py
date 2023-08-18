from keras.layers import SimpleRNN, Dense, Dropout, LSTM, GRU
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers

def build_model(hp, time_stemp=10):
    """
    Builds the RNN model based on the given hyperparameters.

    :param hp: Hyperparameters.
    :param time_stemp: Time step for the data.
    :return: Compiled model.
    """
    model = Sequential()
    rnn_type = hp.Choice('rnn_type', ['SimpleRNN', 'LSTM', 'GRU'])
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.05)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    l1_reg = hp.Float('l1_reg', min_value=0.0, max_value=0.1, step=0.01)  # L1 regularization hyperparameter

    # Adding the respective RNN type with regularization
    if rnn_type == 'SimpleRNN':
        model.add(SimpleRNN(units, input_shape=(1, time_stemp), return_sequences=True, activity_regularizer=regularizers.l1(l1_reg)))
        model.add(SimpleRNN(units, activity_regularizer=regularizers.l1(l1_reg)))
    elif rnn_type == 'LSTM':
        model.add(LSTM(units, input_shape=(1, time_stemp), return_sequences=True, activity_regularizer=regularizers.l1(l1_reg)))
        model.add(LSTM(units, activity_regularizer=regularizers.l1(l1_reg)))
    elif rnn_type == 'GRU':
        model.add(GRU(units, input_shape=(1, time_stemp), return_sequences=True, activity_regularizer=regularizers.l1(l1_reg)))
        model.add(GRU(units, activity_regularizer=regularizers.l1(l1_reg)))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    return model
