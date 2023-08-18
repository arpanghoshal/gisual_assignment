import numpy as np
import math
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from kerastuner.tuners import RandomSearch
from sklearn.metrics import mean_absolute_error
from data_utils import load_data, scale_data, split_data, prepare_data
from model_utils import build_model

# Loading and processing the data
data = load_data("sp500.csv")
scaler, dataset = scale_data(data)
train, test = split_data(dataset)
time_stemp = 10
trainX, trainY = prepare_data(train, time_stemp)
testX, testY = prepare_data(test, time_stemp)
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

# RandomSearch tuner
tuner = RandomSearch(lambda hp: build_model(hp, time_stemp=time_stemp), objective='val_loss', directory='output_dir', project_name='sp500_rnn')

# Splitting training data for validation in hyperparameter tuning
val_size = int(trainX.shape[0] * 0.25)
trainX_tune, valX_tune = trainX[:-val_size], trainX[-val_size:]
trainY_tune, valY_tune = trainY[:-val_size], trainY[-val_size:]

# Hyperparameter tuning
tuner.search(trainX_tune, trainY_tune, epochs=10, validation_data=(valX_tune, valY_tune), batch_size=16)

# Getting the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=3)[0]
print(f"L1 Regularization: {best_hps.get('l1_reg')}")

# Building and summarizing the chosen model
model = tuner.hypermodel.build(best_hps)
print("Chosen Model Summary:")
model.summary()

model_checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True, verbose=1)

# Training the model with the chosen hyperparameters
model.fit(trainX, trainY, epochs=50, batch_size=8, callbacks=[model_checkpoint])

model.load_weights('best_model.h5')

# Making predictions and evaluating the model
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_absolute_error(trainY[0], trainPredict[:, 0]))
testScore = math.sqrt(mean_absolute_error(testY[0], testPredict[:, 0]))
print('Train Score: %.2f MAE' % (trainScore))
print('Test Score: %.2f MAE' % (testScore))

# Plotting the predictions
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict

f, ax = plt.subplots(figsize=(30, 7))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig('predictions.png')  # Save the plot to a file instead of displaying
