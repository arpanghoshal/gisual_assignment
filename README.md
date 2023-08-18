
# Gisual Challenge
by Arpan Ghoshal

---
### Coding challenge requirements satisfied:

1. Design and implement a recurrent neural network (RNN) model for time series forecasting. You can use frameworks like TensorFlow, PyTorch, or any other libraries of your choice. **DONE**: 
2. Preprocess the dataset by splitting it into training and testing sets. Use a suitable strategy to ensure the model can generalize well to unseen data. **DONE**
3. Train your RNN model using the training dataset. Experiment with different architectures, hyperparameters, and optimization techniques to improve performance. **DONE**
4. Evaluate the performance of your model on the testing dataset. Use appropriate evaluation metrics for time series forecasting, such as mean squared error (MSE) or mean absolute error (MAE). **DONE**
5. Provide a clear documentation of your approach, including the rationale behind your model selection, data preprocessing steps, and any modifications made during training. **DONE**
6. Bonus: Implement a method to handle missing data and/or distribution shifts in the time series dataset. **:::** There were no missing points in the data.


---

### How to Use

1. Clone the repo: `git clone <repository_url> && cd <repository_directory>`
2. Create and activate Python 3.11 virtual environment: `python3.11 -m venv myenv && source myenv/bin/activate` (macOS/Linux) or `python3.11 -m venv myenv && .\myenv\Scripts\activate` (Windows)
3. Install required libraries: `pip install -r requirements.txt`
4. Run the code: `python main.py`

---

## Code Structure

### main.py

This script is responsible for loading and processing the data, tuning the hyperparameters of the RNN model, building, training, and evaluating the model, and plotting the predictions.

#### Imports
- `numpy`, `math`, `matplotlib.pyplot` are standard libraries used for numerical operations and plotting.
- `keras.callbacks`, `kerastuner.tuners`, `sklearn.metrics` are used for model training, hyperparameter tuning, and evaluation.
- `data_utils` and `model_utils` contain custom utility functions.

#### Data Loading and Processing
- Loads the SP500 data from the CSV file.
- Scales the data using MinMax scaling.
- Splits the data into training and testing datasets.
- Prepares the data for RNN.

#### Hyperparameter Tuning
- Utilizes `RandomSearch` for tuning hyperparameters.
- Gets the best hyperparameters.

#### Model Building and Training
- Builds the chosen RNN model.
- Utilizes the `ModelCheckpoint` callback.
- Fits the model to the training data.

#### Model Evaluation and Plotting
- Makes predictions on the training and testing datasets.
- Evaluates the model using Mean Absolute Error (MAE).
- Plots the predictions and saves the plot.

### model_utils.py

This script contains a function to build the RNN model based on given hyperparameters.

#### `build_model(hp, time_stemp=10)`
- Builds the RNN model based on hyperparameters.
- Utilizes the `Sequential` model, various RNN layers (`SimpleRNN`, `LSTM`, `GRU`), `Dense`, `Dropout`, `Adam` optimizer, and `regularizers`.
- Returns the compiled model.

### data_utils.py

This script contains utility functions to load, scale, split, and prepare the data.

#### `load_data(filename)`
- Loads the data from a given filename.
- Returns the loaded dataset.

#### `scale_data(data)`
- Scales the data using MinMax scaling.
- Returns the scaler object and scaled data.

#### `split_data(dataset, train_size_ratio=0.85)`
- Splits the data into training and testing datasets.
- Returns training and testing datasets.

#### `prepare_data(data, time_stemp=10)`
- Prepares the data for the RNN.
- Returns prepared X and Y datasets.

---


## Model Architecture

#### RNN Layer
The model begins with one or two RNN layers depending on the chosen type of RNN: SimpleRNN, LSTM, or GRU.

- **SimpleRNN**: The simplest form of RNN, where the output from the previous step is fed into the current step along with the input. It is prone to vanishing or exploding gradients for long sequences.
- **LSTM (Long Short-Term Memory)**: An advanced RNN that is capable of learning long-term dependencies in the sequence data. It utilizes a complex mechanism with gating units to avoid the vanishing gradient problem.
- **GRU (Gated Recurrent Unit)**: Similar to LSTM but with a simpler structure. It also uses gating mechanisms to better capture dependencies in the sequence data.

The choice of RNN type, as well as the number of units in each layer, is determined by hyperparameters.

#### Regularization
L1 regularization is applied to the activity of the RNN layers. It helps prevent overfitting by adding a penalty term based on the absolute values of the model weights. The hyperparameter `l1_reg` controls the amount of this regularization.

#### Dropout Layer
A Dropout layer is included to further reduce overfitting. This layer randomly sets a fraction of the input units to 0 during training, which helps prevent over-reliance on any single unit. The dropout rate is controlled by a hyperparameter.

#### Dense Layer
A Dense layer is used as the output layer. It has a single output unit with a linear activation function, making it suitable for regression tasks like the prediction of continuous values.

#### Loss Function
The loss function used is Mean Squared Error (MSE), which calculates the squared differences between the predicted and actual values. It is a common choice for regression tasks.

#### Optimizer
The Adam optimizer is used to update the model's weights based on the gradients of the loss with respect to the weights. The learning rate is one of the hyperparameters and is essential in controlling how quickly or slowly the model learns.

### Hyperparameter Tuning
The given code leverages hyperparameter tuning to search for the best hyperparameters. The hyperparameters include:
- Type of RNN (SimpleRNN, LSTM, GRU)
- Number of units in the RNN layers
- Dropout rate
- Learning rate
- L1 regularization factor

These hyperparameters are critical in determining the model's capacity, behavior, and ultimately its performance.

### Summary
The model architecture is dynamic, allowing for the configuration of different types of RNNs with various numbers of units, regularization, and other hyperparameters. This flexibility, combined with hyperparameter tuning, enables the model to be tailored to specific sequence prediction tasks, such as forecasting stock market prices in the given context.

---

##### Example Run Score on the Dataset Provided

- Train Score: 4.56 MAE 
- Test Score: 6.97 MAE
