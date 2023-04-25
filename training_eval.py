import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Data Loading
df=pd.read_csv('./Final_Plant1_Data.csv')

# Data Splitting
# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size].values
test_data = df[train_size:].values

# Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define window size for time series data (all data points within window size will be used to predict next data point)
window_size = 24

# Function to create windowed dataset
def create_dataset(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i,-1])
    return np.array(X), np.array(y)


# Create windowed training and testing datasets
train_X, train_y = create_dataset(train_data, window_size)
test_X, test_y = create_dataset(test_data, window_size)

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(window_size, train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train model
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y), verbose=1)

# Evaluate model on test data
test_loss = model.evaluate(test_X, test_y, verbose=0)

print('Test loss: {:.4f}'.format(test_loss))

# Make predictions on test data
test_pred = model.predict(test_X)

# Inverse transform data to get original values
test_pred = scaler.inverse_transform(np.concatenate((test_X[:,-1,:-1], test_pred), axis=1))[:,-1]
test_y = scaler.inverse_transform(np.concatenate((test_X[:,-1,:-1], test_y.reshape(-1,1)), axis=1))[:,-1]

# Plot predicted vs actual values

# Create new dataframe with DATE_TIME and predicted/actual values
test_df = pd.DataFrame({'DATE_TIME': df.index[train_size+window_size:], 'Predicted': test_pred, 'Actual': test_y})

# Set DATE_TIME as index
test_df.set_index('DATE_TIME', inplace=True)

# Plot predicted and actual values
plt.figure(figsize=(15,5))
plt.plot(test_df.index, test_df['Predicted'], label='Predicted')
plt.plot(test_df.index, test_df['Actual'], label='Actual')
plt.title('Predicted vs Actual Daily Yield')
plt.xlabel('Date')
plt.ylabel('Daily Yield')
plt.legend()
plt.show()

# Prediction
# Get the last 24 hours of the test data
last_24_hours = df[-96:]

# Normalize the last 24 hours of the test data
last_24_hours = scaler.transform(last_24_hours)

# Create a windowed dataset with a window size of 24
future_X, _ = create_dataset(last_24_hours, window_size)

# Make predictions on the future dataset
future_pred = model.predict(future_X)

# Inverse transform the predictions to get the original values
future_pred = scaler.inverse_transform(np.concatenate((future_X[:,-1,:-1], future_pred), axis=1))[:,-1]

# Create a datetime range for the next 2 days with the same frequency as the data
date_range = pd.date_range(start=df.index[-1], periods=192, freq='15min')

# Create a new date range with a smaller number of periods
future_range = pd.date_range(start=date_range[-48], periods=len(future_pred)+48, freq='15min')[48:]

# Create a dataframe with the predicted values and the datetime range
future_df2 = pd.DataFrame({'DATE_TIME': future_range, 'Predicted': future_pred})

# Set DATE_TIME as index
future_df2.set_index('DATE_TIME', inplace=True)
future_df_fin = pd.concat([future_df, future_df2], axis=0)

future_df_fin = pd.concat([future_df, future_df2], axis=0)

plt.figure(figsize=(15,5))
plt.plot(test_df.index, test_df['Predicted'], label='Predicted')
plt.plot(test_df.index, test_df['Actual'], label='Actual')
plt.plot(future_df_fin.index, future_df_fin['Predicted'], label='Future Predicted')
plt.title('Predicted vs Actual Daily Yield')
plt.xlabel('Date')
plt.ylabel('Daily Yield')
plt.legend()
plt.show()