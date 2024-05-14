# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

Train=pd.read_csv('/kaggle/input/dataset/train.csv')
Test=pd.read_csv('/kaggle/input/dataset/test.csv')
sample_submission= pd.read_csv('/kaggle/input/dataset/sample_submission.csv')

main_df = Train.copy()

# Replace ".." values with zeros in the main_df
main_df.replace("..", 0, inplace=True)

main_df.head(13)

# Get all unique values in the 'Indicator' column
all_attributes = Train['Indicator'].unique()
print(all_attributes)

# Get all unique values in the "Country Name" column
unique_country_names = main_df['Country Name'].unique()
print(unique_country_names)


# Filter the DataFrame by all of the attributes
filtered_rows = main_df[main_df['Indicator'].isin(all_attributes)]
filtered_rows

#Drop Country Code Column
filtered_rows.drop(columns=['Country Code'], inplace=True)
filtered_rows.head(14)

#Drop Country Name Column
filtered_rows.drop(columns=['Country Name'], inplace=True)
filtered_rows.head(14)

# Create a dictionary to store DataFrames for each country
country_dataframes = {}

# Iterate over unique country names
for i, country_name in enumerate(unique_country_names):
    # Calculate the start and end indices for the current country
    start_index = i * 12
    end_index = (i + 1) * 12
    # Get the rows corresponding to the current country
    country_df = filtered_rows.iloc[start_index:end_index].reset_index(drop=True)  # Remove index
    
    # Transpose the dataframe
    country_df = country_df.T
    
    # Get the values of the second row as column names
    column_names = country_df.iloc[0].values

    # Set the column names of country_df
    country_df.columns = column_names

    # Add a new column named "Year" at the beginning
    country_df.insert(0, "Year", country_df.index)

    # Delete the first row
    country_df = country_df.drop(country_df.index[0])

    # Replace indexes with range values
    country_df.index = range(len(country_df))

    # Extract the first four characters of the years and convert them to integers
    country_df['Year'] = country_df['Year'].apply(lambda x: int(x[:4]))

    # Store the DataFrame in the dictionary with country name as key
    country_dataframes[country_name] = country_df


country_df = country_dataframes['Afghanistan']
country_df


# Variables for training
variables = list(country_df)[1:13]
variables

# Making the variables of data type float so that we don't lose any inforamtion while normalization
country_df_for_training = country_df[variables].astype(float)
country_df_for_training

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# Normalizing the dataset
scaler = StandardScaler()
scaler = scaler.fit(country_df_for_training)
country_df_for_training_scaled = scaler.transform(country_df_for_training)

# country_df_for_training_scaled

# Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of years we want to look into the future based on the past years
n_past = 5  # Number of past years we want to use to predict the future.

for i in range(n_past, len(country_df_for_training_scaled) - n_future +1):
    trainX.append(country_df_for_training_scaled[i - n_past:i, 0:country_df_for_training.shape[1]])
    trainY.append(country_df_for_training_scaled[i + n_future - 1:i + n_future, -1])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

print(country_df_for_training_scaled)
print("----------------------------------------------------------------------------------------------")
print(trainX)
print("----------------------------------------------------------------------------------------------")
print(trainY)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(trainX, trainY, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss') 
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

n_future = 11   # since n_past is 5, 2015 is 11 years apart
forcast = model.predict(trainX[-n_future:])  # shape = (n, 1) where n is the n years of prediction
forcast

forcast_copies = np.repeat(forcast, country_df_for_training.shape[1], axis=-1)
forcast_copies

y_pred_future = scaler.inverse_transform(forcast_copies)[:,-1]
y_pred_future



main_test_df = Test.copy()

# Replace ".." values with zeros in the main_df
main_test_df.replace("..", 0, inplace=True)

main_test_df.head(13)


# Filter the DataFrame by all of the attributes
filtered_test_rows = main_test_df[main_test_df['Indicator'].isin(all_attributes)]
filtered_test_rows

#Drop Country Name Column
filtered_test_rows.drop(columns=['Country Name'], inplace=True)
filtered_test_rows.head(14)

# Create a dictionary to store DataFrames for each country for testing 
country_test_dataframes = {}

# Iterate over unique country names
for i, country_name in enumerate(unique_country_names):
    # Calculate the start and end indices for the current country
    start_index = i * 11
    end_index = (i + 1) * 11
    # Get the rows corresponding to the current country
    country_test_df = filtered_test_rows.iloc[start_index:end_index].reset_index(drop=True)  # Remove index
    
    # Transpose the dataframe
    country_test_df = country_test_df.T
    
    # Get the values of the second row as column names
    column_names = country_test_df.iloc[0].values

    # Set the column names of country_df
    country_test_df.columns = column_names

    # Add a new column named "Year" at the beginning
    country_test_df.insert(0, "Year", country_test_df.index)

    # Delete the first row
    country_test_df = country_test_df.drop(country_test_df.index[0])

    # Replace indexes with range values
    country_test_df.index = range(len(country_test_df))

    # Extract the first four characters of the years and convert them to integers
    country_test_df['Year'] = country_test_df['Year'].apply(lambda x: int(x[:4]))

    # Store the DataFrame in the dictionary with country name as key
    country_test_dataframes[country_name] = country_test_df


country_test_df = country_test_dataframes['Afghanistan']
country_test_df


# Variables for testing
variables = list(country_test_df)[1:12]
variables

# Making the variables of data type float so that we don't lose any inforamtion while normalization
country_df_for_testing = country_test_df[variables].astype(float)
country_df_for_testing

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# Normalizing the dataset
scaler = StandardScaler()
scaler = scaler.fit(country_df_for_testing)
country_df_for_testing_scaled = scaler.transform(country_df_for_testing)

country_df_for_testing_scaled
