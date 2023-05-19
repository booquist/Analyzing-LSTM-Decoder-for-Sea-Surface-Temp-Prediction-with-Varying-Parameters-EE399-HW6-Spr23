# Analyzing Performance of LSTM-Decoder for Sea Surface Temperature Prediction with Varying Parameters

**Author:** Brendan Oquist <br>
**Abstract:** This project is a part of EE 399 Spring Quarter 2023. It explores the use of an LSTM-Decoder neural network model to predict sea-surface temperatures. The performance of the model is examined with respect to varying factors such as time lag, level of noise, and the number of sensors. 

## I. Introduction and Overview
In this project, we employ the use of the LSTM-Decoder model on sea-surface temperature data. The training and testing process involve experimenting with different time lags, adding Gaussian noise to the data, and analyzing the impact of the number of sensors on the model's performance. The goal is to understand how these factors influence the accuracy and reliability of the LSTM-Decoder model in predicting sea-surface temperatures.

## II. Theoretical Background

This section provides the necessary mathematical and conceptual foundations for understanding the use of the LSTM-Decoder neural network model for predicting sea-surface temperatures. 

### Long Short-Term Memory Networks (LSTMs)

Long Short-Term Memory networks (LSTMs) are a specific type of Recurrent Neural Network (RNN) designed to avoid the long-term dependency problem that traditional RNNs face. They accomplish this by using a system of 'gates' that control the flow of information into, within, and out of the memory cells in the network. These gates allow LSTMs to selectively remember or forget information, which makes them particularly effective for many tasks involving sequence data, like time series prediction.

### LSTM-Decoder Model

The LSTM-Decoder model in this project is a type of architecture that utilizes LSTM cells in both the encoder and decoder parts of the model. The encoder processes the input sequence and compresses the information into a context vector. This context vector is then used by the decoder to produce the output sequence. This model architecture is especially well-suited to tasks where the input and output sequences can have different lengths and are not aligned element-wise, which is typically the case in sequence-to-sequence prediction tasks.

### Performance Metrics and Variable Analysis

The performance of the LSTM-Decoder model is evaluated using different metrics. Primarily, we use the Mean Squared Error (MSE) metric, which measures the average squared difference between the predicted and actual values:

MSE = (1/n) Σ (y_i - ŷ_i)^2

where `y_i` are the actual values, `ŷ_i` are the predicted values, and `n` is the number of samples. Lower MSE values indicate better model performance.

The performance is also analyzed as a function of several variables, namely, time lag, noise level, and the number of sensors. The time lag variable is manipulated to understand its impact on prediction accuracy. Noise is added to the data to evaluate the model's robustness to noisy inputs. Finally, the impact of the number of sensors on the performance is examined, providing insights into the data quantity-quality trade-off in the model's predictive capabilities.

## III. Algorithm Implementation and Development

The code for the assignment is split into three parts: 

- The first part analyzes the performance of the model as a function of time lag.

```python
# Set the desired time lag values
lag_values = [26, 52, 78, 104, 130]

# Load the data
load_X = load_data('SST')

# Define other parameters
num_sensors = 3
load_size = load_X.shape[0]
sensor_locations = np.random.choice(load_X.shape[1], size=num_sensors, replace=False)
sc = MinMaxScaler()
sc = sc.fit(load_X[:, sensor_locations])  # Use only the selected sensor locations

# Initialize lists to store the performance results
mse_values = []

for lag in lag_values:
    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((load_size - lag, lag, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = sc.transform(load_X[i:i+lag, sensor_locations])

    # Generate datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_in = torch.tensor(all_data_in, dtype=torch.float32).to(device)
    data_out = torch.tensor(sc.transform(load_X[lag:, sensor_locations]), dtype=torch.float32).to(device)
    dataset = TimeSeriesDataset(data_in, data_out)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Train the SHRED model
    shred = models.SHRED(num_sensors, num_sensors, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset.dataset, valid_dataset.dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=False)

    # Evaluate the model on the test set
    test_recons = sc.inverse_transform(shred(dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(dataset.Y.detach().cpu().numpy())
    mse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    mse_values.append(mse)

# Plot the performance as a function of the time lag variable
plt.plot(lag_values, mse_values, marker='o')
plt.xlabel('Time Lag')
plt.ylabel('MSE')
plt.title('Performance as a Function of Time Lag')
plt.show()
```
The second part analyzes the performance as a function of noise level: 
```python
# Set the desired noise variance levels
noise_variances = [0.01, 0.05, 0.1, 0.2, 0.5]

# Load the data
load_X = load_data('SST')

# Define other parameters
num_sensors = 3
load_size = load_X.shape[0]
sensor_locations = np.random.choice(load_X.shape[1], size=num_sensors, replace=False)

# Initialize lists to store the performance results
mse_values = []

for noise_variance in noise_variances:
    # Generate noisy data
    noisy_load_X = load_X.copy()
    for i in range(load_X.shape[1]):
        noise = np.random.normal(0, noise_variance, load_X.shape[0])
        noisy_load_X[:, i] += noise

    # Scale the noisy data
    sc = MinMaxScaler()
    sc = sc.fit(noisy_load_X[:, sensor_locations])

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((load_size - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = sc.transform(noisy_load_X[i:i+lags, sensor_locations])

    # Generate datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_in = torch.tensor(all_data_in, dtype=torch.float32).to(device)
    data_out = torch.tensor(sc.transform(noisy_load_X[lags:, sensor_locations]), dtype=torch.float32).to(device)
    dataset = TimeSeriesDataset(data_in, data_out)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Train the SHRED model
    shred = models.SHRED(num_sensors, num_sensors, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset.dataset, valid_dataset.dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=False)

    # Evaluate the model on the test set
    test_recons = sc.inverse_transform(shred(dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(dataset.Y.detach().cpu().numpy())
    mse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    mse_values.append(mse)
```
Lastly, we analyze the performance as a function of number of sensors: 
```python 
# Set the desired number of sensors
num_sensors_values = [1, 2, 3, 4, 5]

# Load the data
load_X = load_data('SST')

# Define other parameters
lags = 52
load_size = load_X.shape[0]
sensor_locations = np.arange(load_X.shape[1])  # All sensor locations

# Initialize lists to store the performance results
mse_values = []

for num_sensors in num_sensors_values:
    # Randomly select sensor locations
    selected_sensor_locations = np.random.choice(sensor_locations, size=num_sensors, replace=False)

    # Scale the data
    sc = MinMaxScaler()
    sc = sc.fit(load_X[:, selected_sensor_locations])

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((load_size - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = sc.transform(load_X[i:i+lags, selected_sensor_locations])

    # Generate datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_in = torch.tensor(all_data_in, dtype=torch.float32).to(device)
    data_out = torch.tensor(sc.transform(load_X[lags:, selected_sensor_locations]), dtype=torch.float32).to(device)
    dataset = TimeSeriesDataset(data_in, data_out)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Train the SHRED model
    shred = models.SHRED(num_sensors, num_sensors, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset.dataset, valid_dataset.dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=False)

    # Evaluate the model on the test set
    test_recons = sc.inverse_transform(shred(dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(dataset.Y.detach().cpu().numpy())
    mse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    mse_values.append(mse)
```
## IV. Computational Results

We'll examine the performance of the model based on the three different parameters: time lag, noise level, and the number of sensors.

1. **Performance as a Function of Time Lag**

   Here, we observe that the model's Mean Squared Error (MSE) slightly decreases as the time lag increases. There is a marginal jump in the MSE from 0.0150 to 0.01652 as we move from a time lag of 26 to 52. This behavior suggests that there might not be a clear correlation between the time lag and the MSE. The variability could be due to the nature of the time series data or the training dynamics of the model.

![image](https://github.com/booquist/Analyzing-LSTM-Decoder-for-Sea-Surface-Temp-Prediction-with-Varying-Parameters-EE399-HW6-Spr23/assets/103399658/a26a807b-8a60-4397-9682-f0a9a91241e4)

2. **Performance as a Function of Noise Level**

   There's a clear positive correlation between the noise level and MSE. As we increase the noise level from 0.01 to 0.5, the MSE increases linearly from 0.016 to 0.029. This outcome aligns with our expectations as higher noise levels are likely to make the time series data more difficult to predict, resulting in higher prediction errors.

![image](https://github.com/booquist/Analyzing-LSTM-Decoder-for-Sea-Surface-Temp-Prediction-with-Varying-Parameters-EE399-HW6-Spr23/assets/103399658/c5d5ff0c-c92a-4b4f-85f3-f16b88470236)

3. **Performance as a Function of the Number of Sensors**

   The MSE appears to slightly increase as we raise the number of sensors from 1 to 5 (MSE from 0.0149 to 0.018). Contrary to the initial expectation that increasing the number of sensors would improve the model's performance, we observe a slight degradation. A possible explanation could be overfitting - with more sensors, the model might overfit the training data, thus reducing its performance on the validation data. Alternatively, adding more sensors might introduce more complexity or noise into the data that the model fails to generalize well.

![image](https://github.com/booquist/Analyzing-LSTM-Decoder-for-Sea-Surface-Temp-Prediction-with-Varying-Parameters-EE399-HW6-Spr23/assets/103399658/ad3baecd-4752-484d-8991-95fb1d90fe2c)

In conclusion, these findings illustrate the importance of carefully tuning the model parameters for optimal performance, including the time lag, noise level, and the number of sensors used.


