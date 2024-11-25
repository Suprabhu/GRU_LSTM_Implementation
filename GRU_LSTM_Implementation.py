#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable


# In[2]:


# Load the dataset from the local file
data = pd.read_csv('C:\\Users\\supra\\Downloads\\air+quality\\AirQualityUCI.csv', sep=';', decimal=',', na_values=-200, low_memory=False)

# Drop unwanted columns and clean the data
data = data.iloc[:, :-2]  # Remove the last two columns
data.columns = data.columns.str.strip()  # Strip whitespace from column names
data = data.dropna()  # Drop rows with missing values


# In[3]:


# Parse Date and Time into a single timestamp column
data['Timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
data = data.drop(columns=['Date', 'Time'])  # Drop original Date and Time columns


# In[4]:


# Select features and target variable
target_column = 'CO(GT)'
features = data.drop(columns=['Timestamp', target_column]).columns
data = data.loc[data[target_column] > 0]  # Remove invalid CO(GT) rows


# In[5]:


# Normalize the data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
data[target_column] = scaler.fit_transform(data[[target_column]])


# In[7]:


def create_sequences(data, target, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Ensure we're selecting the correct range for sequences
        X.append(data.iloc[i:i+seq_length].values)  # Use `.iloc` for proper row indexing
        y.append(data.iloc[i+seq_length][target])  # Use `.iloc` to ensure valid row access
    return np.array(X), np.array(y)



# In[8]:


seq_length = 10
X, y = create_sequences(data, target_column, seq_length)


# In[9]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Ensure X and y are numeric types before converting to tensors
X, y = create_sequences(data, target_column, seq_length)

# Check and convert X and y to numpy arrays of floats if needed
X = np.array(X, dtype=np.float32)  # Convert to float32 to ensure compatibility
y = np.array(y, dtype=np.float32)  # Convert to float32

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)



# In[12]:


# Select features and target variable
target_column = 'CO(GT)'
features = data.drop(columns=['Timestamp', target_column]).columns  # Exclude 'Timestamp' and target column

# Normalize the data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
data[target_column] = scaler.fit_transform(data[[target_column]])

# Prepare input sequences for time series prediction
def create_sequences(data, target, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][features].values)  # Use only numeric features
        y.append(data.iloc[i+seq_length][target])  # Use target value for prediction
    return np.array(X), np.array(y)

# Create sequences from the data
seq_length = 10
X, y = create_sequences(data, target_column, seq_length)

# Check the data types of X and y
print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

# Convert X and y to numpy arrays of float32
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# In[13]:


# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


# In[14]:


# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out


# In[15]:


# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.flatten(), y_train)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            val_loss = criterion(test_outputs.flatten(), y_test)
            test_loss.append(val_loss.item())
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    return train_loss, test_loss


# In[16]:


# Instantiate and train LSTM and GRU models
input_size = len(features)
hidden_size = 64
output_size = 1

lstm_model = LSTMModel(input_size, hidden_size, output_size)
gru_model = GRUModel(input_size, hidden_size, output_size)

print("Training LSTM Model...")
lstm_train_loss, lstm_test_loss = train_model(lstm_model, X_train, y_train, X_test, y_test)

print("Training GRU Model...")
gru_train_loss, gru_test_loss = train_model(gru_model, X_train, y_train, X_test, y_test)


# In[17]:


# Plot the training and validation loss
plt.plot(lstm_train_loss, label="LSTM Training Loss")
plt.plot(lstm_test_loss, label="LSTM Validation Loss")
plt.plot(gru_train_loss, label="GRU Training Loss")
plt.plot(gru_test_loss, label="GRU Validation Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()


# In[18]:


# Predict and visualize results
def plot_predictions(model, X, y, title):
    model.eval()
    with torch.no_grad():
        predictions = model(X).flatten().numpy()
    plt.plot(predictions, label='Predicted')
    plt.plot(y, label='Actual')
    plt.legend()
    plt.title(title)
    plt.show()

plot_predictions(lstm_model, X_test, y_test, "LSTM Predictions vs Actual")
plot_predictions(gru_model, X_test, y_test, "GRU Predictions vs Actual")


# In[19]:


from sklearn.metrics import mean_squared_error

# Predict using the trained models
lstm_model.eval()  # Set LSTM model to evaluation mode
gru_model.eval()   # Set GRU model to evaluation mode

with torch.no_grad():
    # LSTM predictions
    lstm_predictions = lstm_model(X_test).flatten().numpy()
    # GRU predictions
    gru_predictions = gru_model(X_test).flatten().numpy()

# Calculate MSE for both models
lstm_mse = mean_squared_error(y_test.numpy(), lstm_predictions)
gru_mse = mean_squared_error(y_test.numpy(), gru_predictions)

print(f"LSTM Mean Squared Error: {lstm_mse}")
print(f"GRU Mean Squared Error: {gru_mse}")


# In[20]:


# Plot training and validation loss for both models
plt.plot(lstm_train_loss, label="LSTM Training Loss")
plt.plot(lstm_test_loss, label="LSTM Validation Loss")
plt.plot(gru_train_loss, label="GRU Training Loss")
plt.plot(gru_test_loss, label="GRU Validation Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()


# In[ ]:




