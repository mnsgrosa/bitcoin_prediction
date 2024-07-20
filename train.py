import yfinance as yf
import pandas as pd
import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LSTMAtt(nn.Module):
    def __init__(self, input_dimension, hidden_unit_dimension, output_dimension = 1, number_of_layers = 1):
        super(LSTMAtt, self).__init__()
        self.hidden_unit_dimension = hidden_unit_dimension
        self.number_of_layers = number_of_layers
        self.lstm = nn.LSTM(input_dimension, hidden_unit_dimension, number_of_layers, batch_first = True)
        self.attention = nn.Linear(hidden_unit_dimension, 1)
        self.fc = nn.Linear(hidden_unit_dimension, output_dimension)

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim = 1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim = 1)
        out = self.fc(context_vector)
        return out

class Stock:
    def __init__(self, ticket:str, device:str):
        self.device = device
        self.ticket = yf.Ticker(ticket)
        self.data = None
        self.X = None
        self.y = None
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def get_last_xtime_history(self, time_period:str = '5y', target_column:str = 'Close'):
        self.data = self.ticket.history(period = time_period)[target_column].values
        return self

    def prepare_the_data(self):
        self.data = MinMaxScaler(feature_range = (0, 1)).fit_transform(self.data.reshape(-1, 1))
        self.X = self.data[:-1]
        self.y = self.data[1:]

        self.X = self.X.reshape(-1, 1, 1)
        self.y = self.y.reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

        self.X_train_tensor = torch.tensor(self.X_train, dtype = torch.float32).to(self.device)
        self.X_test_tensor = torch.tensor(self.X_test, dtype = torch.float32).to(self.device)
        self.y_train_tensor = torch.tensor(self.y_train, dtype = torch.float32).to(self.device)
        self.y_test_tensor = torch.tensor(self.y_test, dtype = torch.float32).to(self.device)

        return self

def train_step(modelo, X_train, y_train, optimizer, criterion):
    modelo.train()
    y_hat = modelo(X_train)
    loss = criterion(y_hat, y_train)
    train_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return train_loss

def test_step(modelo, X_test, y_test, criterion):
    modelo.eval()
    with torch.inference_mode():
        y_hat = modelo(X_test)
        test_loss = criterion(y_hat, y_test).item()

    return test_loss

def workflow(modelo, X_train, X_test, y_train, y_test, optimizer, criterion, epochs = 100):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_results = train_step(modelo, X_train, y_train, optimizer, criterion)
        train_losses.append(train_results)
        test_results = test_step(modelo, X_test, y_test, criterion)
        test_losses.append(train_results)
    return train_losses, test_losses 

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo = LSTMAtt(input_dimension = 1, hidden_unit_dimension = 64).to(device)
    
    bitcoin = Stock('BTC-GBP', device)
    bitcoin.get_last_xtime_history().prepare_the_data()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)

    train_loss, test_loss = workflow(modelo, bitcoin.X_train_tensor, bitcoin.X_test_tensor, bitcoin.y_train_tensor, bitcoin.y_test_tensor, optimizer, criterion)
    
    plt.plot(range(len(train_loss)), train_loss)
    plt.plot(range(len(test_loss)), test_loss)
    plt.savefig('teste.png')