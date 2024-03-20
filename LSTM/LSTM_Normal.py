import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

workdir = "Prediction_2019_2023/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def prepare_data(data, feature_columns, target, test_size=0.25):
    X = data[feature_columns].values
    y = data[target].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, shuffle=False)

    return torch.tensor(X_train, dtype=torch.float).unsqueeze(1), torch.tensor(X_test, dtype=torch.float).unsqueeze(
        1), torch.tensor(y_train, dtype=torch.float), torch.tensor(y_test, dtype=torch.float), scaler_y


data_original = pd.read_csv(workdir + 'Merged_Solar_Energy_Data.csv', parse_dates=['date'], low_memory=False)
targets = ['Wind Onshore [MWh] Berechnete Auflösungen', 'Wind Offshore [MWh] Berechnete Auflösungen',
           'Photovoltaik [MWh] Berechnete Auflösungen']
window_sizes = [1, 3, 5, 8, 10]

for target in targets:
    for window_size in window_sizes:
        # Datensatz für jede Fenstergröße neu vorbereiten
        data = data_original.copy()

        # Datumskomponenten und Feature Engineering
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['hour'] = data['date'].dt.hour
        data['weekday'] = data['date'].dt.weekday

        # Beschränkung des Datensatzes auf das Jahr 2020
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        data = data.loc[mask]

        data[target] = data[target].str.replace(
            '.', '').str.replace(',', '.').astype(float)

        # Sliding Window-Feature-Engineering
        feature_columns = ['year', 'month', 'day', 'hour', 'weekday']
        lag_features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse',
                        'sun_zenith_angle', 'sunshine_duration', 'wind_direction', 'wind_speed']
        for offset in range(1, window_size + 1):
            for column in lag_features:
                feature_name = f'{column}_lag_{offset}'
                data[feature_name] = data.groupby('station_id')[column].shift(offset)
                feature_columns.append(feature_name)

        # Entfernen von Zeilen mit NaN-Werten nach dem Hinzufügen von verzögerten Features
        data.dropna(inplace=True)

        # Auswählen der Features und der Zielvariable
        X = data[feature_columns]
        y = data[target]

        X_train, X_test, y_train, y_test, scaler_y = prepare_data(data, feature_columns, target)

        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False)
        test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

        model = LSTMModel(input_dim=len(feature_columns), hidden_dim=64, layer_dim=1, output_dim=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(-1))
                loss.backward()
                optimizer.step()

        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu())

        predictions = np.concatenate(predictions, axis=0)
        predictions = scaler_y.inverse_transform(predictions)
        y_test = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

        rmse = sqrt(mean_squared_error(y_test, predictions))
        print(f'RMSE for {target} with window size {window_size}: {rmse}')
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Tatsächliche Werte', color='blue')
        plt.plot(predictions, label='Vorhergesagte Werte', color='red', alpha=0.7)
        plt.title(f'Tatsächliche vs. Vorhergesagte {target} (Fenstergröße: {window_size})')
        plt.xlabel('Zeit (Index)')
        plt.ylabel(target)
        plt.legend()
        plt.savefig(f"{workdir}LSTM_Prediction_{target}_Window_{window_size}.png")
        plt.show()
