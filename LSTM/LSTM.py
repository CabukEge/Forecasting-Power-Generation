import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates

# CUDA-Unterstützung prüfen und das entsprechende Gerät auswählen
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("using cuda")

# Konfiguration
workdir = "Prediction_LSTM_2019-2019/"
filename = 'Merged_Solar_Energy_Data.csv'
data_path = workdir + filename
start_date = "2019-01-01"
end_date = "2019-12-31"
targets = ['Wind Onshore [MWh] Berechnete Auflösungen', 'Wind Offshore [MWh] Berechnete Auflösungen', 'Photovoltaik [MWh] Berechnete Auflösungen']
window_sizes = [1, 3, 5, 8, 10]  # Verschiedene zu testende Fenstergrößen

# Grundlegende LSTM-Konfiguration
input_dim = 0
hidden_dim = 64
layer_dim = 1
output_dim = 1
batch_size = 64
n_epochs = 50
learning_rate = 0.001

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

# Daten laden und vorbereiten
data = pd.read_csv(data_path, low_memory=False)
data['date'] = pd.to_datetime(data['date'])
mask = (data['date'] >= start_date) & (data['date'] <= end_date)
data = data.loc[mask]

# Zielvariablen in Float konvertieren
for target in targets:
    data[target] = data[target].str.replace('.', '').str.replace(',', '.').astype(float)

# Datumskomponenten extrahieren
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday

# Funktionen zur Datenverarbeitung definieren
def prepare_data(data, feature_columns, target, test_size=0.2):
    # Features und Zielvariable auswählen
    X = data[feature_columns].values
    y = data[target].values.reshape(-1, 1)

    # Daten skalieren
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler_y

def train_model(X_train, y_train, X_test, y_test, scaler_y):
    # Tensoren für PyTorch vorbereiten
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # DataLoader für das Training vorbereiten
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    # Modell initialisieren und trainieren
    input_dim = X_train.shape[1]
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f'Anzahl der Features: {X_train.shape[1]}')
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

    # Vorhersagen auf Testdaten
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor.unsqueeze(1))
        predictions = scaler_y.inverse_transform(predictions.cpu().numpy())
        y_test_inverse = scaler_y.inverse_transform(y_test)
    return predictions, y_test_inverse

# Hauptloop für verschiedene Fenstergrößen
for window_size in window_sizes:
    print(f'\nFenstergröße: {window_size}')
    feature_columns = ['year', 'month', 'day', 'weekday']
    lag_features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse',
                    'sun_zenith_angle', 'sunshine_duration', 'wind_direction', 'wind_speed']
    for offset in range(1, window_size + 1):
        for column in lag_features:
            feature_name = f'{column}_lag_{offset}'
            data[feature_name] = data.groupby('station_id')[column].shift(offset)
            feature_columns.append(feature_name)

    data.dropna(inplace=True)
    input_dim = len(feature_columns) - feature_columns.index('year')  # Berechne input_dim basierend auf der Anzahl der Features

    # Für jede Zielvariable das Modell trainieren und Ergebnisse visualisieren
    for target in targets:
        print(f'Training für Ziel: {target}')
        X_train, X_test, y_train, y_test, scaler_y = prepare_data(data, feature_columns, target)
        predictions, y_test_inverse = train_model(X_train, y_train, X_test, y_test, scaler_y)

        # Aggregierte Vorhersagen für jede Zeitstempelstation
        agg_predictions = np.mean(predictions, axis=1)

        # Aggregierte Vorhersagen für jeden Tag
        grouped_data = pd.DataFrame({'Tatsächlich': y_test_inverse.flatten(), 'Vorhersage': agg_predictions.flatten()}, index=data['date'].iloc[-len(predictions):])
        grouped_data = grouped_data.groupby(grouped_data.index.date).mean()

        # Visualisierung
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_data.index.to_numpy(), grouped_data['Tatsächlich'].values, label='Tatsächlich', color='blue')
        plt.plot(grouped_data.index.to_numpy(), grouped_data['Vorhersage'].values, label='Vorhersage', color='red',
                 alpha=0.5)
        test_start_date = data['date'].iloc[len(X_train)].strftime("%Y-%m-%d")
        plt.axvline(x=pd.to_datetime(test_start_date), color='green', linestyle='--', label='Beginn der Testdaten')

        plt.title(
            f'Vorhersage vs Tatsächlich für {target} \n (Fenstergröße: {window_size})\nStartdatum: {grouped_data.index[0].strftime("%Y-%m-%d")}, Enddatum: {grouped_data.index[-1].strftime("%Y-%m-%d")}')
        plt.xlabel('Zeitindex')
        plt.ylabel('Wert')
        plt.legend()
        plt.show()

        # Ergebnisse speichern
        results_df = pd.DataFrame({'Datum': grouped_data.index, 'Tatsächlich': grouped_data['Tatsächlich'], 'Vorhersage': grouped_data['Vorhersage']})
        csv_path = f'{workdir}{target}_window_{window_size}_predictions_aggregated.csv'
        results_df.to_csv(csv_path, index=False)
        print(f'Ergebnisse gespeichert in: {csv_path}')

