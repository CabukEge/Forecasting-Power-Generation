import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# Festlegen des Arbeitsverzeichnisses und Laden der Daten
workdir = "Prediction_2019_2019/"
data = pd.read_csv(workdir + 'Merged_Solar_Energy_Data.csv', parse_dates=['date'], low_memory=False)

# Filtern der Daten auf den Zeitraum 2019 bis 2022
data = data[(data['date'].dt.year >= 2019) & (data['date'].dt.year <= 2019)]

# Umwandeln der Zielvariablen-Werte in Floats und Bereinigen der Daten
data['Wind Onshore [MWh] Berechnete Auflösungen'] = pd.to_numeric(data['Wind Onshore [MWh] Berechnete Auflösungen'].str.replace('.', '').str.replace(',', '.'), errors='coerce')
data.dropna(subset=['Wind Onshore [MWh] Berechnete Auflösungen'], inplace=True)

# Definieren der Features
features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sun_zenith_angle', 'sunshine_duration', 'wind_direction', 'wind_speed']

# Erstellen der Sliding-Window-Features
window_size = 5
for feature in features:
    for window in range(1, window_size + 1):
        data[f'{feature}_lag_{window}'] = data[feature].shift(window)

# Entfernen von Zeilen mit fehlenden Werten nach der Erstellung der Sliding-Window-Features
data.dropna(inplace=True)

# Vorbereiten der Feature- und Zielvariablen
X = data[[f'{feature}_lag_{window}' for feature in features for window in range(1, window_size + 1)]]
y = data['Wind Onshore [MWh] Berechnete Auflösungen']

# Datennormalisierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature-Auswahl
selector = SelectKBest(k='all')  # 'all' benutzen, um alle Features zu behalten
X_selected = selector.fit_transform(X_scaled, y)

# Aufteilung der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, shuffle=False)

# Initialisieren des Random Forest-Modells
rf = RandomForestRegressor(random_state=42)

# Hyperparameter für GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [3, 5, 7],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'min_impurity_decrease': [0.0, 0.0001, 0.001]
}

# Durchführung der GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Auswahl des besten Modells
best_rf = grid_search.best_estimator_

# Vorhersagen des Testsets mit Random Forest
predictions_rf = best_rf.predict(X_test)

# Berechnen des RMSE und weiterer Metriken für die Ensemble-Vorhersagen
rmse = sqrt(mean_squared_error(y_test, predictions_rf))
r2 = r2_score(y_test, predictions_rf)
mae = mean_absolute_error(y_test, predictions_rf)

# Ausgabe der Ergebnisse
print("Beste Hyperparameter für Random Forest:", grid_search.best_params_)
print("RMSE für das Ensemble:", rmse)
print("R² für das Ensemble:", r2)
print("MAE für das Ensemble:", mae)

