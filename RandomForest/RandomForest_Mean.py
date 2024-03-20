import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

workdir = "Prediction_2019_2023/"

# Laden der Daten
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

        # Umwandlung der Zielvariablen in numerische Werte
        data[target] = data[target].str.replace('.', '').str.replace(',', '.').astype(float)

        # Aggregieren der Wetterdaten durch Bilden des Durchschnitts über alle Stationen pro Zeitstempel
        weather_features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse',
                            'sun_zenith_angle', 'sunshine_duration', 'wind_direction', 'wind_speed']

        data = data.groupby(['date', 'year', 'month', 'day', 'hour', 'weekday'])[
            weather_features + [target]].mean().reset_index()

        lag_features = []
        for offset in range(1, window_size + 1):
            for column in weather_features:
                feature_name = f'{column}_lag_{offset}'
                data[feature_name] = data[column].shift(offset)
                lag_features.append(feature_name)
        data.dropna(inplace=True)
        # Auswählen der Features und der Zielvariable nach der Aggregation
        feature_columns = ['year', 'month', 'day', 'hour', 'weekday'] + weather_features + lag_features
        X = data[feature_columns]
        y = data[target]

        # Aufteilen der Daten in Trainings- und Testsets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

        # Standardisierung der Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Erstellen des Random Forest Modells mit festen Hyperparametern
        rf = RandomForestRegressor(max_depth=30, max_features='sqrt', min_impurity_decrease=0.0001,
                                   min_samples_leaf=2, min_samples_split=3, n_estimators=150, random_state=42)

        # Modell mit Trainingsdaten trainieren
        rf.fit(X_train_scaled, y_train)

        # Vorhersagen für das Testset
        predictions = rf.predict(X_test_scaled)

        # Berechnen des RMSE (Root Mean Squared Error)
        rmse = sqrt(mean_squared_error(y_test, predictions))

        print(f"RMSE für {target} und Fenstergröße {window_size}: {rmse}")

        # Plot der tatsächlichen vs. vorhergesagten Werte
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label='Tatsächliche Werte')
        plt.plot(predictions, label='Vorhergesagte Werte', alpha=0.5)
        plt.title(f'Tatsächliche vs. Vorhergesagte {target} (Fenstergröße: {window_size})')
        plt.xlabel('Zeit (Index)')
        plt.ylabel(target)
        plt.legend()
        plt.savefig(
            f"{workdir}RF_Aggregated_Prediction_{target}_Window_{window_size}.png")  # Speichert den Plot als PNG
        plt.show()
