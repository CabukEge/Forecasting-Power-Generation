import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

workdir = "Prediction_2019_2023/"

# Laden der Daten
data_original = pd.read_csv(workdir + 'Merged_Solar_Energy_Data.csv', parse_dates=['date'], low_memory=False)
targets = ['Wind Onshore [MWh] Berechnete Auflösungen', 'Wind Offshore [MWh] Berechnete Auflösungen', 'Photovoltaik [MWh] Berechnete Auflösungen']
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

        # Aufteilen der Daten in Trainings- und Testsets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

        # Standardisierung der Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Erstellen des Random Forest Modells mit festen Hyperparametern
        rf = RandomForestRegressor(
            max_depth=30,
            max_features='sqrt',
            min_impurity_decrease=0.0001,
            min_samples_leaf=2,
            min_samples_split=3,
            n_estimators=150,
            random_state=42
        )

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
       # plt.plot(predictions, label='Vorhergesagte Werte', alpha=0.5)
        plt.title(f'Tatsächliche Werte')
        plt.xlabel('Zeit (Index)')
        plt.ylabel(f'{target}')
        plt.legend()
        plt.savefig(f"{workdir}Prediction_Window_{target}_{window_size}.png") # Speichert den Plot als PNG
        plt.show()

        # Nach dem Training des Modells die Feature-Wichtigkeiten extrahieren
        #feature_importances = rf.feature_importances_

        ## Die Indizes der Features sortieren, basierend auf ihrer Wichtigkeit, in absteigender Reihenfolge
        #sorted_indices = np.argsort(feature_importances)[::-1]
#
        ## Die 8 wichtigsten Features anzeigen
        #print("Die 8 wichtigsten Features und ihre Wichtigkeiten:")
        #for idx in sorted_indices[:8]:
        #    print(f"{feature_columns[idx]}: {feature_importances[idx]:.4f}")
#
        ## Optional: Plot der Feature-Wichtigkeiten
        #plt.figure(figsize=(10, 6))
        #plt.title(f"Feature Wichtigkeiten für {target} mit Window {window_size}")
        #plt.bar(range(8), feature_importances[sorted_indices[:8]], align="center")
        #plt.xticks(range(8), np.array(feature_columns)[sorted_indices[:8]], rotation=45)
        #plt.xlim([-1, 8])
        #plt.tight_layout()
        #plt.show()
