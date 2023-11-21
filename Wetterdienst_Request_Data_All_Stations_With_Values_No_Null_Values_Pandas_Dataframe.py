import pandas as pd
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest

# Konfigurationseinstellungen
settings = Settings(
    ts_shape="long",  # tidy data
    ts_humanize=True,  # humanized parameters
    ts_si_units=True  # convert values to SI units
)

print("Starte Datenabruf für Stationsinformationen...")

# Anfrage für Stationsinformationen
stations_request = DwdObservationRequest(
    parameter=["solar", "wind_speed"],
    resolution="daily",
    start_date="2023-01-01",
    end_date="2023-02-01",
    settings=settings
)

# Liste aller Stationen abrufen und in Pandas DataFrame konvertieren
stations_df = stations_request.all().df.to_pandas()
print("Stationsinformationen erfolgreich abgerufen und in 'stations.csv' gespeichert.")
stations_df.to_csv('stations.csv', index=False)

# Erstelle eine leere DataFrame für die gesammelten Daten
collected_data = pd.DataFrame()

print("Starte Datenabruf für Wetterdaten von Stationen mit gültigen Daten...")

# Zähler für die Anzahl der verarbeiteten Stationen
processed_stations = 0

# Datenabfrage für Stationen mit gültigen Daten
for station_id in stations_df['station_id']:
    if processed_stations >= 20:
        break

    print(f"Starte Datenabruf für Station {station_id}...")
    try:
        request = DwdObservationRequest(
            parameter=["solar", "wind_speed"],
            resolution="daily",
            start_date="2022-01-01",
            end_date="2022-02-01",
            settings=settings
        ).filter_by_station_id(station_id)

        data_df = request.values.all().df.to_pandas()

        # Filtere leere Werte heraus
        data_df = data_df.dropna(subset=['value', 'quality'])

        if not data_df.empty:
            data_df['station_id'] = station_id
            collected_data = pd.concat([collected_data, data_df])
            processed_stations += 1
            print(f"Daten für Station {station_id} erfolgreich abgerufen und hinzugefügt.")
        else:
            print(f"Keine gültigen Daten für Station {station_id} verfügbar.")
    except Exception as e:
        print(f"Fehler bei der Abfrage der Station {station_id}: {e}")

print("Gültige Wetterdaten erfolgreich abgerufen. Speichere Daten in 'collected_data.csv'.")

# Speichere die gesammelten Daten in einer CSV-Datei
collected_data.to_csv('collected_data.csv', index=False)
print("Datenabruf und Speicherung abgeschlossen.")
