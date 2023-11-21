import polars as pl
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

# Liste aller Stationen abrufen
stations = stations_request.all().df
print("Stationsinformationen erfolgreich abgerufen.")
stations.to_pandas().to_csv('stations.csv', index=False)
print("Stationsinformationen in 'stations.csv' gespeichert.")

# Erstelle eine leere DataFrame für die gesammelten Daten
collected_data = pl.DataFrame()

print("Starte Datenabruf für Wetterdaten...")
# Anfrage für spezifische Parameter
# ...
for station_id in stations['station_id']:
    try:
        request = DwdObservationRequest(
            parameter=["solar", "wind_speed"],
            resolution="daily",
            start_date="2022-01-01",
            end_date="2022-02-01",
            settings=settings
        ).filter_by_station_id(station_id)

        data = request.values.all().df
        if not data.is_empty():
            # Korrigierte Zeile: Verwenden Sie with_columns, um die station_id-Spalte hinzuzufügen
            data = data.with_columns([pl.lit(station_id).alias('station_id')])
            collected_data = pl.concat([collected_data, data])
            print(f"Daten für Station {station_id} erfolgreich abgerufen.")
        else:
            print(f"Keine Daten für Station {station_id} verfügbar.")
    except Exception as e:
        print(f"Fehler bei der Abfrage der Station {station_id}: {e}")

# Speichere die gesammelten Daten in einer CSV-Datei
collected_data.to_pandas().to_csv('collected_data.csv', index=False)
print("Alle Wetterdaten in 'collected_data.csv' gespeichert.")
print("Datenabruf abgeschlossen.")
