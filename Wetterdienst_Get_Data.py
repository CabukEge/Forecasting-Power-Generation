import pandas as pd
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import os

# Konfigurationseinstellungen
settings = Settings(
    ts_shape="long",  # tidy data
    ts_humanize=True,  # humanized parameters
    ts_si_units=True  # convert values to SI units
)

print("Starte Datenabruf für Stationsinformationen...")

# Anfrage für Stationsinformationen
stations_request = DwdObservationRequest(
    parameter=["solar", "climate_summary"],
    resolution="daily",
    start_date="2015-01-01",
    end_date="2023-10-01",
    settings=settings
)

# Liste aller Stationen abrufen und in Pandas DataFrame konvertieren
stations_df = stations_request.all().df.to_pandas()
stations_df.to_csv('stations.csv', index=False)
print("Stationsinformationen erfolgreich abgerufen und in 'stations.csv' gespeichert.")

final_csv_file = 'Collected_Solar_Climate_Summary_Data.csv'

print("Starte Datenabruf für Wetterdaten von allen Stationen...")

# Datenabfrage für alle Stationen
for station_id in stations_df['station_id']:
    print(f"Starte Datenabruf für Station {station_id}...")
    try:
        request = DwdObservationRequest(
            parameter=["solar", "climate_summary"],
            resolution="daily",
            start_date="2015-01-01",
            end_date="2023-10-01",
            settings=settings
        ).filter_by_station_id(station_id)

        data_df = request.values.all().df.to_pandas()

        # Filtere leere Werte heraus
        data_df = data_df.dropna(subset=['value', 'quality'])

        if not data_df.empty:
            data_df['station_id'] = station_id
            # Füge die Daten zur finalen CSV-Datei hinzu
            if not os.path.exists(final_csv_file):
                data_df.to_csv(final_csv_file, index=False)
            else:
                data_df.to_csv(final_csv_file, mode='a', header=False, index=False)
            print(f"Daten für Station {station_id} erfolgreich abgerufen und hinzugefügt.")
        else:
            print(f"Keine gültigen Daten für Station {station_id} verfügbar.")
    except Exception as e:
        print(f"Fehler bei der Abfrage der Station {station_id}: {e}")

print("Datenabruf und Speicherung abgeschlossen.")
