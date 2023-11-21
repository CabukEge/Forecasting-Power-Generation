from wetterdienst import Settings
import polars as pl
from wetterdienst.provider.dwd.observation import DwdObservationRequest

# Konfigurationseinstellungen
settings = Settings(
    ts_shape="long",  # tidy data
    ts_humanize=True,  # humanized parameters
    ts_si_units=True  # convert values to SI units
)

# Anfrage für Stationsinformationen
request = DwdObservationRequest(
    parameter=["solar", "wind_speed"],
    resolution="daily",
    start_date="2023-01-01",
    end_date="2023-02-01",
    settings=settings
)

# Liste aller Stationen abrufen
stations = request.all()

# Konvertiere die Daten in einen Pandas DataFrame und speichere sie als CSV-Datei
stations_df = stations.to_pandas()
stations_df.to_csv('stations.csv', index=False)

# Drucke die Stationen-ID und Namen (Optionale Ausgabe)
print(stations_df[["station_id", "name"]])
