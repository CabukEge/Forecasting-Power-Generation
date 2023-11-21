import polars as pl
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest

# Konfigurationseinstellungen
settings = Settings(
    ts_shape="long",  # tidy data
    ts_humanize=True,  # humanized parameters
    ts_si_units=True  # convert values to SI units
)

# Anfrage für spezifische Parameter
request = DwdObservationRequest(
    parameter=["solar", "wind_speed"],
    resolution="daily",
    start_date="2022-01-01",
    end_date="2022-02-01",
    settings=settings,
).filter_by_station_id('5906')

data = request.values.all()

print(data)

# Konvertiere die Daten in einen Pandas DataFrame und speichere sie als CSV-Datei
data.df.to_pandas().to_csv('data.csv', index=False)

