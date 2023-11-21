import polars as pl
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import json

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
    start_date="2023-01-01",
    end_date="2023-02-01",
    settings=settings
)

# Daten abrufen
data = request.discover()

# Konvertieren des Dictionaries in einen formatierten String
formatted_data = json.dumps(data, indent=4)

# Ausgabe des formatierten Strings
print(formatted_data)