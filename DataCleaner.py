import pandas as pd
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import os


workdir="Prediction_2019_2023/"
energy_csv=workdir+'Realisierte_Erzeugung_201901010000_202312312359_Stunde.csv'
###
# GET DATA
###
#
#
#
#
# Konfigurationseinstellungen
settings = Settings(
    ts_shape="long",  # tidy data
    ts_humanize=True,  # humanized parameters
    ts_si_units=True  # convert values to SI units
)

print("Starte Datenabruf für Stationsinformationen...")

# Anfrage für Stationsinformationen
stations_request = DwdObservationRequest(
    parameter=["solar", "wind"],
    resolution="hourly",
    start_date="2019-01-01",
    end_date="2023-12-31",
    settings=settings
)

# Liste aller Stationen abrufen und in Pandas DataFrame konvertieren
stations_df = stations_request.all().df.to_pandas()
stations_df.to_csv('stations.csv', index=False)
print("Stationsinformationen erfolgreich abgerufen und in 'stations.csv' gespeichert.")

final_csv_file = workdir+'Collected_Solar_Climate_Summary_Data.csv'

print("Starte Datenabruf für Wetterdaten von allen Stationen...")

# Datenabfrage für alle Stationen
for station_id in stations_df['station_id']:
    print(f"Starte Datenabruf für Station {station_id}...")
    try:
        request = DwdObservationRequest(
            parameter=["solar", "wind"],
            resolution="hourly",
            start_date="2019-01-01",
            end_date="2023-12-31",
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



###
# Pivotisieren
###
#
#
#
# Daten laden
data = pd.read_csv(workdir+'Collected_Solar_Climate_Summary_Data.csv')

# Daten pivotisieren, einschließlich station_id und dataset im Index
pivoted_data_with_station_id = data.pivot_table(index=['station_id', 'date'], columns='parameter', values='value', aggfunc='first')
pivoted_data = data.pivot_table(index='date', columns='parameter', values='value', aggfunc='first')

# Ergebnis speichern
pivoted_data_with_station_id.to_csv(workdir+'Pivoted_Data_with_StationID.csv')





###
# Drop Nulls
###
#
#
#
# Pfad zur Eingabe-CSV-Datei
input_file = workdir+'Pivoted_Data_with_StationID.csv'

# Pfad zur Ausgabe-CSV-Datei ohne Nullwerte
output_file = workdir+'Pivoted_Data_with_StationID_without_Nulls.csv'

# Laden der Daten aus der CSV-Datei
data = pd.read_csv(input_file)

# Entfernen aller Zeilen, die Nullwerte enthalten
filtered_data = data.dropna()

# Speichern der gefilterten Daten in einer neuen CSV-Datei
filtered_data.to_csv(output_file, index=False)

print(f"Die gefilterte Datei wurde erfolgreich gespeichert als: {output_file}")



###
# Change Datetime in Energy csv
###
#
#
#

# Laden der Energieerzeugungsdaten
energy_data_path = energy_csv
energy_data = pd.read_csv(energy_data_path, delimiter=';', decimal=',')

# Umwandeln der Datums- und Zeitangaben in eine einheitliche 'date'-Spalte
energy_data['date'] = pd.to_datetime(energy_data['Datum'] + ' ' + energy_data['Anfang'], dayfirst=True).dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

# Entfernen der Spalten 'Datum', 'Anfang' und 'Ende'
energy_data = energy_data.drop(['Datum', 'Anfang', 'Ende'], axis=1)

# Pfad für die gespeicherte Datei festlegen
output_path = workdir+'Adjusted_Realisierte_Erzeugung_with_New_Date.csv'

# Angepasste Daten in einer CSV-Datei speichern
energy_data.to_csv(output_path, index=False)

# Rückgabe des Pfads zur neuen Datei
print(f"Die angepassten realisierten Stromerzeugungsdaten wurden erfolgreich gespeichert: {output_path}")




###
# Mergen
###
#
#
#

# Pfad zur angepassten realisierten Stromerzeugungsdaten-CSV
adjusted_energy_data_path = workdir+'Adjusted_Realisierte_Erzeugung_with_New_Date.csv'

# Pfad zur Solar-Daten-CSV ohne Nullwerte
solar_data_path = workdir+'pivot_data.csv'

# Laden der angepassten realisierten Stromerzeugungsdaten
adjusted_energy_data = pd.read_csv(adjusted_energy_data_path)

# Laden der Solar-Daten ohne Nullwerte
solar_data = pd.read_csv(solar_data_path)

# Durchführen des Merges auf Basis der 'date'-Spalte
merged_data = pd.merge(solar_data, adjusted_energy_data, on='date', how='inner')

# Pfad für die gespeicherte, zusammengeführte Datei festlegen
merged_output_path = workdir+'Pivoted_Merged_Solar_Energy_Data.csv'

# Zusammengeführte Daten in einer CSV-Datei speichern
merged_data.to_csv(merged_output_path, index=False)

print(f"Die zusammengeführten Daten wurden erfolgreich gespeichert: {merged_output_path}")
