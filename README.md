# compressor-simulation

Dieses Repository enthält Skripte zur Simulation und Parameter-Identifikation (Fitting) des Molinaroli-Kompressormodells (Original)
und des modifizierten Kompressormodells mit hinzugefügtem Öleinfluss (Modified) aus **VCLibPy**.  
Es ist so aufgebaut, dass man das Repo klonen und anschließend die Skripte direkt ausführen kann.

## Inhalt / Ziele

- Simulation eines einzelnen Betriebspunktes (Kompressor isoliert)
- Parameter-Fitting des Molinaroli-Modells (Original/Modified) auf Basis eines Messdatensatzes
- Sensitivitätsanalyse der einzelnen Parameter
- Ausgabe der Ergebnisse als CSV (Parameter + Vorhersagen/Residuen)

## Fitting (`scripts/fit_parameters.py`)

**Fitting der 8 Parameter des Molinaroli Modells (Ua_suc_ref,Ua_dis_ref,Ua_amb,A_tot,A_dis,V_IC,alpha_loss,W_dot_loss_ref)**
- Fitting Datensatz als csv im Ordner data ablegen (Zeile 1: Einheiten; Zeile 2: Spaltennamen; ab Zeile 3: Daten)
  - Ölbezeichnung (z.B. LPG100, LPG68)
  - P1_mean (bar) – Saugdruck 
  - T1_mean (°C) – Sauggastemperatur 
  - P2_mean (bar) – Austrittsdruck 
  - Tamb_mean (°C) – Umgebungstemperatur 
  - T2_mean (°C) – gemessene Austrittstemperatur 
  - suction_mf_mean (g/s) – gemessener Massenstrom 
  - Pel_mean (W) – gemessene elektrische Leistung 
  - N (1/min) – Drehzahl
- Startparameter anpassbar in start_params.csv (optional, sonst Default-Werte)
- Fitting durchführen in fit.parameter.py
Beispielaufruf:
python scripts/fit_parameters.py --csv data/Datensatz_Fitting_1.csv --oil LPG100 --x0_csv data/start_params.csv
  - Einstellbare Optionen über python scripts/fit_parameters.py --help
  - Unter anderem model (original/modified), oil (LPG100/LPG68/all)

## Sensitivitätsanalyse (`scripts/sensitivity_analysis.py`)

**Dieses Skript führt eine Sensitivitätsanalyse der Molinaroli-Modellparameter durch. Dabei wird jeweils ein Parameter um einen festgelegten Faktor `±delta` variiert (z.B. ±5 % oder ±50 %), während alle anderen Parameter konstant bleiben. Für jeden variierten Parametersatz wird der Datensatz erneut simuliert und der Einfluss auf die Modellgüte (basierend auf den Residuen) ausgewertet.**
- Erkennen, welche Parameter die Ergebnisse (ṁ, P_el und ggf. T_dis) am stärksten beeinflussen  
- Abschätzen, welche Parameter sich für ein Fitting besonders lohnen bzw. welche schlecht identifizierbar sind
- CSV im Ordner `results/` mit Sensitivitätskennwerten pro Parameter
Beispielaufruf:
python scripts/sensitivity_analysis.py --csv data/Datensatz_Fitting_1.csv --oil LPG100 --model original --delta 0.05
- oil (LPG100/LPG68/all), model(original/modified), delta ~ Variation des Parameters

