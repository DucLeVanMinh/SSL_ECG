#!/bin/bash

cd /content/SSL_ECG/datasets/cinc20

# download and unzip 
kaggle datasets download -d bjoernjostein/georgia-12lead-ecg-challenge-database
unzip -q georgia-12lead-ecg-challenge-database.zip -d .
rm -r georgia-12lead-ecg-challenge-database.zip

kaggle datasets download -d bjoernjostein/st-petersburg-incart-12lead-arrhythmia-database
unzip -q st-petersburg-incart-12lead-arrhythmia-database.zip -d .
rm -r st-petersburg-incart-12lead-arrhythmia-database.zip

kaggle datasets download -d bjoernjostein/ptbxl-electrocardiography-database
unzip -q ptbxl-electrocardiography-database.zip -d . 
rm -r ptbxl-electrocardiography-database.zip 

kaggle datasets download -d bjoernjostein/ptb-diagnostic-ecg-database
unzip -q ptb-diagnostic-ecg-database.zip -d . 
rm -r ptb-diagnostic-ecg-database.zip 

kaggle datasets download -d bjoernjostein/china-12lead-ecg-challenge-database
unzip -q china-12lead-ecg-challenge-database.zip -d . 
rm -r china-12lead-ecg-challenge-database.zip 

kaggle datasets download -d bjoernjostein/china-physiological-signal-challenge-in-2018
unzip -q china-physiological-signal-challenge-in-2018.zip -d .
rm -r china-physiological-signal-challenge-in-2018.zip 

kaggle datasets download -d bjoernjostein/physionet-snomed-mappings
unzip -q physionet-snomed-mappings.zip -d .
rm -r physionet-snomed-mappings.zip

# build the cinc20 dataset
python build_cinc20_dataset.py
