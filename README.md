# Trajectory Prediction Project

## Overview
This project processes AIS data from vessels and predicts their trajectories using machine learning models. The system:
1. Processes raw AIS data from [NACTrans Vessel Tracking](https://nactrans.myvessel.cn/)
2. Generates machine learning-ready datasets
3. Produces trajectory predictions in Excel format
4. Provides a web interface for visualization

## Implementation

### Project Description
- **Data Processing**: `data_processing.py` converts raw AIS data into processed formats, generating:
  - `ais_processed_scalers.pkl` (normalization parameters)
  - `ais_processed_train.pkl` (training data)
  - `ais_processed_valid.pkl` (validation data) 
  - `ais_processed_test.pkl` (testing data)

- **Model Prediction**: `models.py` processes the data and outputs:
  - `Predicted_trajectory_ofTest.xlsx` (final predictions)

- **Web Interface**: `app.py` provides interactive visualization via Streamlit

### Requirements
Install all dependencies with:
```bash
pip install -r requirements.txt
