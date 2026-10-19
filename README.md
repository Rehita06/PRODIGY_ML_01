# House Price Intelligence

An end-to-end machine learning project for residential
house price prediction using structured housing data.

## Features

- Exploratory data analysis
- Missing-value handling
- Numeric and categorical preprocessing
- Linear Regression and Ridge Regression
- Random Forest model comparison
- Cross-validation and regression metrics
- Prediction visualization
- Streamlit demonstration application

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Streamlit
- joblib

## Project Workflow

1. Inspect and clean the data.
2. Prepare numeric and categorical features.
3. Compare regression models.
4. Evaluate predictions on held-out data.
5. Run an interactive prediction demo.

## Running the Project

Install dependencies:

    python -m pip install -r requirements.txt

Train the models:

    python src/train.py

Evaluate the selected model:

    python src/evaluate.py

Launch the application:

    streamlit run app.py

## Limitations

The dataset represents homes in Ames, Iowa.
Predictions are educational estimates and are not
professional property valuations.
