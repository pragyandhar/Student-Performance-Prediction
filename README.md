# Student Performance Prediction

A comprehensive end-to-end machine learning project that predicts student exam performance based on various demographic and academic factors. This application uses multiple regression models and provides both a command-line training pipeline and a Flask web interface for making predictions.

## Project Overview

This project demonstrates a complete ML pipeline including data ingestion, exploratory data analysis, data transformation, model training with hyperparameter tuning, and model deployment. It aims to predict student math exam scores based on factors such as gender, race/ethnicity, parental education level, lunch type, and reading/writing scores.

## Features

- **Multiple ML Models**: Implements 7 different regression algorithms:
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Linear Regression
  - XGBoost Regressor
  - CatBoost Regressor
  - AdaBoost Regressor

- **Automated Model Selection**: Uses GridSearchCV for hyperparameter tuning and automatically selects the best performing model

- **Comprehensive Data Pipeline**:
  - Data ingestion and train-test splitting
  - Numerical feature scaling (median imputation + StandardScaler)
  - Categorical feature encoding (most frequent imputation + OneHotEncoding)
  - Preprocessor serialization for consistent prediction

- **Web Interface**: Flask-based user-friendly web application for making predictions

- **Robust Error Handling**: Custom exception handling with detailed error tracking

- **Logging**: Comprehensive logging system for debugging and monitoring

## Project Structure

```
.
├── app.py                          # Flask application entry point
├── requirements.txt                # Project dependencies
├── setup.py                        # Setup configuration
├── README.md                       # Project documentation
│
├── artifacts/                      # Generated model and data files
│   ├── data.csv                   # Raw dataset
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Test dataset
│   ├── model.pkl                  # Trained model
│   └── preprocessor.pkl           # Data preprocessor object
│
├── src/                           # Source code directory
│   ├── __init__.py
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   ├── utils.py                   # Utility functions (save/load objects, model evaluation)
│   │
│   ├── components/                # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Feature preprocessing and transformation
│   │   └── model_trainer.py       # Model training and selection
│   │
│   └── pipeline/                  # ML pipelines
│       ├── __init__.py
│       ├── train_pipeline.py      # Training pipeline (for future use)
│       └── predict_pipeline.py    # Prediction pipeline
│
├── notebook/                      # Jupyter notebooks
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb  # Exploratory data analysis
│   ├── 2. MODEL TRAINING.ipynb           # Model training notebook
│   └── data/
│       └── stud.csv               # Source student data
│
└── templates/                     # Flask HTML templates
    ├── index.html                 # Landing page
    └── home.html                  # Prediction form page
```

## Dataset Features

The project uses a student performance dataset with the following input features:

**Demographic Factors:**
- `gender`: Male or Female
- `race_ethnicity`: Group A through E
- `parental_level_of_education`: Various education levels

**Academic Factors:**
- `lunch`: Standard or Free/reduced
- `test_preparation_course`: Completed or Not completed
- `reading_score`: Student's reading score (0-100)
- `writing_score`: Student's writing score (0-100)

**Target Variable:**
- `math_score`: Student's math exam score (what we're predicting)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Student-Performance-Prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the data ingestion component which will execute the complete pipeline:

```bash
python src/components/data_ingestion.py
```

This will:
1. Load the student dataset
2. Split data into train (80%) and test (20%) sets
3. Transform and preprocess features
4. Train all 7 regression models with hyperparameter tuning
5. Select and save the best performing model
6. Save the preprocessor for consistent predictions

### Making Predictions via Web Interface

1. **Start the Flask application:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Fill in the student information:**
   - Select gender
   - Select race/ethnicity
   - Select parental education level
   - Select lunch type
   - Select test preparation status
   - Enter reading score
   - Enter writing score

4. **Submit the form** to get the predicted math exam score

### Making Predictions Programmatically

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd

# Create custom data
data = CustomData(
    gender='male',
    race_ethnicity='group B',
    parental_level_of_education="bachelor's degree",
    lunch='standard',
    test_preparation_course='completed',
    reading_score=85,
    writing_score=88
)

# Convert to dataframe
pred_df = data.get_data_as_data_frame()

# Make prediction
predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(pred_df)
print(f"Predicted Math Score: {prediction[0]}")
```

## Dependencies

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML Models**: scikit-learn, xgboost, catboost
- **Web Framework**: Flask
- **Serialization**: dill, pickle

See `requirements.txt` for complete list and versions.

## Architecture

### Data Pipeline

```
Raw Data
    ↓
Data Ingestion (Train-Test Split 80-20)
    ↓
Data Transformation
├─ Numerical Features → Imputation (Median) → StandardScaler
└─ Categorical Features → Imputation (Most Frequent) → OneHotEncoding → StandardScaler
    ↓
Model Training (GridSearchCV with 7 models)
    ↓
Best Model Selection (Highest R² Score)
    ↓
Model & Preprocessor Serialization
```

### Prediction Flow

```
User Input
    ↓
CustomData Class → DataFrame
    ↓
Load Preprocessor (preprocessor.pkl)
    ↓
Feature Transformation
    ↓
Load Trained Model (model.pkl)
    ↓
Make Prediction
    ↓
Return Math Score Prediction
```

## Model Evaluation

The project uses **GridSearchCV** with 3-fold cross-validation to:
- Perform hyperparameter tuning for each model
- Find optimal parameters for each algorithm
- Evaluate models using **R² Score** on test data
- Select the model with the highest test R² score

### Hyperparameter Ranges

- **Decision Tree**: Criterion options (squared_error, friedrich_mse, absolute_error, poisson)
- **Random Forest**: n_estimators [8, 16, 32, 64, 128, 256]
- **Gradient Boosting**: learning_rate [0.001, 0.05, 0.01, 0.1], subsample [0.6-0.9], n_estimators [8-256]
- **XGBoost**: learning_rate [0.001-0.1], n_estimators [8-256]
- **CatBoost**: depth [6, 8, 10], learning_rate [0.01, 0.05, 0.1], iterations [30, 50, 100]
- **AdaBoost**: learning_rate [0.001-0.5], n_estimators [8-256]

## Error Handling

The project includes custom exception handling:
- **CustomException**: Captures error details including file name, line number, and error message
- **Logging**: All operations are logged with timestamps for debugging
- **Graceful Failure**: Detailed error messages help identify issues quickly

## Logging

Logs are generated in the `logs/` directory with timestamped filenames containing:
- Timestamps
- Line numbers
- Module names
- Log levels (INFO, ERROR, etc.)
- Detailed messages

## Key Components Explanation

### 1. **Data Ingestion** (`src/components/data_ingestion.py`)
- Reads student dataset from CSV
- Performs 80-20 train-test split
- Saves processed datasets to artifacts folder

### 2. **Data Transformation** (`src/components/data_transformation.py`)
- Creates preprocessing pipeline
- Handles numerical features: median imputation + standard scaling
- Handles categorical features: mode imputation + one-hot encoding + scaling
- Saves preprocessor for future predictions

### 3. **Model Trainer** (`src/components/model_trainer.py`)
- Trains 7 different regression models
- Performs hyperparameter tuning with GridSearchCV
- Evaluates models using R² score
- Saves best performing model

### 4. **Prediction Pipeline** (`src/pipeline/predict_pipeline.py`)
- Loads trained model and preprocessor
- Transforms input features using saved preprocessor
- Makes predictions on new data
- Includes CustomData class for user input handling

### 5. **Flask Application** (`app.py`)
- Provides web interface for predictions
- Routes:
  - `/` - Landing page
  - `/predictdata` - Prediction form and results

## Configuration Files

### setup.py
Defines project metadata and automatically installs requirements from `requirements.txt`

### requirements.txt
Lists all Python package dependencies needed for the project

## Future Improvements

- Implement cross-validation for more robust evaluation
- Add feature importance analysis
- Create API endpoints for programmatic predictions
- Add data visualization dashboard
- Implement model versioning
- Add model performance monitoring
- Create unit tests and integration tests
- Support for model retraining with new data

## Author

**Pragyan** - Original Project Creator

## License

This project is open source and available for educational purposes.

## Contact & Support

For questions, issues, or suggestions, please refer to the project repository or contact the development team.
