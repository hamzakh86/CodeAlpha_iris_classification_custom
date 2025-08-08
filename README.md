# Iris Flower Classification - Custom Project

## Description

This project implements a complete Iris flower classification solution with enhancements over the reference project. It includes several machine learning algorithms, an interactive Streamlit interface, and advanced visualizations.

## Features

*   7 different classification algorithms
*   Interactive web interface with Streamlit
*   Advanced data visualizations
*   Automatic hyperparameter optimization
*   Comprehensive evaluation metrics
*   Best model saving

## Installation

### Prerequisites

*   Python 3.7 or newer
*   pip (Python package manager)

### Installation Steps

1.  **Decompress the archive**
    
    ```shell
    tar -xzf iris_classification_custom.tar.gz
    cd iris_classification_custom
    ```
    
2.  **Install dependencies**
    
    ```shell
    pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
    ```
    
3.  **Run model training (optional)**
    
    ```shell
    python iris_classification_enhanced.py
    ```
    
4.  **Launch the Streamlit application**
    
    ```shell
    streamlit run streamlit_app.py
    ```
    
5.  **Access the application** Open your browser and go to the displayed address (usually [http://localhost:8501]())
    

## Project Structure

    iris_classification_custom/
    ├── iris_classification_enhanced.py    # Main training script
    ├── streamlit_app.py                   # Streamlit web application
    ├── iris_classification_main.ipynb     # Modified Jupyter Notebook
    ├── best_iris_model.pkl               # Pre-trained model
    ├── iris_data_visualization.png       # Data visualizations
    ├── confusion_matrix.png              # Confusion matrix
    └── model_comparison.png              # Model comparison
    

## Usage

### Python Script

```shell
python iris_classification_enhanced.py
```

This script executes the complete pipeline: data loading, model training, evaluation, and saving.

### Streamlit Application

```shell
streamlit run streamlit_app.py
```

Launches the interactive web interface for:

*   Exploring data
*   Making real-time predictions
*   Visualizing results

### Jupyter Notebook

```shell
jupyter notebook iris_classification_main.ipynb
```

Notebook version for interactive exploration.

## Implemented Models

1.  Logistic Regression
2.  K-Nearest Neighbors (KNN)
3.  Support Vector Machine (SVM)
4.  Naive Bayes
5.  Decision Tree
6.  Random Forest
7.  Gradient Boosting

## Evaluation Metrics

*   Accuracy
*   Precision
*   Recall
*   F1-Score
*   Cross-validation

## Improvements over the original project

*   Addition of new algorithms (Random Forest, Gradient Boosting)
*   Interactive Streamlit interface
*   Automatic hyperparameter optimization
*   Advanced visualizations with Plotly
*   Extended evaluation metrics
*   Modular and object-oriented code
*   Complete documentation

## Author

Hamza Khaled - Data Science Internship Project

## License

This project is developed as part of an academic internship.


