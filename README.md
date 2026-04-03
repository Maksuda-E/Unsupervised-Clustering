# UCLA Neural Networks Project

This project modularizes a Jupyter Notebook machine learning workflow into a Python project and deploys the model with Streamlit.

## Project Objective

Predict whether a student has a high or low chance of admission into UCLA using a neural network classifier.

## Notebook Logic Used

The project follows the notebook workflow:

- Convert `Admit_Chance` into binary target using threshold `0.80`
- Drop `Serial_No`
- Treat `University_Rating` and `Research` as categorical
- Apply one-hot encoding
- Split data using:
  - `test_size=0.20`
  - `random_state=123`
  - `stratify=y`
- Scale features using `MinMaxScaler`
- Train `MLPClassifier` with:
  - `activation='tanh'`
  - `batch_size=50`
  - `hidden_layer_sizes=3`
  - `max_iter=200`
  - `random_state=123`

## Project Structure

```text
UCLA_Neural-Networks/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Admission_Predict_Ver1.1.csv
│
├── artifacts/
├── logs/
│
└── src/
    ├── config.py
    ├── logger.py
    ├── custom_exception.py
    ├── data_loader.py
    ├── preprocess.py
    ├── train.py
    ├── evaluate.py
    └── predict.py