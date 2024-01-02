from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import time
from classifier import NaiveBayesClassifier, accuracy_score, confusion_matrix, crossval_predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*", "methods": ["POST"], "allow_headers": ["Content-Type", "Authorization"]}})

def load_dataset(name):
    if name == 'iris':
        file_path = 'datasets/iris.csv'
        df = pd.read_csv(file_path)
        species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        df['species'] = df['species'].map(species_mapping)
    elif name == 'banknote':
        file_path = 'datasets/banknote_authentication.csv'
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Dataset not found")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return file_path, df, X, y

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dataset_name = data['dataset']
    prediction_type = data['prediction_type']

    file_path, df, X, y = load_dataset(dataset_name)
    classifier = NaiveBayesClassifier()

    start_time = time.time()
    if prediction_type == 'standard':
        classifier.fit(X, y)
        train_time = time.time() - start_time
        start_time = time.time()
        predictions = classifier.predict(X)
        eval_time = time.time() - start_time
    elif prediction_type == 'crossval':
        start_time = time.time()
        predictions = crossval_predict(X, y, folds=5)
        train_time = eval_time = time.time() - start_time
    else:
        return jsonify({'error': 'Invalid prediction type'}), 400

    accuracy = accuracy_score(y, predictions)
    conf_matrix = confusion_matrix(y, predictions)

    return jsonify({
        'file': file_path,
        'number_of_examples': len(df),
        'number_of_attributes': len(df.columns) - 1,
        'number_of_classes': len(np.unique(y)),
        'training_time': train_time,
        'evaluation_time': eval_time,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
