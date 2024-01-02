import numpy as np
import pandas as pd
import time
from classifier import NaiveBayesClassifier  # Ensure your classifier is in this file

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_dataset(dataset_name, dataset_df, classifier, species_mapping=None):
    # Encoding the species if required (for the Iris dataset)
    if species_mapping:
        dataset_df['species'] = dataset_df['species'].map(species_mapping)

    # Extracting features and labels
    X = dataset_df.iloc[:, :-1].values
    y = dataset_df.iloc[:, -1].values

    # Dataset information
    print(f"Dataset: {dataset_name}")
    print(f"Number of examples: {len(dataset_df)}")
    print(f"Number of attributes: {len(dataset_df.columns) - 1}")
    print(f"Number of classes: {len(np.unique(y))}\n")

    # Training the classifier
    start_time = time.time()
    classifier.fit(X, y)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.3f} sec")

    # Evaluating the classifier
    start_time = time.time()
    predictions = classifier.predict(X)
    evaluation_time = time.time() - start_time
    print(f"Evaluation time: {evaluation_time:.3f} sec")

    # Calculating accuracy
    accuracy = accuracy_score(y, predictions)
    correct_predictions = np.sum(y == predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%  ({correct_predictions}/{len(y)} correctly classified)\n")

# Load datasets
iris_df = pd.read_csv('datasets/iris.csv')
banknote_df = pd.read_csv('datasets/banknote_authentication.csv')

# Species mapping for the Iris dataset
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

# Instantiate the classifier
classifier = NaiveBayesClassifier()

# Evaluate datasets
evaluate_dataset("Iris Dataset", iris_df, classifier, species_mapping)
evaluate_dataset("Banknote Authentication Dataset", banknote_df, classifier)
