import pandas as pd

# Load Banknote Dataset
banknote_df = pd.read_csv('datasets/banknote_authentication.csv')
X_banknote = banknote_df.iloc[:, :-1].values  # All rows, all columns except the last
y_banknote = banknote_df.iloc[:, -1].values   # All rows, last column

# Load Iris Dataset
iris_df = pd.read_csv('datasets/iris.csv')
X_iris = iris_df.iloc[:, :-1].values  # All rows, all columns except the last


# Manually Encode the 'species' column
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_iris = iris_df.iloc[:, -1].map(species_mapping).values
