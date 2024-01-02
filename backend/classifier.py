import numpy as np

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_stats = {}

        for c in self.classes:
            X_c = X[y == c]
            n_samples_for_class = len(X_c)
            self.class_stats[c] = {
                'log_prior_probability': np.log(n_samples_for_class / len(X)),
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0, ddof=1) 
            }

    def predict(self, X):
        predictions = []
        for x in X:
            log_probabilities = self.calculate_class_probabilities(x)
            best_label = max(log_probabilities, key=log_probabilities.get)
            predictions.append(best_label)
        return np.array(predictions)

    def calculate_class_probabilities(self, input_vector):
        log_probabilities = {}
        for class_value, class_stats in self.class_stats.items():
            log_prob = class_stats['log_prior_probability']
            for i in range(len(class_stats['mean'])):
                log_prob += self.gaussian_log_probability(
                    input_vector[i], class_stats['mean'][i], class_stats['var'][i])
            log_probabilities[class_value] = log_prob
        return log_probabilities

    def gaussian_log_probability(self, x, mean, var):
        # avoid division by zero in case variance is zero
        var = max(var, 1e-6)

        # use log probabilities for numerical stability
        exponent = -((x - mean) ** 2 / (2 * var))
        log_prob = exponent - np.log(np.sqrt(2 * np.pi * var))
        return log_prob

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_indices = {cls: i for i, cls in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        matrix[class_indices[true]][class_indices[pred]] += 1
    return matrix

def crossval_predict(X, y, folds):
    if folds <= 1 or folds > len(X):
        raise ValueError("Invalid number of folds")
    
    classifier = NaiveBayesClassifier()

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // folds
    all_predictions = np.empty_like(y)

    for i in range(folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i != folds - 1 else len(X)
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train = y[train_indices]
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        all_predictions[test_indices] = predictions

    return all_predictions
