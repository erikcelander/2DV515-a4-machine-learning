import numpy as np


class NaiveBayesClassifier:
    def fit(self, X, y):
            self.classes = np.unique(y)
            self.class_stats = {}

            for c in self.classes:
                X_c = X[y == c]
                n_samples_for_class = len(X_c)
                self.class_stats[c] = {
                    # Store the log of the prior probability
                    'log_prior_probability': np.log(n_samples_for_class / len(X)),
                    'mean': np.mean(X_c, axis=0),
                    'var': np.var(X_c, axis=0, ddof=1)  # ddof=1 for sample variance
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
            # Start with the log of the prior probability
            log_prob = class_stats['log_prior_probability']
            for i in range(len(class_stats['mean'])):
                log_prob += self.gaussian_log_probability(
                    input_vector[i], class_stats['mean'][i], class_stats['var'][i])
            log_probabilities[class_value] = log_prob
        return log_probabilities


    def gaussian_log_probability(self, x, mean, var):
        # Avoid division by zero in case variance is zero
        var = max(var, 1e-6)
        # Use log probabilities for numerical stability
        exponent = -((x - mean) ** 2 / (2 * var))
        log_prob = exponent - np.log(np.sqrt(2 * np.pi * var))
        return log_prob
