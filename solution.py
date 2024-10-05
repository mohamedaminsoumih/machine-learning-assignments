import numpy as np
iris = np.genfromtext("iris.txt")
import random

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        return np.mean(iris[:, :-1], axis=0) 

    def empirical_covariance(self, iris):
        return np.cov(iris[:, :-1], rowvar=False)

    def feature_means_class_1(self, iris):
        class_1_data = iris[iris[:, -1] == 1]  # Filtrer les points dont le label est 1
        return np.mean(class_1_data[:, :-1], axis=0)  # Moyenne des attributs pour la classe 1

    def empirical_covariance_class_1(self, iris):
        class_1_data = iris[iris[:, -1] == 1]  # Filtrer les points dont le label est 1
        return np.cov(class_1_data[:, :-1], rowvar=False)  # Matrice de covariance pour la classe 1


class HardParzen:
    def __init__(self, h):
        self.h = h

    def fit(self, train_inputs, train_labels):
        self.X_train = train_inputs
        self.Y_train = train_labels

    def predict(self, test_data):
        predictions = []
        for x in test_data:
            distances = np.sum(np.abs(self.X_train - x), axis=1)  # Distance L1
            neighbors_in_window = self.Y_train[distances <= self.h]

            if len(neighbors_in_window) == 0:
                label_list = np.unique(self.Y_train)
                predictions.append(self.draw_rand_label(x, label_list))
            else:
                predicted_label = np.bincount(neighbors_in_window).argmax()
                predictions.append(predicted_label)
        
        return np.array(predictions)


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def fit(self, train_inputs, train_labels):
        self.X_train = train_inputs
        self.Y_train = train_labels

    def predict(self, test_data):
        predictions = []
        for x in test_data:
            weights = np.exp(-np.sum(np.abs(self.X_train - x), axis=1) / (2 * self.sigma ** 2))  # Poids RBF
            weighted_labels = np.bincount(self.Y_train, weights=weights)
            predicted_label = np.argmax(weighted_labels)
            predictions.append(predicted_label)
        
        return np.array(predictions)


def split_dataset(iris):
    train_indices = np.where(np.arange(len(iris)) % 5 < 3)[0]  # Indices pour l'entraînement
    val_indices = np.where(np.arange(len(iris)) % 5 == 3)[0]  # Indices pour la validation
    test_indices = np.where(np.arange(len(iris)) % 5 == 4)[0]  # Indices pour le test

    train_set = iris[train_indices]
    val_set = iris[val_indices]
    test_set = iris[test_indices]

    return train_set, val_set, test_set


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        model = HardParzen(h)
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_val)
        return np.mean(predictions != self.y_val)  # Taux d'erreur

    def soft_parzen(self, sigma):
        model = SoftRBFParzen(sigma)
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_val)
        return np.mean(predictions != self.y_val)  # Taux d'erreur


def get_test_errors(iris):

    train_set, val_set, test_set = split_dataset(iris)
    x_train, y_train = train_set[:, :-1], train_set[:, -1]
    x_val, y_val = val_set[:, :-1], val_set[:, -1]
    x_test, y_test = test_set[:, :-1], test_set[:, -1]

    h_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    sigma_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    best_h_error = float('inf')
    best_sigma_error = float('inf')
    best_h = None
    best_sigma = None

    for h in h_values:
        error = ErrorRate(x_train, y_train, x_val, y_val).hard_parzen(h)
        if error < best_h_error:
            best_h_error = error
            best_h = h

    for sigma in sigma_values:
        error = ErrorRate(x_train, y_train, x_val, y_val).soft_parzen(sigma)
        if error < best_sigma_error:
            best_sigma_error = error
            best_sigma = sigma

    # Calculer le taux d'erreur sur l'ensemble de test
    hard_parzen_model = HardParzen(best_h)
    hard_parzen_model.fit(x_train, y_train)
    hard_parzen_test_error = np.mean(hard_parzen_model.predict(x_test) != y_test)

    soft_rbf_parzen_model = SoftRBFParzen(best_sigma)
    soft_rbf_parzen_model.fit(x_train, y_train)
    soft_rbf_parzen_test_error = np.mean(soft_rbf_parzen_model.predict(x_test) != y_test)

    return np.array([hard_parzen_test_error, soft_rbf_parzen_test_error])



def random_projections(X, A):
        return (1 / np.sqrt(2)) * np.dot(X, A)  # Projection aléatoire
