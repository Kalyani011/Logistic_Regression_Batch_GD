import pandas as pd
import numpy as np


def read_dataset(data_file):
    """
    Method to read dataset from csv/txt file
    :param data_file: file name of dataset file
    :return: data_samples dataframe with all columns from dataset file
    """
    # data_samples = pd.read_csv(data_file, sep='\t')
    data_samples = pd.read_csv(data_file)
    for columns in data_samples:
        if data_samples[columns].dtypes not in ['int64', 'float64']:
            # Removing extra whitespaces
            data_samples[columns] = data_samples[columns].str.strip()
    return data_samples


def get_train_test_data(data_samples):
    """
    Method to split entire data sample into train and test data
    :param data_samples: dataframe with all columns from dataset
    :return: tuple of separated train and test data
    """
    # Shuffling the entire dataframe
    shuffled_samples = data_samples.sample(frac=1).reset_index(drop=True)
    # Calculating 2/3rd count
    train_count = int((2 * len(shuffled_samples)) / 3)
    # Taking 2/3rd values into train set
    train = shuffled_samples.iloc[:train_count, :]
    # Taking 1/3rd values into test set
    test = shuffled_samples.iloc[train_count:, :]
    return train, test


def get_features_targets(data_samples, target_name, positive_label):
    print(data_samples)
    """
    Method to split data_samples into features array and target array
    :param data_samples: dataframe with all columns from dataset
    :param target_name: column name of target labels
    :param positive_label: Positive label tag name
    :return: tuple of separated features and targets
    """
    # Getting all features from dataset
    feature_names = [name for name in data_samples.columns if name != target_name]
    # Converting positive and negative labels to 0s and 1s
    if type(positive_label) == 'str':
        targets = [1 if target == positive_label else 0 for target in data_samples[target_name]]
    else:
        targets = data_samples[target_name]
    return np.array(data_samples[feature_names]), np.array(targets)


def normalize_features(features, mean_arr, std_arr, test=False):
    """
    Method to perform z-normalisation on training and testing data
    :param features: numpy array with all columns representing features for dataset
    :param mean_arr: array of mean values for each column (initially empty if running for training set)
    :param std_arr: array of standard deviation values for each columns (initially empty if running for training set)
    :param test: boolean value to determine whether to use mean and standard deviation calculated for training set
    :return: normalized features nd-array, mean calculated for training columns, standard deviation calculated for training columns
    """
    for i in range(len(features)):
        if not test:
            mean_arr.append(np.mean(features[i]))
            std_arr.append(np.mean(features[i]))
        for j in range(len(features[i])):
            features[i, j] = (features[i, j] - mean_arr[i]) / std_arr[i]
    return features.T, mean_arr, std_arr


def add_X0(X):
    """
    Method to add x0 term to each sample to be multiplied with bias theta0
    :param X: Set of all features for all samples
    :return: Set of all features for all samples with additional x0 term
    """
    # Creating array of ones to correspond to theta0
    X0 = np.ones(len(X)).reshape(len(X), -1)
    # Adding first column as X0
    return np.concatenate((X0, X), axis=1)


def get_hypothesis(X, theta):
    """
    Method to calculate hypothesis for Logistic Regression 1/1+e^-(theta.T * X)
    :param X: feature values of all feature columns for all samples of length len(X) (Vector X)
    :param theta: theta parameter values for all columns (Vector theta)
    :return: logistic regression hypothesis values for given X and theta for all samples
    """
    theta_transpose_X = np.dot(theta, X.T)
    return 1 / (1 + (np.exp(-1 * theta_transpose_X)))


def get_cost(X, y, theta, n):
    """
    Method to calculate cost J(theta) for all samples
    :param X: feature values of all feature columns for all samples of length n (Vector X)
    :param y: target values for all samples of length n (Vector y)
    :param theta: theta parameter values for all columns (Vector theta)
    :param n: total number of samples
    :return: cost J and hypothesis values for all samples of length n
    """
    H = get_hypothesis(X, theta)
    J = (-1/n) * np.sum(np.add(np.multiply(y, np.log(H)), np.multiply((1 - y), np.log(1 - H))))
    return J, H


def train_logistic_regression(X, y, iterations):
    """
    Method to train logistic regression algorithm
    :param X: feature values of all feature columns for all samples of length len(X) (Vector X)
    :param y: target values for all samples of length len(X) (Vector y)
    :param iterations: Limit on number of epochs
    :return: array of optimized theta values for all features
    """
    n = len(X)
    # Initializing all theta values to zero
    theta = np.zeros(X.shape[1])
    # Initializing list of costs
    J_theta = []
    # Initializing cost and hypothesis
    cost, H = get_cost(X, y, theta, n)
    # Setting learning rate alpha to fixed value 0.1
    alpha = 0.1
    # Initializing epoch (iterations)
    epoch = 0
    # Stopping condition for gradient descent (Limited number of iterations)
    while epoch <= iterations:
        # Calculating derivative of cost
        delta = np.dot((H - y), X)
        # Calculating theta values
        theta -= ((alpha/n) * delta)
        # Updating cost and hypothesis with new theta values
        cost, H = get_cost(X, y, theta, n)
        J_theta.append(cost)
        epoch += 1
    return theta, J_theta


def predict(X, theta):
    """
    Method to perform prediction on test dataset
    :param X: dataset values to be classified
    :param theta: array of optimized theta values gained from training
    :return: array of predicted classes for each sample
    """
    # Calculating hypothesis value for each sample
    h_theta_x = get_hypothesis(X, theta)
    # Setting threshold to 0.5
    threshold = 0.5
    predictions = []
    # Predicting class for each sample
    for probability in h_theta_x:
        if probability >= threshold:
            predictions.append(1)
        elif probability < threshold:
            predictions.append(0)
    return predictions


def get_accuracy(expected, predicted):
    """
    Method to calculate accuracy of the algorithm
    :param expected: array of expected label values
    :param predicted: array of predicted label values
    :return: accuracy ranging between 0 and 1
    """
    correct = 0
    for i in range(len(expected)):
        if expected[i] == predicted[i]:
            correct += 1
    return correct / len(expected)

