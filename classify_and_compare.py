import sys
import LogisticRegression as lR
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def run_classification(file_name, target_column, positive_label, iterations, output_file_name):
    """
    Method to run classification using logistic regression with both local and sklearn implementation
    :param file_name: csv/txt dataset filename
    :param target_column: column name of labels
    :param positive_label: positive label value
    :param iterations: epoch limit
    :param output_file_name: output file name to save predicted and actual values for local implementation
    :return: tuple of local accuracy and sklearn accuracy
    """
    # Reading dataset file into pandas data frame
    samples = lR.read_dataset(file_name)

    # Dividing samples into train and test data
    train_data, test_data = lR.get_train_test_data(samples)

    # Dividing data set into features and target class
    train_features, train_targets = lR.get_features_targets(train_data, target_column, positive_label)
    test_features, test_targets = lR.get_features_targets(test_data, target_column, positive_label)

    # Normalizing features (train and test) for feature scaling
    mean_arr = []
    std_arr = []
    normalized_train_features, mean_arr, std_arr = lR.normalize_features(train_features.T, mean_arr, std_arr)
    normalized_test_features = lR.normalize_features(test_features.T, mean_arr, std_arr, True)[0]

    # Training logistic regression and getting theta values
    theta, costs = lR.train_logistic_regression(lR.add_X0(normalized_train_features), train_targets, iterations)

    # Predicting target class for test data
    local_predictions = lR.predict(lR.add_X0(normalized_test_features), theta)

    # Saving output (predicted and actual values) to csv file
    output = pd.DataFrame(np.column_stack((local_predictions, test_targets)), columns=['Predicted', 'Actual'])
    output.to_csv("./"+output_file_name+".csv")

    # Plotting and saving cost curve for current iteration
    plt.plot(costs)
    plt.title("Gradient Descent - Cost Curve")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.savefig(output_file_name + "_cost_curve.png")
    plt.close()

    # Running Logistic Regression from sklearn on same train and test datasets
    # Setting iterations and alpha same as local implementation
    sklearn_classifier = LogisticRegression(max_iter=iterations, C=0.1)
    sklearn_classifier.fit(normalized_train_features, train_targets)
    sklearn_predictions = sklearn_classifier.predict(normalized_test_features)

    return lR.get_accuracy(test_targets, local_predictions), lR.get_accuracy(test_targets, sklearn_predictions)


def plot_accuracy_comparison(localLR, sklearnLR):
    """
    Method to plot a comparison curve for accuracies of each 10 runs of algorithm
    between sklearn implementation and local implementation of Logistic Regression Algorithm
    :param localLR: List of accuracies provided by local implementation
    :param sklearnLR: List of accuracies provided by sklearn implementation
    """
    divisions = range(1, 11)
    plt.plot(divisions, localLR, label='Local Logistic Regression', marker='o')
    plt.plot(divisions, sklearnLR, label='sklearn Logistic Regression', marker='o')
    plt.title("Accuracy Comparison of Implementations")
    plt.xlabel("Random Divisions")
    plt.ylabel("Accuracy")
    plt.xticks(divisions)
    plt.legend()
    plt.savefig('accuracy_comparison.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # sys.argv[1]: csv/txt dataset file name
    # sys.argv[2]: target values column name
    # sys.argv[3]: positive label tag name
    # sys.argv[4]: iteration limit on gradient descent
    if len(sys.argv) == 5:
        file_name = sys.argv[1]
        target_column = sys.argv[2]
        positive_label = sys.argv[3]
        iteration = int(sys.argv[4])
        avg_accuracy_local = 0
        avg_accuracy_sklearn = 0
        localLR_accuracies = []
        sklearnLR_accuracies = []
        division_names = []
        for i in range(10):
            division_names.append("Random Division " + str(i + 1))
            local_accuracy, sklearn_accuracy = run_classification(file_name, target_column, positive_label, iteration,
                                                                  division_names[i])
            avg_accuracy_local += local_accuracy
            localLR_accuracies.append(local_accuracy)
            avg_accuracy_sklearn += sklearn_accuracy
            sklearnLR_accuracies.append(sklearn_accuracy)
        # Saving all accuracies to csv file
        output = pd.DataFrame(np.column_stack((localLR_accuracies, sklearnLR_accuracies)), columns=['Local Implementation', 'Sklearn Implementation'])
        output.insert(0, "Division", division_names)
        print(output)
        output.to_csv("./accuracy_comparison" + ".csv")
        # Calculating and displaying average accuracies
        avg_accuracy_local /= 10
        print("\nAverage Local Accuracy:", avg_accuracy_local)
        avg_accuracy_sklearn /= 10
        print("Average Sklearn Accuracy:", avg_accuracy_sklearn)
        # Plotting accuracy comparison curve
        plot_accuracy_comparison(localLR_accuracies, sklearnLR_accuracies)
    else:
        print("Please enter following arguments in order while running the program:")
        print("1. Argument 1: csv/txt dataset file name")
        print("2. Argument 2: Target column name")
        print("3. Argument 3: Positive label tag name")
        print("4. Argument 4: Number of Iterations")
