import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for pr, gt in zip(prediction, ground_truth):
        if pr == gt & gt == True:
            tp += 1
        if pr == gt & gt == False:
            tn += 1
        if pr != gt & gt == True:
            fn += 1
        if pr != gt & gt == False:
            fp += 1

    # import pdb;
    # pdb.set_trace()

    # from sklearn.metrics import precision_recall_curve
    # precision, recall, _ = precision_recall_curve(ground_truth, prediction)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print("@@@@@ precision = inf")

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        print("@@@@@ recall = inf")


    accuracy = (tp + tn) / (tp + tn + fn + fp)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        print("@@@@@ f1 = 0")
        f1 = 0
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = 0
    for pr, gt in zip(prediction, ground_truth):
        if pr == gt:
            accuracy += 1
    return accuracy / len(prediction)


def multiclass_confusion_matrix(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    confusion_matrix
    '''
    # TODO: Implement computing accuracy
    matrix = np.zeros((10, 10), dtype=int)
    for pr, gt in zip(prediction, ground_truth):
        matrix[pr][gt] += 1
    return matrix