import os
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

import src.models.visualize as vs


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    results = {}

    # fit the learner to the training data
    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()

    results['train_time'] = end - start

    # make predictions on the test set, and make predictions on the first 2000 training samples
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:2000])
    end = time()

    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(y_train[:2000], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = accuracy_score(y_train[:2000], predictions_train)
    results['f_test'] = f1_score(y_test, predictions_test)

    print('{} trained on {} samples.'.format(learner.__class__.__name__, sample_size))

    return results


dataset_path = os.path.join('..', '..', 'data', 'raw', 'phishing_websites.csv')
phishing_dataset = pd.read_csv(dataset_path)
phishing_df = pd.DataFrame(phishing_dataset)
num_phishy = len(phishing_df[phishing_df['Result'] == -1])
num_legitimate = len(phishing_df[phishing_df['Result'] == 1])
phishing_classifications = phishing_df['Result']
phishing_features = phishing_df.drop('Result', axis=1)

X_train, X_test, y_train, y_test = train_test_split(phishing_features,
                                                    phishing_classifications,
                                                    test_size=0.2,
                                                    random_state=0)

print('Training set has {} samples.'.format(X_train.shape[0]))
print('Testing set has {} samples.'.format(X_test.shape[0]))

baseline_accuracy = num_phishy / len(phishing_df) * 100
false_positives = num_legitimate
true_positives = num_phishy
false_negatives = 0
baseline_precision = true_positives / (true_positives + false_positives)
baseline_recall = true_positives / (true_positives + false_negatives)
baseline_f1_score = 2 * baseline_precision * baseline_recall / (baseline_precision + baseline_recall)

print("Naive predictor: [Accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, \
f1_score: {:.4f}]".format(baseline_accuracy, baseline_precision, baseline_recall, baseline_f1_score))

random_forest_clf = RandomForestClassifier(random_state=42)
gradient_boosting_clf = GradientBoostingClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)

one_percent_sample = int(len(X_train) * 0.01)
ten_percent_sample = int(len(X_train) * 0.1)
hundred_percent_sample = int(len(X_train))

results = {}
for classifier in [random_forest_clf, gradient_boosting_clf, xgb_clf]:
    classifier_name = classifier.__class__.__name__
    results[classifier_name] = {}
    for i, samples in enumerate([one_percent_sample, ten_percent_sample, hundred_percent_sample]):
        results[classifier_name][i] = train_predict(classifier, samples, X_train, y_train, X_test, y_test)

vs.evaluate(results, baseline_accuracy, baseline_f1_score)
