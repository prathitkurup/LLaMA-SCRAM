import math
import csv
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix

results_csv = 'original_prompt_testing_results_275stp.csv'

true_pos = true_neg = false_pos = false_neg = 0
odd_labeled = -1
y_true_raw = []
y_pred_raw = []

y_true_clean = []
y_pred_clean = []

with open(results_csv, 'r') as file:
    csv_reader = csv.reader(file)

    for row in csv_reader:
        comment = row[0]
        actual_label = row[1]
        model_labeled_sentiment = row[2]
        converted_actual_sentiment = row[4]
        converted_model_labeled_sentiment = row[5]
        new_comp_label = row[6]

        y_true_raw.append(actual_label)
        y_pred_raw.append(model_labeled_sentiment)

        y_true_clean.append(converted_actual_sentiment)
        y_pred_clean.append(converted_model_labeled_sentiment)

        if converted_model_labeled_sentiment == "Positive" and new_comp_label == '1':
            true_pos += 1
        elif converted_model_labeled_sentiment == "Negative" and new_comp_label == '1':
            true_neg += 1
        elif converted_actual_sentiment == "Negative" and converted_model_labeled_sentiment == "Positive":
            false_pos += 1
        elif converted_actual_sentiment == "Positive" and converted_model_labeled_sentiment == "Negative":
            false_neg += 1
        else:
            odd_labeled += 1

        

print(results_csv.upper(), "\n"
      "Calculations based on converted/collapsed label values:", "\n"
      "True Positives: ", true_pos, "\n"
      "True Negatives: ", true_neg, "\n"
      "False Positives: ", false_pos, "\n"
      "False Negatives: ", false_neg, "\n"
      "Oddly Labeled: ", odd_labeled, "\n")

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
# sklearn_matthews_corrcoef_cleaned = matthews_corrcoef(y_true_clean, y_pred_clean)
# sklearn_matthews_corrcoef_raw = matthews_corrcoef(y_true_raw, y_pred_raw)
manual_matthews_corrcoef_cleaned = (true_pos * true_neg - false_pos * false_neg) / math.sqrt((true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg))

print("Precision: ", precision, "\n"
      "Recall: ", recall, "\n"
      "F1 Score: ", f1_score, "\n"
      "Accuracy: ", accuracy, "\n"
    #   "Matthews CorrCoef (Sklearn) for RAW data: ", sklearn_matthews_corrcoef_raw, "\n"
    #   "Matthews CorrCoef (Sklearn) for CONVERTED data: ", sklearn_matthews_corrcoef_cleaned, "\n"
      "Matthews Correlation Coefficient: ", manual_matthews_corrcoef_cleaned)

confusion_matrix_binary = confusion_matrix(y_true_clean, y_pred_clean, labels=["Positive", "Negative"])
print("Confusion Matrix (using converted labels): \n", confusion_matrix_binary)