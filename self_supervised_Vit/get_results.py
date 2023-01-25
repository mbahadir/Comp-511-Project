import numpy as np
import pandas as pd
import csv
from sklearn.metrics import classification_report
import json
import sys, getopt



def main(argv):
    DATA_PATH = argv
    print(DATA_PATH)
    results = pd.read_csv(DATA_PATH)


    predictions = results["Pred"].values
    labels = results["Label"].values

    ## Results
    confusion_matrix = pd.crosstab(np.array(labels), np.array(predictions), rownames=["Actuals"], colnames=["Predictions"])
    print(confusion_matrix, "\n")

    information = lambda current_row: current_row.split()[1:4] if(len(current_row.split()) == 5) else current_row.split()[2:5] if(len(current_row.split()) == 6) else [current_row.split()[1]]
    # integer_value = lambda list_of_strings: list(map(np.float_, list_of_strings))

    def representation(value):
        if len(value) == 0:
            return "*" * 10
        return information(value)

    class_results = classification_report(labels, predictions, digits=3)
    print(class_results)

    class_results = class_results.strip()
    class_results = class_results.splitlines()
    class_results = class_results[2:]
    class_results = list(map(representation, class_results))

    string_results_init = [ word for sentence in class_results for word in sentence]
    string_results = ','.join(map(str,string_results_init))

    for current_results in class_results:
        print(','.join(current_results))

    print("")
    print("*" * 100)
    
    


if __name__ == "__main__":
    main(sys.argv[1])