import time
import pandas as pd
import numpy as np

from sklearn.model_selection import (train_test_split, cross_validate)
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             make_scorer)

class Harness:
    
    def __init__(self, scorer, random_state=2021):
        self.scorer = scorer
        self.history = pd.DataFrame(columns=['Name', 'F1', 'Recall',
                                             'Precision', 'CV_Time(sec)', 'Notes'])

    def report(self, model, X, y, name, notes='', cv=5,):
        start_time = time.time()
        self.scores = cross_validate(model, X, y, 
                                 scoring=self.scorer, cv=cv)
        end_time = time.time()
        cv_time = end_time - start_time
        frame = pd.DataFrame([[name, self.scores['test_f1'].mean(),
                               self.scores['test_recall'].mean(),
                               self.scores['test_precision'].mean(),
                               cv_time, notes]],
                             columns=['Name', 'F1', 'Recall',
                                      'Precision', 'CV_Time(sec)', 'Notes'])
        self.history = self.history.append(frame)
        self.history = self.history.reset_index(drop=True)
        self.history = self.history.sort_values('F1')
        self.print_error(name, self.scores['test_f1'].mean())
        return [self.scores['test_f1'].mean(), self.scores['test_recall'].mean(),
                self.scores['test_precision'].mean()]

    def print_error(self, name, Accuracy):
        print(f"{name} has an average F1 of {self.scores['test_f1'].mean()}")
        print(f"{name} has an average Recall of {self.scores['test_recall'].mean()}")
        print(f"{name} has an average Precision of {self.scores['test_precision'].mean()}")