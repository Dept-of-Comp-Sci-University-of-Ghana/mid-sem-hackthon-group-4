import joblib
import numpy as np
import pandas as pd

# Path to model

path = "academic_success_classifier.pkl"

model = joblib.load(path)

np.array(['Graduate', 'Dropout', 'Enrolled'], dtype=object)
'''
Encoded values

'Graduate': 2
'Dropout': 0
'Enrolled': 1

'''

example_record = [1,17,1,9238,1,1,125.0,1,19,19,9,9,119.8,1,0,0,1,0,0,18,0,0,6,8,4,11.6,0,0,6,9,0,0.0,0,11.1,0.6,2.02]
example_target = 0

model

"""# Testing model"""

import unittest

class TestModel(unittest.TestCase):
    # Check if the model returns the correct prediction for the example record
    def test_model(self):
        prediction = model.predict([example_record])
        self.assertEqual(prediction[0], example_target)

if __name__ == "__main__":
  unittest.main(argv=['first-arg-is-ignored'], exit=False)