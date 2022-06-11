import csv

from pandas import DataFrame


class DataWriter:
    def __init__(self, predictions, data: DataFrame):
        self.data = data
        self.predictions = predictions

    def setNewPredictions(self, predictions):
        self.predictions = predictions

    # writes predictions and data to a given file
    def write(self, fileName):
        with open(fileName, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Email', 'Spam'])
            for rIndex, row in self.data.iterrows():
                testEmail = row["Email Text"]
                print(f"{'Spam' if self.predictions[rIndex] == 1 else 'Not Spam'} =>  {testEmail}")
                writer.writerow([testEmail, self.predictions[rIndex]])
                print()
            print(f"Predictions written to {fileName}")
