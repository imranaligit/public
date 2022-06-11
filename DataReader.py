import pandas as pd


# Reads data from files and converts to DataFrame
class DataReader:
    def __init__(self, trainFile, testFile,
                 category="Prediction", fill=0,
                 delColumns=["Email No."], emailTextColumn="Email Text"):
        self.testData = None
        self.testRawData = pd.read_csv(testFile)
        self.trainData = pd.read_csv(trainFile)
        # Deleting columns which are not needed
        self.__deleteColumns(delColumns)
        # Replace NaN and Empty to default values
        self.trainData.fillna(fill, inplace=True)
        self.trainData.replace("", fill, inplace=True)
        # Initializing map to hold word and its count
        self.wordsCount = {k: 0 for k in self.trainData.columns.values}
        # Removing prediction column from the map of counts
        del self.wordsCount[category]
        # sets testData by removing numbers and punctuations and counting words
        self.__loadTestData(emailTextColumn)

    def __deleteColumns(self, delColumns):
        for column in delColumns:
            if column in self.trainData.columns:
                self.trainData.drop(columns=[column], inplace=True)

    def __loadTestData(self, emailTextColumn):
        emailRows = []
        header = [x for x in self.wordsCount.keys()]
        for rIndex, row in self.testRawData.iterrows():
            testEmail = row[emailTextColumn]
            # Removing new line and carriage return and splitting by space
            testEmailData = testEmail.replace("\n", " ").replace("\r", "").lower().split(" ")

            # Removing numbers and punctuations then counting words
            for w in testEmailData:
                word = ''.join(x for x in w if x.isalnum())
                if word in self.wordsCount:
                    self.wordsCount[word] += 1
            emailRows.append([x for x in self.wordsCount.values()])
            self.wordsCount = {x: 0 for x in self.wordsCount}
        self.testData = pd.DataFrame(emailRows, columns=header)

    def getTrainInput(self):
        return self.trainData.iloc[:, :-1]

    def getTrainOutput(self):
        return self.trainData.iloc[:, -1]

    def getTestInput(self):
        return self.testData

    def getTestInputRaw(self):
        return self.testRawData
