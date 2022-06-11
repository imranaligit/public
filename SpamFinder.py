import os
import sys

import DataReader as dR
import DataWriter as dWr
import GraphGenerator as gG
import ModelAssessor as mA

# name of the prediction and trained model file
outputFile = "predictions.csv"
finalModelFile = "finalModel.sav"

# Reading command line arguments first one is training dataset and second is prediction dataset
trainingFile, emailFile = sys.argv[-2], sys.argv[-1]

# Requirement: Reading CSV files using pandas
dataReader = dR.DataReader(trainingFile, emailFile)

# Requirement: using numpy & scipy to process data
graphGenerator = gG.GraphGenerator()

# Requirement: Assessing and training model using scikit learn
modelAssessor = mA.ModelAssessor(mA.MET_F1, {mA.MET_F1: 'f1', mA.MET_ACCURACY: 'accuracy'},
                                 dataReader.getTrainInput(), dataReader.getTrainOutput())

# predict using dumped model or find best model and predict
if os.path.exists(finalModelFile) and not ("-f" in sys.argv):
    print("Predicting...")
    predictions = modelAssessor.predict(dataReader.getTestInput(), finalModelFile)
else:
    # Requirement: Showing distribution of training data using matplotlib
    graphGenerator.generateDataDistributionMatplotPieChart(dataReader.getTrainOutput())

    modelAssessor.runModels()
    # Requirement: Using bokeh to visualize performance of different models
    graphGenerator.generateModelAccuraciesUsingBokeh(*(modelAssessor.getBestModelMetric()))

    print(f"Writing model to {finalModelFile}")
    modelAssessor.writeFinalModel(finalModelFile)
    print("Predicting...")
    predictions = modelAssessor.predict(dataReader.getTestInput())

print(predictions)

# Requirement: Processed data is written to .csv
dataWriter = dWr.DataWriter(predictions, dataReader.getTestInputRaw())
dataWriter.write(outputFile)
