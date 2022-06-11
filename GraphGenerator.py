import matplotlib.pyplot as plot
from bokeh import plotting as bplot
import numpy as np
import scipy as spy


class GraphGenerator:

    def __init__(self, lineWidth=2, circleSize=10, circleFillColor="white",
                 spamLabel="Spam", notSpamLabel="Not Spam"):
        self.spamLabel = spamLabel
        self.notSpamLabel = notSpamLabel
        self.lineWidth = lineWidth
        self.circleSize = circleSize
        self.circleFillColor = circleFillColor

    # Show distribution of train data and its percentage
    def generateDataDistributionMatplotPieChart(self, trainY):
        spamCount = sum(x == 1 for x in trainY)
        notSpamCount = sum(x == 0 for x in trainY)
        print(f"spam count = {spamCount} and not spam count = {notSpamCount}")
        plot.pie([spamCount, notSpamCount],
                 labels=[self.spamLabel, self.notSpamLabel], autopct='%1.1f%%')
        plot.title("Input data")
        plot.show()

    # Show accuracies and their standard deviation
    def generateModelAccuraciesUsingBokeh(self, modelNames, metric, accuracies, accuraciesVariance):
        bokehFigure = bplot.figure(title="Models' Performance", x_axis_label='Models',
                                   y_axis_label=metric + ' and STD', x_range=modelNames)
        bokehFigure.line(np.array(modelNames), spy.array(accuracies), legend_label=metric,
                         line_width=self.lineWidth, color="blue")
        bokehFigure.circle(modelNames, accuracies, legend_label=metric, color='blue',
                           fill_color=self.circleFillColor, size=self.circleSize)
        bokehFigure.line(modelNames, accuraciesVariance, legend_label=metric + " STD",
                         color="red", line_width=self.lineWidth)
        bokehFigure.circle(modelNames, accuraciesVariance, legend_label=metric + " STD",
                           color='red', fill_color=self.circleFillColor, size=self.circleSize)
        bplot.show(bokehFigure)
