import pickle
import sys

from sklearn import linear_model, svm, gaussian_process, tree, ensemble, neighbors, neural_network
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split

from sklearn import linear_model

FOLDS = 10
MLP_C = "mlp"
E_TREES, SVC = "extraT", "svc"
RNN, KNN = "rNN", "kNN"
E_DTREE, DTREE = "eTree", "dTree",
L_RIDGE_CV, L_LOGIST_RCV, GAU_PROCESS = "ridgeCV", "logRegCV", "gProcess"
L_SVC, L_LOGIST_R, L_P_AGG_C, L_SGD, L_RIDGE = "lSVC", "logReg", "pAggC", "sgd", "ridge"
E_HIST_GB, E_VOTE, E_STACK, E_RANDF, E_GB, E_BAG, E_ADAB = "histGB", "vote", "stack", "randF", "GB", "bag", "adaB"

MEAN_PREFIX, STD_PREFIX, TEST_PREFIX = 'mean_test_', 'std_test_', 'test_'
MODEL, HYPER_PARAM, CV_RESULT, BEST_INDEX, PARAMS = "model", "hyperParam", "cvResult", "bestIndex", "params"

# Metrics
MET_F1, MET_ACCURACY = "F1", "accuracy"


class ModelAssessor:
    # @primaryMetric to choose best model
    # @metrics list of metrics to generate
    def __init__(self, primaryMetric, metrics, trainX, trainY):
        self.primaryMetric = primaryMetric
        self.metrics = metrics
        self.trainX = trainX
        self.trainY = trainY
        self.bestModelName = None
        self.accuracies = []
        self.modelNames = []
        self.accuraciesVariance = []
        self.models = {
            L_LOGIST_R: {CV_RESULT: None, BEST_INDEX: 0, MODEL: linear_model.LogisticRegression(), HYPER_PARAM: {
                "random_state": [8], "solver": ['lbfgs'], "max_iter": [2000]
            }},
            KNN: {CV_RESULT: None, BEST_INDEX: 0, MODEL: neighbors.KNeighborsClassifier(), HYPER_PARAM: {
                'n_neighbors': [5], 'metric': ['euclidean']}},
            # E_RANDF: {CV_RESULT: None, BEST_INDEX: 0, MODEL: ensemble.RandomForestClassifier(), HYPER_PARAM: {
            #    'n_estimators': [130, 140, 150, 200],
            #    'bootstrap': [True, False]
            # }},
            # E_GB: {CV_RESULT: None, MODEL: ensemble.GradientBoostingClassifier(), HYPER_PARAM: {
            #
            # }},
        }

    # Perform GridSearch cross-validation for all the models
    def runModels(self):
        for k, v in self.models.items():
            sys.stdout.write(f"\rExecuting Model = {k}")
            sys.stdout.flush()
            search = GridSearchCV(v[MODEL], param_grid=v[HYPER_PARAM],
                                  cv=StratifiedKFold(n_splits=FOLDS),
                                  scoring=self.metrics, refit=self.primaryMetric)
            search.fit(self.trainX, self.trainY)
            v[CV_RESULT] = search.cv_results_
            v[BEST_INDEX] = search.best_index_
        print("\n--------------------------")

    # Choose the best model
    def getBestModelMetric(self):
        highestAccuracy = 0
        self.accuracies = []
        self.accuraciesVariance = []
        self.bestModelName = None
        for k, v in self.models.items():
            self.modelNames.append(k)
            print(f"Model = {k}, Best params = {v[CV_RESULT][PARAMS][v[BEST_INDEX]]}")
            accuracy = v[CV_RESULT][MEAN_PREFIX + self.primaryMetric][v[BEST_INDEX]]
            accuracyVariance = v[CV_RESULT][STD_PREFIX + self.primaryMetric][v[BEST_INDEX]]
            self.accuracies.append(accuracy)
            self.accuraciesVariance.append(accuracyVariance)

            if highestAccuracy < accuracy:
                highestAccuracy = accuracy
                self.bestModelName = k

        for i, m in enumerate(self.models.keys()):
            print(f"Mean {self.primaryMetric} CV = {self.accuracies[i]} for {m}")
        print("--------------------------")
        return self.modelNames, self.primaryMetric, self.accuracies, self.accuraciesVariance

    def writeFinalModel(self, fileName):
        modelParams = self.models[self.bestModelName][CV_RESULT][PARAMS][self.models[self.bestModelName][BEST_INDEX]]
        print("Best model = ", modelParams)
        bestModel = self.models[self.bestModelName][MODEL]
        bestModel.set_params(**modelParams)
        bestModel.fit(self.trainX, self.trainY)
        pickle.dump(bestModel, open(fileName, 'wb'))
        print(f"Saved Model to {fileName}")

    def predict(self, testData, loadModel=None):
        if loadModel is None:
            model = self.models[self.bestModelName][MODEL]
        else:
            model = pickle.load(open(loadModel, 'rb'))
        return model.predict(testData)
