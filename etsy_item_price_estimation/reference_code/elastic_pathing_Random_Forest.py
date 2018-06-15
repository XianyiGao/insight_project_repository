import sys
import os
import math
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from decimal import Decimal
from sklearn2pmml import PMMLPipeline
import sklearn, sklearn.externals.joblib, sklearn_pandas
from sklearn.externals import joblib

class dataPreperation:
    maxNumTraces = 0
    maxNumIntersections = 0
    maxNumTurns = 0
    maxNumVolunteers = 0
    speedData = []
    angleData = []
    directionData = []
    numOfPathsData = []
    labelData = []
    originalSpeedData = []
    driverData = []
    traceData = []
    intersectionData = []
    turnData = []
    fileName = ""
    def __init__(self, maxNumTraces, maxNumIntersections, maxNumTurns, maxNumVolunteers, file_name):
        self.maxNumTraces = maxNumTraces
        self.maxNumIntersections = maxNumIntersections
        self.maxNumTurns = maxNumTurns
        self.maxNumVolunteers = maxNumVolunteers
        self.fileName = file_name
    def resampling(self, dataList, m, n):
        x = 0.0
        y = 0.0
        position = 0
        y1 = 0
        y2 = 0
        x0 = 0
        y0 = 0
        output = []
        step = m * 1.0 / n
        for i in range(1,n+1):
            position = step * i
            x = int(math.floor(position))
            y = int(math.ceil(position))
            if x == y:
                output.append(dataList[x-1])
            else:
                if x == 0:
                    output.append(dataList[0])
                elif x == n:
                    output.append(dataList[x-1])
                else:
                    y1 = dataList[x-1]
                    y2 = dataList[y-1]
                    x0 = position - x
                    y0 = x0 * (y2 - y1) + y1
                    output.append(y0)
        return output

    def readData(self):

        volunteer = range(1, self.maxNumVolunteers + 1)
        trace = range(1, self.maxNumTraces + 1)
        intersection = range(1, self.maxNumIntersections)
        turn = range(self.maxNumTurns)
        count = 0
        for i in volunteer:
            for j in trace:
                for k in intersection:
                    for z in turn:
                        txtFileName = self.fileName + "P" + str(i) + "_track-" + str(j) + "_" + str(k) + "_output" + str(
                            z) + ".txt"
                        txtFile = Path(txtFileName)
                        if txtFile.exists():
                            with open(txtFileName, 'rt') as singleFile:
                                n = int(singleFile.readline())
                                time = [int(x) for x in singleFile.readline().split()]
                                speed = [float(x) for x in singleFile.readline().split()]
                                angle = float(singleFile.readline())
                                direction = int(
                                    singleFile.readline())  # 0 is straight, -1 is right turn, 1 is left turn
                                numOfPaths = int(singleFile.readline())
                                label = int(singleFile.readline())
                                if len(speed) > 0:
                                    speedProcessed = self.resampling(speed, len(speed), 10)
                            count += 1
                            self.speedData.append(speedProcessed)
                            self.angleData.append(angle)
                            self.directionData.append(direction)
                            self.labelData.append(label)
                            self.numOfPathsData.append(numOfPaths)
                            self.originalSpeedData.append(speed)
                            self.driverData.append(i)
                            self.traceData.append(j)
                            self.intersectionData.append(k)
                            self.turnData.append(z)

def determine_categorical_columns(df):
    categorical_columns = []
    x = 0
    for col in df.dtypes:
        if col == 'object':
            val = df[df.columns[x]].iloc[0]
            if not isinstance(val,Decimal):
                categorical_columns.append(df.columns[x])
        x += 1
    return categorical_columns



class loadData:
    checkFile = []
    newData = []
    speedData = []
    angleData = []
    directionData = []
    labelData = []
    numOfPathsData = []
    originalSpeedData = []
    driverData = []
    traceData = []
    intersectionData = []
    turnData = []

    def __init__(self, check_file, source_file):
        self.checkFile = Path(check_file+"/turnData.txt")
        if not(self.checkFile.exists()):
            self.newData = dataPreperation(102, 150, 6, 6, source_file)
            self.newData.readData()
            self.speedData = self.newData.speedData
            self.angleData = self.newData.angleData
            self.directionData = self.newData.directionData
            self.labelData = self.newData.labelData
            self.numOfPathsData = self.newData.numOfPathsData
            self.originalSpeedData = self.originalSpeedData
            self.driverData = self.newData.driverData
            self.traceData = self.newData.traceData
            self.intersectionData = self.newData.intersectionData
            self.turnData = self.newData.turnData
            with open(check_file+"/speedData.txt", "wb") as fp:
                pickle.dump(self.speedData, fp)
            with open(check_file+"/angleData.txt", "wb") as fp:
                pickle.dump(self.angleData, fp)
            with open(check_file+"/directionData.txt", "wb") as fp:
                pickle.dump(self.directionData, fp)
            with open(check_file+"/labelData.txt", "wb") as fp:
                pickle.dump(self.labelData, fp)
            with open(check_file+"/numOfPathsData.txt", "wb") as fp:
                pickle.dump(self.numOfPathsData, fp)
            with open(check_file+"/originalSpeedData.txt", "wb") as fp:
                pickle.dump(self.originalSpeedData, fp)
            with open(check_file+"/driverData.txt", "wb") as fp:
                pickle.dump(self.driverData, fp)
            with open(check_file+"/traceData.txt", "wb") as fp:
                pickle.dump(self.traceData, fp)
            with open(check_file+"/intersectionData.txt", "wb") as fp:
                pickle.dump(self.intersectionData, fp)
            with open(check_file+"/turnData.txt", "wb") as fp:
                pickle.dump(self.turnData, fp)
        else:
            with open(check_file+"/speedData.txt", "rb") as fp:
                self.speedData = pickle.load(fp)
            with open(check_file+"/angleData.txt", "rb") as fp:
                self.angleData = pickle.load(fp)
            with open(check_file+"/directionData.txt", "rb") as fp:
                self.directionData = pickle.load(fp)
            with open(check_file+"/labelData.txt", "rb") as fp:
                self.labelData = pickle.load(fp)
            with open(check_file+"/numOfPathsData.txt", "rb") as fp:
                self.numOfPathsData = pickle.load(fp)
            with open(check_file+"/originalSpeedData.txt", "rb") as fp:
                self.originalSpeedData = pickle.load(fp)
            with open(check_file+"/driverData.txt", "rb") as fp:
                self.driverData = pickle.load(fp)
            with open(check_file+"/traceData.txt", "rb") as fp:
                self.traceData = pickle.load(fp)
            with open(check_file+"/intersectionData.txt", "rb") as fp:
                self.intersectionData = pickle.load(fp)
            with open(check_file+"/turnData.txt", "rb") as fp:
                self.turnData = pickle.load(fp)


NJ_data = loadData("processed_data", "EP_data/")
# output to a file for checking
f = open("data_checking.txt", "w")
f.write('index, angle, direction, numOfPaths, speed[1], speed[2], speed[3], speed[4], speed[5], speed[6], speed[7], speed[8], speed[9], speed[10], label, driver, trace, intersection, turn\n')
for i in range(0,len(NJ_data.turnData)):
    f.write(str(i) + ', ' + str(NJ_data.angleData[i]) + ', ' + str(NJ_data.directionData[i]) + ', ' + str(NJ_data.numOfPathsData[i]) + ', ' + str(NJ_data.speedData[i][0]) + ', ' + str(NJ_data.speedData[i][1]) + ', ' + str(NJ_data.speedData[i][2])
            + ', ' + str(NJ_data.speedData[i][3]) + ', ' + str(NJ_data.speedData[i][4]) + ', ' + str(NJ_data.speedData[i][5]) + ', ' + str(NJ_data.speedData[i][6]) + ', ' + str(NJ_data.speedData[i][7])
            + ', ' + str(NJ_data.speedData[i][8]) + ', ' + str(NJ_data.speedData[i][9]) + ', ' + str(NJ_data.labelData[i]) + ', ' + str(NJ_data.driverData[i]) + ', ' + str(NJ_data.traceData[i]) + ', ' + str(NJ_data.intersectionData[i]) + ', ' + str(NJ_data.turnData[i]) + '\n')
# preparation of data
f.close()

index = range(1,len(NJ_data.turnData)+1)
index = [str(e) for e in index]
index = np.asarray(index).T
angleData = np.asarray(NJ_data.angleData).T
speedData = np.asarray(NJ_data.speedData)
directionData = np.asarray(NJ_data.directionData).T
labelData = [int(e) for e in NJ_data.labelData]
labelData = np.asarray(labelData).T
numOfPathsData = np.asarray(NJ_data.numOfPathsData).T
driverData = np.asarray(NJ_data.driverData).T

dataEP1 = np.stack((index, angleData, directionData, numOfPathsData), axis=1)
dataEP2 = np.stack((labelData, driverData), axis=1)
dataEP = np.concatenate((dataEP1, speedData, dataEP2), axis=1)
headline = np.array(['index', 'angle', 'direction', 'numOfPaths', 'speed[1]', 'speed[2]', 'speed[3]', 'speed[4]',
                    'speed[5]', 'speed[6]', 'speed[7]', 'speed[8]', 'speed[9]', 'speed[10]', 'label', 'driver'])

dfEP = pd.DataFrame(dataEP, columns=headline)

# splitting of training and testing data

testDriver = '6'
train, test = dfEP[dfEP['driver'] != testDriver], dfEP[dfEP['driver'] == testDriver]
np.random.seed(1) #1, 321
train = train.iloc[np.random.permutation(len(train))]
train = train.reset_index(drop=True)
print(len(train))
print(len(test))

features = dfEP.columns[1:14]

param_grid = {
    'n_estimators': [300],
    'n_jobs' : [-1],
    #'max_depth': [7],
    'random_state': [0]
    #'min_samples_split' : [2, 4],
    #'min_samples_leaf' : [1, 3]
}
clfIni = RandomForestClassifier()
grid_clf = GridSearchCV(clfIni, param_grid, cv=10)
grid_clf.fit(train[features], train['label'])
print(grid_clf.best_params_)
clf = grid_clf.best_estimator_

print (np.mean(cross_val_score(clf, train[features], train['label'], cv=10)))



#--------svm starting
#clf = svm.SVC(kernel='rbf')
#clf.fit(train[features], train['label'])

#--------svm ending
results = clf.predict(train[features]) # test[features]

#probability = clf.predict_proba(train[features]) # test[features]

values = pd.crosstab(train['label'], results, rownames=['Actual Label'], colnames=['Predicted Label']) # test['label']

accuracy = 1 - (values.iat[0,1] + values.iat[1,0])/(values.iat[0,0] + values.iat[0,1] + values.iat[1,0] + values.iat[1, 1])

print(accuracy)
print(values.iat[0,0], values.iat[0,1])
print(values.iat[1,0], values.iat[1,1])
results2 = clf.predict(test[features])
#probability2 = clf.predict_proba(test[features])
values2 = pd.crosstab(test['label'], results2, rownames=['Actual Label'], colnames=['Predicted Label'])
accuracy2 = 1 - (values2.iat[0,1] + values2.iat[1,0])/(values2.iat[0,0] + values2.iat[0,1] + values2.iat[1,0] + values2.iat[1, 1])
print(accuracy2)

categorical_columns = determine_categorical_columns(dfEP)
categorical_columns = np.array(categorical_columns[1:14])
pipeline = PMMLPipeline([
  ("pretrained-estimator", clf)
])
pipeline.target_field = "label"
pipeline.active_fields = categorical_columns

result3 = pipeline.predict(test[features])
values3 = pd.crosstab(test['label'], result3, rownames=['Actual Label'], colnames=['Predicted Label'])
probability3 = pipeline.predict_proba(test[features])
accuracy3 = 1 - (values3.iat[0,1] + values3.iat[1,0])/(values3.iat[0,0] + values3.iat[0,1] + values3.iat[1,0] + values3.iat[1, 1])
print(accuracy3)
name = "trained_models/RFpipeline"+ str(testDriver) + ".pkl.z"
joblib.dump(pipeline, name, compress = 9)

np.savetxt('test.txt', test.values, fmt='%s', delimiter="\t", header="index\tangle\tdirection\tnumOfPaths\tspeed[1]\tspeed[2]\tspeed[3]\tspeed[4]\tspeed[5]\tspeed[6]\tspeed[7]\tspeed[8]\tspeed[9]\tspeed[10]\tlabel\tdriver")
np.savetxt('predicted.txt', result3, fmt='%s', delimiter="\t", header="value")
np.savetxt('predicted_prob.txt', probability3, fmt='%s', delimiter="\t", header="0\t1")


# This is for the processing of identical route dataset for different individual drivers
Individual_data = loadData("processed_data_individual", "EP_data_individual/")
f = open("data_checking_individual.txt", "w")
f.write('index, angle, direction, numOfPaths, speed[1], speed[2], speed[3], speed[4], speed[5], speed[6], speed[7], speed[8], speed[9], speed[10], label, driver, trace, intersection, turn\n')
for i in range(0,len(Individual_data.turnData)):
    f.write(str(i) + ', ' + str(Individual_data.angleData[i]) + ', ' + str(Individual_data.directionData[i]) + ', ' + str(Individual_data.numOfPathsData[i]) + ', ' + str(Individual_data.speedData[i][0]) + ', ' + str(Individual_data.speedData[i][1]) + ', ' + str(Individual_data.speedData[i][2])
            + ', ' + str(Individual_data.speedData[i][3]) + ', ' + str(Individual_data.speedData[i][4]) + ', ' + str(Individual_data.speedData[i][5]) + ', ' + str(Individual_data.speedData[i][6]) + ', ' + str(Individual_data.speedData[i][7])
            + ', ' + str(Individual_data.speedData[i][8]) + ', ' + str(Individual_data.speedData[i][9]) + ', ' + str(Individual_data.labelData[i]) + ', ' + str(Individual_data.driverData[i]) + ', ' + str(Individual_data.traceData[i]) + ', ' + str(Individual_data.intersectionData[i]) + ', ' + str(Individual_data.turnData[i]) + '\n')
# preparation of data
f.close()
indexD2 = range(1,len(Individual_data.turnData)+1)
indexD2 = [str(e) for e in indexD2]
indexD2 = np.asarray(indexD2).T
angleDataD2 = np.asarray(Individual_data.angleData).T
speedDataD2 = np.asarray(Individual_data.speedData)
directionDataD2 = np.asarray(Individual_data.directionData).T
labelDataD2 = [int(e) for e in Individual_data.labelData]
labelDataD2 = np.asarray(labelDataD2).T
numOfPathsDataD2 = np.asarray(Individual_data.numOfPathsData).T
driverDataD2 = np.asarray(Individual_data.driverData).T

dataD2EP1 = np.stack((indexD2, angleDataD2, directionDataD2, numOfPathsDataD2), axis=1)
dataD2EP2 = np.stack((labelDataD2, driverDataD2), axis=1)
dataD2EP = np.concatenate((dataD2EP1, speedDataD2, dataD2EP2), axis=1)
headlineD2 = np.array(['index', 'angle', 'direction', 'numOfPaths', 'speed[1]', 'speed[2]', 'speed[3]', 'speed[4]',
                    'speed[5]', 'speed[6]', 'speed[7]', 'speed[8]', 'speed[9]', 'speed[10]', 'label', 'driver'])

dfD2EP = pd.DataFrame(dataD2EP, columns=headlineD2)
TrainD2 = dfEP
TestD2 = dfD2EP
print(len(TrainD2))
np.random.seed(1234)
TrainD2 = TrainD2.iloc[np.random.permutation(len(TrainD2))]
TrainD2 = TrainD2.reset_index(drop=True)
print(len(TrainD2))
print(len(TestD2))

param_grid = {
    'n_estimators': [300],
    'n_jobs' : [-1],
    'max_depth': [10],
    'random_state': [0],
    #'min_samples_split' : [4]
    #'min_samples_leaf' : [1, 3]
}
clfIni = RandomForestClassifier()
grid_clf = GridSearchCV(clfIni, param_grid, cv=10)
grid_clf.fit(TrainD2[features], TrainD2['label'])
print(grid_clf.best_params_)
clf = grid_clf.best_estimator_

print (np.mean(cross_val_score(clf, TrainD2[features], TrainD2['label'], cv=10)))
resultsD2 = clf.predict(TrainD2[features]) # test[features]

#probability = clf.predict_proba(train[features]) # test[features]

values = pd.crosstab(TrainD2['label'], resultsD2, rownames=['Actual Label'], colnames=['Predicted Label']) # test['label']

accuracyD2 = 1 - (values.iat[0,1] + values.iat[1,0])/(values.iat[0,0] + values.iat[0,1] + values.iat[1,0] + values.iat[1, 1])

print("training accuracy: "+ str(accuracyD2))
print(values.iat[0,0], values.iat[0,1])
print(values.iat[1,0], values.iat[1,1])

results2D2 = clf.predict(TestD2[features])
probability2D2 = clf.predict_proba(TestD2[features])
values2 = pd.crosstab(TestD2['label'], results2D2, rownames=['Actual Label'], colnames=['Predicted Label'])
accuracy2D2 = 1 - (values2.iat[0,1] + values2.iat[1,0])/(values2.iat[0,0] + values2.iat[0,1] + values2.iat[1,0] + values2.iat[1, 1])
print("testing accuracy: " + str(accuracy2D2))

np.savetxt('test30.txt', TestD2.values, fmt='%s', delimiter="\t", header="index\tangle\tdirection\tnumOfPaths\tspeed[1]\tspeed[2]\tspeed[3]\tspeed[4]\tspeed[5]\tspeed[6]\tspeed[7]\tspeed[8]\tspeed[9]\tspeed[10]\tlabel\tdriver")
np.savetxt('predicted30.txt', results2D2, fmt='%s', delimiter="\t", header="value")
np.savetxt('predicted_prob30.txt', probability2D2, fmt='%s', delimiter="\t", header="0\t1")


categorical_columns = determine_categorical_columns(dfEP)
categorical_columns = np.array(categorical_columns[1:14])
pipeline = PMMLPipeline([
  ("pretrained-estimator", clf)
])
pipeline.target_field = "label"
pipeline.active_fields = categorical_columns

result3 = pipeline.predict(TestD2[features])
values3 = pd.crosstab(TestD2['label'], result3, rownames=['Actual Label'], colnames=['Predicted Label'])
probability3 = pipeline.predict_proba(TestD2[features])
accuracy3 = 1 - (values3.iat[0,1] + values3.iat[1,0])/(values3.iat[0,0] + values3.iat[0,1] + values3.iat[1,0] + values3.iat[1, 1])
print(accuracy3)
name = "trained_models/RFpipelineIndividual.pkl.z"
joblib.dump(pipeline, name, compress = 9)


print("done!")
