from sklearn.datasets import load_files
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, hinge_loss
from ipywidgets import FloatProgress
from IPython.display import display
from random import sample
from json import dumps
from numpy import seterr
import pandas as pd
import time as tm
dataset = load_files("./Training Data/3Day", load_content=False)
max_epoch = 10
patience = 3
test_count = 4
validation_count = 2
test_indices = sample(range(0, len(dataset['filenames'])), test_count)
training_data = {
    'filenames': [x for i,x in enumerate(dataset['filenames'].tolist()) if i not in test_indices],
    'targets': [x for i,x in enumerate(dataset['target'].tolist()) if i not in test_indices]
}
test_data = {
    'filenames': [x for i,x in enumerate(dataset['filenames'].tolist()) if i in test_indices],
    'targets': [x for i,x in enumerate(dataset['target'].tolist()) if i in test_indices]
}
print(training_data)
print(test_data)
acceptableSensors = [2]
validation_indices = sample(range(0, len(training_data['filenames'])), validation_count)
validation_data = {
    'filenames': [x for i,x in enumerate(training_data['filenames']) if i in validation_indices],
    'targets': [x for i,x in enumerate(training_data['targets']) if i in validation_indices]
}
print(validation_data)
training_data = {
    'filenames': [x for i,x in enumerate(training_data['filenames']) if i not in validation_indices],
    'targets': [x for i,x in enumerate(training_data['targets']) if i not in validation_indices]
}
print(training_data)
minimumBar = FloatProgress(min=0, max=100, description="Determining Minimum Record Count:")
display(minimumBar)
data = pd.read_csv(dataset['filenames'][0])
data = data.loc[data["sensor"].isin(acceptableSensors)]
minimumLength = data.shape[0]
for filename in dataset['filenames']:
    data =  pd.read_csv(filename)
    data = data.loc[data["sensor"].isin(acceptableSensors)]
    dataLength = data.shape[0]
    if (dataLength < minimumLength):
        minimumLength = dataLength
    minimumBar.value += 1/len(dataset['filenames'])*100
model = SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, max_iter=3000, tol=None, shuffle=True, verbose=0, learning_rate='adaptive', eta0=0.01, early_stopping=False)
trainingBar = FloatProgress(min=0, max=100, description="Training Model:")
display(trainingBar)
lastLoss = 10000
worseCount = 0
validationReports = []
for e in range(0, max_epoch):
    for i in range(0, len(training_data['filenames'])):
        data = pd.read_csv(training_data['filenames'][i])
        data = data.loc[data["sensor"].isin(acceptableSensors)]
        data["displacement"] = data.diffX.abs() + data.diffY.abs() + data.diffZ.abs()
        data = data.sort_values("displacement", ascending=False).head(minimumLength).drop(["displacement"], axis=1).values.flatten()
        model.partial_fit([data], [training_data['targets'][i]], [0,1])
        trainingBar.value += 1/(len(training_data['filenames'])*max_epoch)*100
    validationPredictions = []
    for i in range(0, len(validation_data['filenames'])):
        data = pd.read_csv(validation_data['filenames'][i])
        data = data.loc[data["sensor"].isin(acceptableSensors)]
        data["displacement"] = data.diffX.abs() + data.diffY.abs() + data.diffZ.abs()
        data = data.sort_values("displacement", ascending=False).head(minimumLength).drop(["displacement"], axis=1).values.flatten()
        validationPredictions.append(model.predict([data])[0])
    validationReport = classification_report(validation_data['targets'], validationPredictions, zero_division=1, output_dict=True, target_names=dataset["target_names"])
    validationLoss = hinge_loss(validation_data['targets'], validationPredictions)
    validationReport["model_loss"] = validationLoss
    validationReports.append(validationReport)
    print("Val Repo", validationReport)
    print("Val Loss", validationLoss)
    if lastLoss <= validationLoss:
        worseCount += 1
    else:
        worseCount = 0
    if worseCount == patience:
        break
    lastLoss = validationLoss
predictionsBar = FloatProgress(min=0, max=100, description="Making Predictions:")
display(predictionsBar)
seterr(divide = 'ignore')
seterr(invalid='ignore')
predictions = []
for i in range(0, len(test_data['filenames'])):
    data = pd.read_csv(test_data['filenames'][i])
    data = data.loc[data["sensor"].isin(acceptableSensors)]
    data["displacement"] = data.diffX.abs() + data.diffY.abs() + data.diffZ.abs()
    data = data.sort_values("displacement", ascending=False).head(minimumLength).drop(["displacement"], axis=1).values.flatten()
    predictions.append(model.predict([data])[0])
    predictionsBar.value += 1/len(test_data['filenames'])*100    
report = classification_report(test_data['targets'], predictions, zero_division=1, output_dict=True, target_names=dataset["target_names"])
with open("./Model Results/sv/3day/gyro/results_{:.0f}.json".format(tm.time()), "w") as f:
    f.write(dumps({
        'training_data': training_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'predictions': [int(x) for x in predictions],
        'classification_report': report,
        'epochs': e,
        'reports': validationReports
    }, indent=4))
print(report)
