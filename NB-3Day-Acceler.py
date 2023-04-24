from sklearn.datasets import load_files
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from ipywidgets import FloatProgress
from IPython.display import display
from random import sample
from json import dumps
from numpy import seterr
import pandas as pd
import time as tm
dataset = load_files("./Training Data/3Day", load_content=False)
test_indices = sample(range(0, len(dataset['filenames'])), 4)
training_data = {
    'filenames': [x for i,x in enumerate(dataset['filenames'].tolist()) if i not in test_indices],
    'targets': [x for i,x in enumerate(dataset['target'].tolist()) if i not in test_indices]
}
test_data = {
    'filenames': [x for i,x in enumerate(dataset['filenames'].tolist()) if i in test_indices],
    'targets': [x for i,x in enumerate(dataset['target'].tolist()) if i in test_indices]
}
print(test_data)
acceptableSensors = [1]
minimumBar = FloatProgress(min=0, max=100, description="Determining Minimum Record Count:")
display(minimumBar)
data = pd.read_csv(dataset['filenames'][0])
data = data.loc[data["sensor"].isin(acceptableSensors)]
print(data)
minimumLength = data.shape[0]
for filename in dataset['filenames']:
    data =  pd.read_csv(filename)
    data = data.loc[data["sensor"].isin(acceptableSensors)]
    dataLength = data.shape[0]
    if (dataLength < minimumLength):
        minimumLength = dataLength
        print(minimumLength)
    minimumBar.value += 1/len(dataset['filenames'])*100
model = GaussianNB()
trainingBar = FloatProgress(min=0, max=100, description="Training Model:")
display(trainingBar)
for i in range(0, len(training_data['filenames'])):
    data = pd.read_csv(training_data['filenames'][i])
    data = data.loc[data["sensor"].isin(acceptableSensors)]
    data["displacement"] = data.diffX.abs() + data.diffY.abs() + data.diffZ.abs()
    data = data.sort_values("displacement", ascending=False).head(minimumLength).drop(["displacement"], axis=1).values.flatten()
    model.partial_fit([data], [training_data['targets'][i]], [0,1])
    trainingBar.value += 1/len(training_data['filenames'])*100
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
with open("./Model Results/nb/3day/accel/results_{:.0f}.json".format(tm.time()), "w") as f:
    f.write(dumps({
        'training_data': training_data,
        'test_data': test_data,
        'predictions': [int(x) for x in predictions],
        'classification_report': report
    }, indent=4))
print(report)
