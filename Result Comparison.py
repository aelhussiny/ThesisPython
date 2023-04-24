import matplotlib.pyplot as plt
import os
import json
import numpy as np
models = ["nb","lg","sv"]
times = ["3day","1day","1hor","5min"]
sensors = ["allsensors","accel","gyro","ambi","accelgyro","accellight","gyrolight"]
model_results = {
    'accuracy': [],
    'female\nprecision': [],
    'female\nrecall': [],
    'female\nf1-score': [], 
    'male\nprecision': [], 
    'male\nrecall': [], 
    'male\nf1-score': [], 
    'mavg\nprecision': [], 
    'mavg\nrecall': [], 
    'mavg\nf1-score': [], 
    'wavg\nprecision': [], 
    'wavg\nrecall': [], 
    'wavg\nf1-score': [], 
}
bargroups = []
for model in models:
    for time in times:
        for sensor in sensors:
            path = os.path.join(r"C:\Users\ahme8608\Documents\Uni\Thesis\Test Notebooks\Model Results", model, time, sensor)
            if (os.path.exists(path)):
                bargroups.append("{}\n{}\n{}".format(model, time, sensor))
                results_file = os.path.join(path, sorted(os.listdir(path), reverse=True)[0])
                with open(results_file) as f:
                    result_data = json.load(f)
                    model_results['accuracy'].append(round(result_data["classification_report"]["accuracy"]*100, 2))
                    model_results['female\nprecision'].append(round(result_data["classification_report"]["female"]["precision"]*100, 2))
                    model_results['female\nrecall'].append(round(result_data["classification_report"]["female"]["recall"]*100, 2))
                    model_results['female\nf1-score'].append(round(result_data["classification_report"]["female"]["f1-score"]*100, 2))
                    model_results['male\nprecision'].append(round(result_data["classification_report"]["male"]["precision"]*100, 2))
                    model_results['male\nrecall'].append(round(result_data["classification_report"]["male"]["recall"]*100, 2))
                    model_results['male\nf1-score'].append(round(result_data["classification_report"]["male"]["f1-score"]*100, 2))
                    model_results['mavg\nprecision'].append(round(result_data["classification_report"]["macro avg"]["precision"]*100, 2))
                    model_results['mavg\nrecall'].append(round(result_data["classification_report"]["macro avg"]["recall"]*100, 2))
                    model_results['mavg\nf1-score'].append(round(result_data["classification_report"]["macro avg"]["f1-score"]*100, 2))
                    model_results['wavg\nprecision'].append(round(result_data["classification_report"]["weighted avg"]["precision"]*100, 2))
                    model_results['wavg\nrecall'].append(round(result_data["classification_report"]["weighted avg"]["recall"]*100, 2))
                    model_results['wavg\nf1-score'].append(round(result_data["classification_report"]["weighted avg"]["f1-score"]*100, 2))
figures = [
    {
        "name": "Model Results",
        "rel_keys": ['accuracy','mavg\nprecision', 'mavg\nrecall', 'mavg\nf1-score']
    }
]
for figure in figures:
    x = np.arange(len(bargroups))
    width = 1/(1+len(figure["rel_keys"]))
    multiplier = 0
    fig, ax = plt.subplots(tight_layout=True)
    fig.set_size_inches(100, 30)
    for attribute, measurement in dict(filter(lambda item: item[0] in figure["rel_keys"], model_results.items())).items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=5)
        multiplier += 1
    ax.set_ylabel("Value")
    ax.set_title(figure["name"])
    ax.set_xticks(x + width, bargroups)
    ax.legend(loc='upper center', ncols=len(figure["rel_keys"]))
    ax.set_ylim(0, 110)
    plt.savefig("{}.jpg".format(figure["name"]))