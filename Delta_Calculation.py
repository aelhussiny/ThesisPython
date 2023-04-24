import pandas as pd
import os
import numpy as np
sensors = [1, 2, 3]
gender = "male"
curslice = "snz"
filenames = []
for (dpath, dnames, fnames) in os.walk(os.path.join(".","output",gender,curslice)):
    filenames.extend(fnames)
filenames = sorted(filenames, key = lambda filename : int(filename[0:filename.index("_")]))
day = 1
totalRows = 0
newTotalRows = 0
for idx in range(0, len(filenames)):
    filename = filenames[idx]
    print("PROGRESS: {}%. FILE NUMBER: {}/{}. NAME: {}".format(idx/len(filenames)*100, idx+1, len(filenames), filename))
    curday = filename[0:filename.index("_")]
    if (curday != day):
        day = curday
    path = os.path.join(".", "output", gender, curslice, filename)
    fileFrame = pd.read_csv(path)
    for sensor in sensors:
        print("--> SENSOR: {}".format(sensor))
        df = fileFrame[fileFrame["sensor"]==sensor].iloc[:, 2:]
        totalRows += len(df.index)
        if len(df.index) > 0:
            df["sensortime"] = df["sensortime"].diff()
            df = df[df["sensortime"] <= 1000]
            if sensor == 3:
                valDiffFrame = pd.DataFrame(df['value']).apply(pd.to_numeric, errors='raise').diff()
                df["diff"] = pd.DataFrame(valDiffFrame["value"].astype(str))
            else:
                valDiffFrame = pd.DataFrame(df['value'].str.split(",").tolist()).apply(pd.to_numeric, errors='coerce').diff()
                if len(valDiffFrame.index) > 0:
                    diffColFrame = pd.DataFrame({"diff": valDiffFrame[0].astype(str) + "," + valDiffFrame[1].astype(str) + "," + valDiffFrame[2].astype(str)})
                    df["diff"] = diffColFrame["diff"].values
            if 'diff' in df.columns:
                df = df[~df["diff"].isin([np.nan, "nan,nan,nan","nan", 0])]
                df = df[~df["sensortime"].isin([np.nan, "nan,nan,nan","nan", 0])]
                newTotalRows += len(df.index)
                df.to_csv("./output/{}/{}/delta/{}_{}.csv".format(gender, curslice, filename[0:filename.index(".")], sensor))
print("Reduction of {}%".format(newTotalRows/totalRows*100))
