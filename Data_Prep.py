import pandas as pd
import os
gender = "female"
slicing = "snz"
filenames = []
for (dpath, dnames, fnames) in os.walk(os.path.join(".","output",gender,slicing,"delta")):
    filenames.extend(fnames)
filenames = sorted(filenames, key = lambda filename : int(filename[0:filename.index("_")]))
lastDay = 1
dayFrame = pd.DataFrame()
for filename in filenames:
    path = os.path.join(".", "output", gender, slicing, "delta", filename)
    fileFrame = pd.read_csv(path)
    sensor = filename[-5:-4]
    print(filename, sensor)
    if not fileFrame.empty:
        if int(sensor) == 1:
            fileFrame[['diffX','diffY','diffZ']] = fileFrame['diff'].str.split(",", expand=True)
            fileFrame[['diffX','diffY','diffZ']] = fileFrame[['diffX','diffY','diffZ']].apply(pd.to_numeric, errors="coerce")
            fileFrame = fileFrame.iloc[:, [1,3,5,6,7]]
        elif int(sensor) == 2:
            fileFrame[['diffX','diffY','diffZ']] = fileFrame['diff'].str.split(",", expand=True)
            fileFrame = fileFrame._convert(numeric=True)
            fileFrame['diffZ'] = fileFrame['diffZ'].fillna(0)
            fileFrame[['diffX','diffY','diffZ']] = fileFrame[['diffX','diffY','diffZ']].apply(pd.to_numeric, errors="coerce")
            fileFrame = fileFrame.iloc[:, [1,3,5,6,7]]
        elif int(sensor) == 3:
            fileFrame['diffX'] = fileFrame['diff']
            fileFrame.insert(len(fileFrame.columns), "diffY", 0)
            fileFrame.insert(len(fileFrame.columns), "diffZ", 0)
            fileFrame[['diffX','diffY','diffZ']] = fileFrame[['diffX','diffY','diffZ']].apply(pd.to_numeric, errors="coerce")
            fileFrame = fileFrame.iloc[:, [1,3,5,6,7]]
        print(fileFrame.head(3))
        if filename[:filename.index("_")] == str(lastDay):
            print("Appending")
            dayFrame = pd.concat([dayFrame, fileFrame])
        else:
            print("Saving")
            if not dayFrame.empty:
                print("DAY " + str(lastDay))
                print(dayFrame.head(3))
                dayFrame.to_csv("./output/{}/{}/delta/nbprep/{}/{}.csv".format(gender, slicing, gender, str(lastDay)), index=False)
                dayFrame = fileFrame
                lastDay = filename[:filename.index("_")]
