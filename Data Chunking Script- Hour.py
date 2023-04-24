import sqlite3
import pandas as pd
con = sqlite3.connect("../Databases/database_copy_1621497074617_day20_m_p.db")
cur = con.cursor()
cur.execute("SELECT * FROM sensordata limit 1")
firstrow = cur.fetchall()[0]
startTime = firstrow[3]
sliceCount = 1
sliceSize = 1 * 60 * 60 * 1000
sliceType = "hour"
offset = 0
total = 0
batchSize = 1000
saveSize = 10000
lastFrameHadData = True
df = pd.read_sql_query("SELECT * from sensordata where 1 != 1", con)
while lastFrameHadData:
    processedTime = 0
    if (len(df.index) > 0):
        processedTime = (df.iloc[-1]["sensortime"] - startTime) / (sliceSize)
        processedId = df.iloc[-1]["_id"]
        print("Processing. Reached {} records totalling {:10.4f} {}s".format(processedId, processedTime, sliceType))
    try:
        newFrame = pd.read_sql_query("SELECT * from sensordata LIMIT {} OFFSET {}".format(batchSize, offset), con)
        newFrameSize = len(newFrame.index)
        total += newFrameSize
        if(newFrameSize > 0):
            offset += batchSize
            outsideTheSlice = newFrame[newFrame["sensortime"] >= startTime + sliceCount * sliceSize]
            if(len(outsideTheSlice.index) > 0):
                df = df.append(newFrame[newFrame["sensortime"] < startTime + sliceCount * sliceSize])
                df.to_csv("./output/male/{}/{}_{}.csv".format(sliceType, sliceCount, total))
                sliceCount += 1
                df = outsideTheSlice
            else:
                df = df.append(newFrame)
                if (len(df.index) >= saveSize):
                    df.to_csv("./output/male/{}/{}_{}.csv".format(sliceType, sliceCount, total))
                    cur.execute("DELETE from sensordata where _id in (SELECT _id from sensordata LIMIT {})".format(offset))
                    offset = 0
                    df = pd.read_sql_query("SELECT * from sensordata where 1 != 1", con)
        else:
            lastFrameHadData = False
            df.to_csv("./output/male/{}/{}_{}.csv".format(sliceType, sliceCount, total))
    except:
        lastFrameHadData = False
        df.to_csv("./output/male/{}/{}_{}.csv".format(sliceType, sliceCount, total))
con.close()
