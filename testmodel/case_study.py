import pickle
import pandas as pd
import numpy as np
if __name__=="__main__":
    testlabelpath = '/data/0shared/chenjiarong/ptb_xl/label_test.pkl'
    testdatapath='/data/0shared/chenjiarong/ptb_xl/ecg12_test.pkl'
    genpath="/data/0shared/chenjiarong/12lead_dataset/AE_12lead_1.pkl"
    #原始数据，已经验证跟file一致
    file1=open(testdatapath,'rb')
    data1 = pickle.load(file1)

    labelfile=open(testlabelpath,"rb")
    labeldata = pickle.load(labelfile)
    label = labeldata.values

    file = open(genpath, "rb")
    data=pickle.load(file)

    GENECG12 = data["GEN_ECG12"]
    ECG12=data["ECG12"]
    genecg12 = []
    ecg12 = []
    labelnew = []
    for i in range(ECG12.shape[0]):
        if len(label[i]) == 1:
            if label[i][0] == 'NORM':
                pass
            else:
                ecg12.append(ECG12[i])
                genecg12.append(GENECG12[i])
                labelnew.append(label[i][0])

    ecg12 = np.asarray(ecg12)
    genecg12 = np.asarray(genecg12)
    labelnew = np.asarray(labelnew)

    series1 = pd.Series(labelnew)
    print(series1.value_counts())


    data = {"ECG12": ecg12, "GEN_ECG12": genecg12, "label": labelnew}
    file = open("/data/0shared/chenjiarong/genECG4label/ECG4label_full.pkl", "wb")
    pickle.dump(data, file)
    file.close()

    series2 = pd.Series(labelnew[:100])
    print(series2.value_counts())
    data = {"ECG12": ecg12[:100], "GEN_ECG12": genecg12[:100], "label": labelnew[:100]}
    file = open("/data/0shared/chenjiarong/genECG4label/ECG4label_100.pkl", "wb")
    pickle.dump(data, file)
    file.close()