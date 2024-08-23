import glob
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
    output_table=0
    output_fig=1
    result_path="/home/chenjiarong/generating12lead_CJ3/results/ptbxl/AE_result/AE_result"
    csv_paths = glob.glob(result_path+"/*.csv")
    sorted_paths = sorted(csv_paths, key=lambda x: (x.split('/')[-1].split('_')[2]))

    result_all=[]
    for csv_path in sorted_paths:
        a = pd.read_csv(csv_path).values
        if a.shape[0]==2203:
            a=np.delete(a,1571,axis=0)
        result_all.append(a)
    if output_fig:
        #1/10/11/12/2
        for j in range(12):
            for i in range(11):
                plt.subplot(3,4,i+1)
                plt.plot(result_all[0][:, i], result_all[j+1][:, i], 'rp')

            plt.show()
    if output_table:
    # 计算差值
        vals=np.zeros((12,11))
        re_vals=np.zeros((12,11))

        for i in range(12):
            for j in range(11):
                temp=abs(result_all[i + 1][:, j] - result_all[0][:, j])
                idx=np.isnan(temp)
                temp=np.delete(temp,idx,axis=0)
                a = np.delete(result_all[0][:,j],idx,axis=0)
                vals[i,j]=np.mean(temp)

                idx2= np.where(a==0.0)
                a = np.delete(a,idx2,axis=0)
                temp=np.delete(temp,idx2,axis=0)
                re_vals[i,j] = np.mean(temp/a)

        a = np.delete(result_all[0], np.where(np.isnan(result_all[0]) == 1)[0], axis=0)
        mean_val=np.mean(a,axis=0)
        df1 = pd.DataFrame(vals, index=[' I', ' II', ' III', ' aVR', ' aVL', ' aVF',
                                        ' V1', ' V2', ' V3', ' V4', ' V5', ' V6'],
                           columns=['HR', '	P_amplitude', '	P_duration', '	PR_interval', '	QRS_amplitude',
                                    '	QRS_duration', '	T_amplitude', '	ST', '	QT_interval', '	QTc',
                                    '	HRV']).round(4)
        df2 = pd.DataFrame(re_vals, index=[' I', ' II', ' III', ' aVR', ' aVL', ' aVF',
                                        ' V1', ' V2', ' V3', ' V4', ' V5', ' V6'],
                           columns=['HR', '	P_amplitude', '	P_duration', '	PR_interval', '	QRS_amplitude',
                                    '	QRS_duration', '	T_amplitude', '	ST', '	QT_interval', '	QTc',
                                    '	HRV']).round(4)
        df3 = pd.DataFrame(mean_val,
                           index=['HR', '	P_amplitude', '	P_duration', '	PR_interval', '	QRS_amplitude',
                                    '	QRS_duration', '	T_amplitude', '	ST', '	QT_interval', '	QTc',
                                    '	HRV']).round(4)
        with pd.ExcelWriter("/home/chenjiarong/generating12lead_CJ3/results/ptbxl/AE_result/diff_with_real12lead.xlsx",
                            engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='absolute_difference', index=True)
            df2.to_excel(writer, sheet_name='relative_difference', index=True)
            df3.to_excel(writer, sheet_name='mean_value', index=True)