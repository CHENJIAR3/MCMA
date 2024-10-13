
<h1 align="center"> 
Multi-Channel Masked Autoencoder and Comprehensive Evaluations for Reconstructing 12-Lead ECG from Arbitrary Single-Lead ECG
</h1>

<h3 align="center">
Accepted by KDD workshop-AIDSH 224</h3>
<h3 align="center">
First author: Jiarong Chen&nbsp;
</h3>

### Updates
- [x] Paper (Waiting for its publication, which included detailed information)
- [x] The application algorithm
- [x] The training algorithm
- [x] The testing algorithm
- [x] Pretrained weights
- [x] The samples for testing
- [x] The datasets
### Contributions
- [x] Reconstructing 12-Lead ECG from Arbitrary Single-Lead ECG
- [x] Comprehensive Evaluations, including signal-level, feature-level, and diagnostic-level
      
### Saved models
- [x] An example model in MCMA could be seen in the Generator file
- [x] All the trained models: https://drive.google.com/drive/folders/1m57dz-FhcQCGNoZ2wxA_sUoHgrrGRHIn?usp=sharing
      
#### ** You can follow this repo for the newest information. **

It is the open-source code for MCMA, which could reconstruct 12-lead ECG with arbitrary single-lead ECG. 
Before running, you should load your ECG signals, and the amplitude unit should be __mv__!!! If not, you should adjust it in advance.
Before running, you should resample this signal as 500Hz.
Before running, you can reshape it into (N,1024,1).
Additionallu, the lead index should be provided. If not, you can try to find it by locating the maximum PCC. The lead index classifcaitionn accuracy in the internal testing dataset is 97.43%. 

### Citation
If you find this project is useful, please cite **Multi-Channel Masked Autoencoder and Comprehensive Evaluations for Reconstructing 12-Lead ECG from Arbitrary Single-Lead ECG**
```
@inproceedings{
chen2024multichannel,
title={Multi-Channel Masked Autoencoder and Comprehensive Evaluations for Reconstructing 12-Lead {ECG} from Arbitrary Single-Lead {ECG}},
author={Jiarong chen and Wanqing Wu and Shenda Hong},
booktitle={Artificial Intelligence and Data Science for Healthcare: Bridging Data-Centric AI and People-Centric Healthcare},
year={2024},
url={https://openreview.net/forum?id=lIX6BKDPJW}
}
```
or 
```
@misc{chen2024multichannelmaskedautoencodercomprehensive,
      title={Multi-Channel Masked Autoencoder and Comprehensive Evaluations for Reconstructing 12-Lead ECG from Arbitrary Single-Lead ECG}, 
      author={Jiarong Chen and Wanqing Wu and Tong Liu and Shenda Hong},
      year={2024},
      eprint={2407.11481},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.11481}, 
}
```
### Files 
It includes demo.py, the trained model, and sample data.
## Datasets

All datasets used in this study are openly available. You can access them through the following links:

- **PTB-XL**: 
  [PhysioNet: PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/)
  
- **CPSC-2018**: 
  [Challenge of CPSC 2018](http://2018.icbeb.org/Challenge.html)
  
- **CODE-test**: 
  [CODE Test Dataset on Zenodo](https://zenodo.org/records/3765780)

Of course, I have uploaded these datasets online.

- For **Baidu Netdisk**: https://pan.baidu.com/s/1yycZodJyFAG95D_O6_jA_Q?pwd=MCMA 
- For  **Google Drive**: https://drive.google.com/drive/folders/1wGtq2D8Ssx9cRx_K3Li3t85nMc7WfTF1?usp=sharing

### Environment
```
conda env create -f environment.yml
```
You may fail to create your environment due to some issues, and I suggest you to pip install some packages.

### Future work
1. I have tried different setting, but more efforts in model designing are necessary for this task.
2. High quality ECG, although this study based on the public dataset, the signal quality influence its evaluation and reconstruction.

### Acknowledgements
**If you have any question, please contact me, I will try my best to help you.**
**Contacting me at chenjr356@gmail.com, chenjr56@mail2.sysu.edu.cn, jiarong.chen@sjtu.edu.cn**
