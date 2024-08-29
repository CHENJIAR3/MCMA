
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

### Contributions
- [x] Reconstructing 12-Lead ECG from Arbitrary Single-Lead ECG
- [x] Comprehensive Evaluations, including signal-level, feature-level, and diagnostic-level
      
### Saved models
- [x] An example model in MCMA could be seen in the Generator file
- [x] All the trained models: https://drive.google.com/drive/folders/1m57dz-FhcQCGNoZ2wxA_sUoHgrrGRHIn?usp=sharing
      
#### ** You can follow this repo for the newest information. **

It is the open-source code for MCMA, which could reconstruct 12-lead ECG with arbitrary single-lead ECG. 
Before running, you should load your ECG signals, and the amplitude unit should be __mv__!!! If not, you should adjust it in advance.

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

### Environment
```
conda env create -f environment.yml
```
### Something Interesting
MCMA can classify the lead index from 1 to 12, if you donot know it in advance. The index of maximum cc may be the ringht lead index.
### Future work
1. I have tried different setting, but more efforts in model designing are necessary for this task.
2. High quality ECG, although this study based on the public dataset, the signal quality influence its evaluation and reconstruction.

### Acknowledgements
**Contacting me at chenjr356@gmail.com, chenjr56@mail2.sysu.edu.cn, jiarong.chen@sjtu.edu.cn**
