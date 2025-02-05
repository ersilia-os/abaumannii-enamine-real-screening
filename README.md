# Acinetobacer baumannii Enamine REAL screening
Screening the Enamine REAL 1.7B Leadlike database against a series of _Acinetobacter baumannii_ activity prediction models.

## Background

This repository contains code to perform a virtual screening for active compounds against _Acinetobacter baumannii_. The goal is to select 10 compounds for experimental testing at the [H3D Centre (Cape Town)](https://h3d.uct.ac.za/), obtained from the Enamine REAL library.

## Data collection

### Training data

Data consists of _Acinetobacter baumannii_ screening data collected from ChEMBL, Co-ADD and Spark. Datasets are binarized (i.e. active/inactive). We have used the following repositories to collect these datasets:

- [chembl-binary-tasks](https://github.com/ersilia-os/chembl-binary-tasks): Datasets collected from ChEMBL.
- [coadd-binary-tasks](https://github.com/ersilia-os/coadd-binary-tasks): Datasets extracted from Co-ADD and Spark (note that Spark is available through the Co-ADD web portal).

Data can be found in the `data/training` directory.

### Libraries for prediction

In addition, we have downloaded the 9.6M and the 95.6M subsets of the [Enamine REAL library](https://enamine.net/compound-collections/real-compounds/real-database-subsets), as well as the 1.7B leadlike subset (January 2025). These files are stored in `data/prediction`.

### Other data

We have also downloaded chemical structures from ChEMBL, found in `data/other`.

## Pipeline

The pipeline to run the virtual screening consists of several steps as listed in the `scripts` folder. Data from resulting from running these scripts is generally stored in the `processed` subfolder.

In this [Google Drive folder](https://drive.google.com/drive/folders/1I_0J3gYGC8oC_1xhg6rmYp576mEja5x1?usp=drive_link) you can find the `data` and `processed` folders with all precalculations done.

### Training and preparation

Steps `00`to `03` are focused on training baseline models for each of the ~30 datasets related to _A.baumannii_. We build Naive Bayes models as well as Random Forest models using FLAML (autoML). In addition, we prepare FPSim2 databases that will be useful for downstream filtering. 

### Fingerprint calculation

In steps `04` to `06`, we calculate Morgan count fingerprints for Enamine REAL compounds. These are stored as `.h5` files.

### Prediction and filtering with a single model

In step `07`, we make predictions using the Naive Bayes models on the 9M library. Based on these predictions, in step `08` we build a master model (single task) that will allow us to rapidly filter compounds from the 1.7B library (step `09`), resulting in roughly 100M compounds.

### Predictions and filtering across all models

The 100M compounds (i.e. subset of the 1.7B leadlike library) are then screened against all FLAML models (step `10`) and subsequently filtered down to ~2000 molecules (step `11`). In addition, note that, to favour novelty, a similarity filter is applied (step `12`) to remove molecules with represented scaffolds in training set, as well as similarity matches against the ChEMBL chemical space.

### Using the Ersilia Model Hub

In steps `13` to `15` we use the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia) to further filter our compounds.

* [eos3804](https://github.com/ersilia-os/eos3804): Third party _Acinetobacter baumannii_ bioactivity prediction model.
* [eos74km](https://github.com/ersilia-os/eos74km): Broadly predicts whether compounds are active against gram-negative bacteria.
* [eos42ez](https://github.com/ersilia-os/eos42ez): Cytotoxicity prediction, originally developed for an AMR screening.

### Selection of candidates

Steps `16`to `17` simply merge results, producing files for manual exploration. An aggregate score can be used to initially rank compounds. In the manual selection, we tried to balance for diversity of synthons, especially given the fact that a few scaffolds dominated the ranking. The `results/Predictions_Abaumannii.xlsx` spreadsheet contains the selected 10 candidates, which are the following:

Enamine identifiers:

```text
s_27____9164738____26671526
s_27____25546400____26671526
s_27____12122390____26671526
s_27____7763618____26671526
s_27____15142214____26671532
s_27____25522992____26671532
s_27____25523342____26671532
m_274552____23521694____25011290____25051654
m_274552____23521164____24988990____25051654
m_274552____14985752____24978212____25051654
```

Compound structures:

```text
C[C@H]1COC2=C3C(=CC(F)=C2N2CCCCO2)C(=O)C(C(=O)O)=CN31
C[C@H]1COC2=C3C(=CC(F)=C2NC[C@@H]2C[C@H]2O)C(=O)C(C(=O)O)=CN31 |&1:13,15|
C[C@H]1COC2=C3C(=CC(F)=C2N[C@H]2C[C@H](O)C2)C(=O)C(C(=O)O)=CN31
C[C@H]1COC2=C3C(=CC(F)=C2NC2(CO)CC2)C(=O)C(C(=O)O)=CN31
O=C(O)C1=CNC2=CC(N3[C@@H]4CC[C@H]3C[C@@H](O)C4)=C(F)C=C2C1=O
CO[C@@H]1C[C@H]2CC[C@@H](C1)N2C1=C(F)C=C2C(=O)C(C(=O)O)=CNC2=C1
COC1CC2CC1CN(C1=C(F)C=C3C(=O)C(C(=O)O)=CNC3=C1)C2
CC1CCCC(C)N(C(=O)C2CNC(=O)CN2C(=O)C2[C@H](C)[C@H]2C)C1 |&1:20,22|
CC1CN(C(=O)[C@@H]2C[C@H](F)CCN2C(=O)C2[C@H](C)[C@H]2C)C(C)C1C |&1:6,8,&2:16,18|
CO[C@@H]1C[C@H](C(=O)NCC2[C@H](C)[C@H]2C)N(C(=O)C2[C@H](C)[C@H]2C)C1 |&1:10,12,&2:18,20|
```

## About the Ersilia Open Source Initiative

[Ersilia](https://ersilia.io) is a tech non-profit organization aimed at fueling research in the Global South. We provide open-source AI/ML support to drug discovery laboratories in Africa and beyond. Visit our [GitBook documentation](https://ersilia.gitbook.io) and our [GitHub profile](https://github.com/ersilia-os) for more information.  