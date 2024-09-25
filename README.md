<div align="center">

# **M**ulti-**O**rgan **FO**undation (**MOFO**) Model

</div>

<div align="center">

## 🔥🔥🔥

#### Updated on 2024.10.06

</div>

#### The MOFO in multi-organ universal segmentation for ultrasound images serve as a pioneering exploration of improving segmentation performance by leveraging semantic and anatomical relationships within ultrasound images of multiple organs!


<hr style=" height:2px;border:none;border-top:2px dotted #185598;" />

## ✨Paper

This repository provides the official implementation of Multi-Organ FOundation (MOFO) Model.

**Multi-Organ Foundation Model for Universal Ultrasound Image Segmentation with Task Prompt and Anatomical Prior**

### Key Features

- Our **MOFO** is a segmentation foundation model formalized for multi-organ ultrasound universal segmentation.
- A hybri d US image database consisting of diverse images of 10 organs from patients across multiple medical centers is assembled to demonstrate the effectiveness of the **MOFO**.

<div align="center">

<img src="documents/Ultrasound images of various organs in the human body.png" width = "426" height = "332" alt="" align=center />

Ultrasound images of various organs in the human body. (a) Midbrain. (b) Thyroid. (c) Breast. (d) Median nerve. (e) Leg muscle. (f) Lymph node. (g) Heart. (h) Kidney. (i) Fetus. (j) Achilles tendon.

</div>

## ✨Architecture of MOFO
The MOFO comprises prompt, vision, and prior branches. The vision branch extracts organ-invariant representations from images for segmentation. The prompt branch aggregates task prompt and image features to generate organ-specific representations, guiding the vision branch in segmenting objects. The prior branch utilizes the anatomical prior encoder to map the prediction maps and ground truth into the prior space, applying constraints to ensure consistency.

<div align="center">

<img src="documents/Overview of MOFO.png" width = "762" height = "399" alt="" align=center />

</div>


## ✨Dataset

|    |      Organ      |  Source | People | People |          Target         | Links                                                                                     |
|---:|:---------------:|:-------:|:------:|:------:|:-----------------------:|:-----------------------------------------------------------------------------------------:|
|  1 |      Breast     |   Open  |    -   |   647  |       Breast tumor      | [BUSI](https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset)|
|  2 |      Fetus      |   Open  |    -   |   999  |        Fetal head       | [HC18](https://zenodo.org/records/1327317)                                                |
|  3 |    Leg muscle   |   Open  |    -   |   309  |     Lower leg muscle    | [FALLMUD](https://kalisteo.cea.fr/index.php/fallmud/)                                     |
|  4 |     Thyroid     |   Open  |    -   |   637  |      Thyroid nodule     | [DDTI](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view)            |
|  5 | Achilles tendon | Private |   172  |  1048  |     Achilles tendon     | -                                                                                         |
|  6 |      Heart      | Private |   218  |   218  | Interventricular septum | -                                                                                         |
|  7 |      Kidney     | Private |   392  |  1102  |       Kidney tumor      | -                                                                                         |
|  8 |    Lymph node   | Private |   815  |   815  |        Lymph node       | -                                                                                         |
|  9 |   Median nerve  | Private |   520  |  1059  |       Median nerve      | -                                                                                         |
| 10 |     Midbrain    | Private |   205  |   205  |         Midbrain        | -                                                                                         |


## ✨Installation & Preliminary
1. Clone the repository.
    ```
    git clone https://github.com/Git-HB-CHEN/MOFO.git
    cd MOFO
    ```
2. Create a virtual environment for MOFO and activate the environment.
    ```
    conda create -n MOFO python=3.9
    conda activate MOFO
    ```
3. Install Pytorch.
   (You can follow the instructions [here](https://pytorch.org/get-started/locally/))

5. Install other dependencies.
   ```
    pip install -r requirements.txt
   ```

## ✨Direct Inference in Your Ultrsound Images
1. Download the [Weight](https://drive.google.com/drive/folders/1SpBRMuM3hCZj9RhriHkMrMIGMcT0TamA?usp=drive_link) of the MOFO

2. Place your ultrasound images in the `examples` folder

3. Infer your ultrasound images with the MOFO
   ```
    python infer.py
   ```

## ✨Training the MOFO with Your OWN Datasets

1. Provide the `Multi-Organ Database` folder to help you organize your dataset. All images were saved in `PNG` format. No special pre-processed methods are used in data preparation.

2. Train the MOFO with your own datasets
   ```
    python train.py
   ```
3. Test the MOFO with your own datasets
   ```
    python test.py
   ```

