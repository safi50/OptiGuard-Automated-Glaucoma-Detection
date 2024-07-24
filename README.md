# OptiGuard: Generalized, Attention-Driven & Explainable Glaucoma Classification

This repository is a part of my final year project in the domain of Deep Learning, Explainable AI, and AI in Healthcare submitted to **the National University of Sciences & Technology (NUST)**. The project details are given below:

## 0. Project Demo:
[Link to Drive](https://drive.google.com/file/d/1aPLS0BRL4FcaeCkvHFiDAtbwQ2lEFaOu/view?usp=sharing)

https://github.com/user-attachments/assets/fee2ff44-7264-4df8-a775-42662ceec8e3



## 1. Abstract
Glaucoma is a leading cause of blindness that requires early diagnosis. This project introduces a generalized, attention driven & explainable system for early glaucoma classification using retinal fundus images via Convolutional Neural Networks (CNNs).
## 2. Problem Statement
Timely & accurate glaucoma detection is a challenge. Manual methods cause treatment 3. delays. This project aims to address this need with an automated explainable CNN system.
## Development Methodology
1. Utilizes the G1020 dataset for training. RFIs are preprocessed and augmented for variability. Detectron2 with Mask RCNN and ResNet-50 backbone is employed for OD and OC segmentation, validated with Average Precision metrics. Computes diagnostic metrics like CDR and NRR area.

2. Uses the SMDG-19 dataset, preprocessed with resizing, normalization, and augmentation including Histogram Equalization and CLAHE. Focuses on OD and OC regions identified by the segmentation module. Trains EfficientNet-B0 with Cross-Entropy loss and Adam optimizer, addressing class imbalance with weighted sampling.

3. Finally, Grad-CAM, Attention Mechanisms and LLM are employed for result explainability and interpretability.

## Technologies Used
- Python, PyTorch, Keras, Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn
- Detectron2
- Flask

## Results: 
<img width="433" alt="Screenshot 2024-07-17 at 7 55 15 PM" src="https://github.com/user-attachments/assets/a0337fb7-03a1-46b8-956c-edb52376be0e">
<img width="433" alt="Screenshot 2024-07-17 at 7 51 58 PM" src="https://github.com/user-attachments/assets/9c10ae4e-deeb-490a-88d1-9dbdf9e7a6de">

## Relevant Links: 
1. G1020 Dataset : https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets
2. SMDG Dataset : https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. You are free to use, distribute, and share this work for non-commercial purposes under the following conditions:
- **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial:** You may not use the material for commercial purposes.
- **NoDerivatives:** If you remix, transform, or build upon the material, you may not distribute the modified material.

## Note:
This project was successfully completed with contributions from: 
    - Syed Safi Ullah Shah (Safi50)
    - Humza Khawar
    - Muhammad Huzaifa
and under the advisory of: 
Dr. Muhammad Naseer Bajwa (https://scholar.google.com.pk/citations?user=PeeIGEgAAAAJ&hl=en)

