# Melanoma Classification with CNN

## Overview  
This project classifies skin lesion images as benign or malignant (melanoma) using a custom-built Convolutional Neural Network (CNN).

## Dataset  
The dataset used for this project can be found at [Melanoma Cancer Image Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/) on Kaggle.

The dataset already included train and test sets. I extracted 20% of the train dataset as dev (or val) dataset.
Each set contains both benign and malignant cases, balanced approximately equally.

## Approach  
- **Data Preprocessing:** Rescaled all images using `ImageDataGenerator` (rescale=1./255).  
- **Data Augmentation:** Applied medically relevant augmentations only on the training set to improve generalization.  
- **Exploratory Data Analysis:** Checked class distribution with bar plots and visualized random image samples with labels.  
- **Model Development:** Built and compared three CNN architectures with varying numbers of convolutional and dense layers.  
- **Model Selection:** Chose the best architecture based on dev set metrics.  
- **Hyperparameter Tuning:** Tuned learning rate and added L2 regularization (`lambda=0.001`) to improve the selected model further  
- **Custom Training Utilities:** Created custom `Checkpoint` and `EarlyStopping` classes to save models meeting criteria (`val_recall > 0.85` and `val_loss < 0.5`).  
- **Model Evaluation:** Selected the best saved model (`model_second`) and evaluated it on the test set at classification threshold 0.4.

## Final Model Performance (Threshold = 0.4)  
| Metric    | Value    |  
|-----------|----------|  
| AUC       | 0.9756   |  
| Accuracy  | 92.75%   |  
| Precision | 91.07%   |  
| Recall    | 94.8%    |  
| F1 Score  | 92.90%   |  

**Confusion Matrix (Test Set: 2000 images, 1000 benign and 1000 malignant):**  

|                 | Predicted Benign | Predicted Malignant |
|-----------------|------------------|--------------------|
| **Actual Benign**    | 907              | 93                 |
| **Actual Malignant** | 52               | 948                |


## Requirements  
Python 3.x, TensorFlow, Keras, scikit-learn, numpy, matplotlib

## Usage  
Run the notebook or Python script to:  
- Preprocess and augment images  
- Train CNN models and tune hyperparameters  
- Save best-performing models using custom callbacks  
- Evaluate and visualize test set performance, including confusion matrix  
- Load and use the saved model ([melanoma_model](https://www.kaggle.com/datasets/nadidixit/melanoma-model))

## Highlights  
- Built CNN models from scratch, without pretrained backbones  
- Emphasis on high recall to minimize false negatives
- Custom checkpoint and early stopping based on recall and loss  
- Thorough performance evaluation including multiple metrics and confusion matrix

## Future work  
- Explore transfer learning with pretrained CNNs (e.g., EfficientNet, ResNet)  
- Deploy model as an interactive web app  
- Validate on external datasets for robustness
