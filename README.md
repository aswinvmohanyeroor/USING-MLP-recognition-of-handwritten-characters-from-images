# Handwritten Character Recognition using Multilayer Perceptron (MLP)

## Overview
This project aims to develop a **Multilayer Perceptron (MLP)** model for recognizing handwritten characters from images. The model will first handle images containing a **single** handwritten character and later be extended to **detect and recognize multiple** handwritten characters in an image.

## Objectives
1. **Develop & Train MLP**: Train an MLP to recognize handwritten characters from single-character images.
2. **Evaluate Performance**: Assess the model using loss curves, accuracy metrics, and other performance measures.
3. **Hyperparameter Tuning**: Improve model accuracy through hyperparameter optimization.
4. **Multi-Character Recognition**: Extend the model to detect and recognize multiple handwritten characters in an image.

## Dataset
We use the **English Handwritten Characters Dataset** from Kaggle:  
[Dataset Link](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset)

This dataset contains images of handwritten English characters that will be used for training, validation, and testing.

## Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/handwritten-mlp.git
cd handwritten-mlp
```

### **2. Install Dependencies**
Ensure you have Python installed. Install required libraries using:
```bash
pip install -r requirements.txt
```

### **3. Download Dataset**
Download the dataset from Kaggle and place it in the `data/` directory.

## Model Architecture
We use a **Multilayer Perceptron (MLP)** architecture with the following layers:
- **Input Layer:** Flattened image pixels
- **Hidden Layers:** Fully connected layers with ReLU activation
- **Output Layer:** Softmax activation for character classification

### **Hyperparameter Tuning**
The following hyperparameters will be tuned:
- Number of hidden layers & neurons per layer
- Learning rate
- Batch size
- Regularization (dropout, L2 weight decay)

## Training & Evaluation
1. **Train the Model**
   ```bash
   python train.py
   ```
2. **Test & Evaluate**
   ```bash
   python evaluate.py
   ```

## Model Expansion: Multi-Character Recognition
The model will be extended to handle images containing **multiple characters** using:
- Image preprocessing for segmentation
- Localization algorithms
- Adjustments to the MLP structure

## Results & Performance
- **Evaluation Metrics:** Accuracy, Loss curves, Precision-Recall
- **Visualization:** Graphs of training loss & accuracy trends

## Justification of Design Choices
Each design choice (e.g., activation functions, layer sizes) will be theoretically justified and validated through experiments.


## Acknowledgments
- Kaggle Dataset
- TensorFlow & PyTorch Documentation

