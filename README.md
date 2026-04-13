# Projects-in-Advanced-Machine-Learning

---

# Project 1


## 🌍 Global Happiness Insights: A Machine Learning Approach
**Predicting National Happiness Scores using Socio-Economic Indicators**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Why are some countries happier than others? This project explores the complex relationship between a nation's socio-economic factors and its citizens' subjective well-being. Using the **World Happiness Report** dataset, we implement and compare traditional machine learning (Random Forest) and deep learning (MLP) to classify happiness levels.

Beyond mere prediction, this project leverages **SHAP (SHapley Additive exPlanations)** to "open the black box," revealing how variables like GDP, social support, and institutional trust non-linearly influence global happiness.

---

## 🛠️ Tech Stack
* **Modeling:** Scikit-learn (Random Forest), Keras/TensorFlow (MLP)
* **Interpretability:** SHAP (KernelExplainer & TreeExplainer)
* **Data Analysis:** Pandas, NumPy, Scipy (Pearson, Spearman, Kendall Correlations)
* **Visualization:** Matplotlib, Seaborn

---

## 📊 Key Research Findings

### 1. Model Performance & Stability
We conducted extensive experimentation with batch sizes and regularization techniques to combat overfitting in small-scale tabular data ($n=95$).
* **Random Forest:** Achieved the highest robustness, acting as our sociological baseline.
* **Regularized MLP:** By implementing **Dropout (0.3)**, we successfully mitigated severe overfitting, reducing the generalization gap and providing a stable foundation for SHAP analysis.

### 2. The Interpretability Divergence
One of the most significant insights from this project is how different models "see" the world:
* **Random Forest (The Balanced Sociologist):** Evenly distributes importance between GDP and Social Support.
* **Neural Network (The Feature Synthesizer):** While anchoring on GDP, it captures deeper non-linear signals from institutional trust (perception of corruption) and long-term education.



---

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/WrsRosanne/Projects-in-Advanced-Machine-Learning.git](https://github.com/WrsRosanne/Projects-in-Advanced-Machine-Learning.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow shap matplotlib seaborn
    ```
3.  **Run the Notebook:**
    Open `QMSSGR5074_Project_1_GitHub_Version.ipynb` in Jupyter or Google Colab and run all cells.

---

## 📁 Repository Structure
* `QMSSGR5074_Project_1_GitHub_Version.ipynb`: The primary research notebook including EDA, modeling, and SHAP analysis.
* `WHR_2023.csv`: The cleaned raw data.
* `newcountryvars.csv`: The cleaned raw data.

---

## 👥 Contributors
* **Rongshan Wei**
* **Lin Gan**

*Special thanks to the QMSS program at Columbia University for the academic support.*


---

# Project 2

## 🫁 Deep Learning for Medical Diagnostics: COVID-19 X-Ray Classification
**Comparative Analysis of Custom CNNs and State-of-the-Art Transfer Learning Architectures**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.0+](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-F44336?logo=keras&logoColor=white)](https://keras.io/)

## 📌 Project Overview
Can deep learning reliably distinguish between COVID-19, Viral Pneumonia, and healthy lung tissue? This project tackles the challenge of medical image classification using a dataset of chest X-rays. We moved beyond simple modeling to explore the **architectural resilience** of various deep learning frameworks.

The core of this research involves a rigorous comparison between a custom-built **Baseline CNN** and five pre-trained architectures (**ResNet50, MobileNetV2, DenseNet121, and VGG16**). We specifically investigate why "legacy" models like VGG16 fail in modern medical pipelines and how **Residual Connections** and **Batch Normalization** act as critical stabilizers in high-stakes diagnostic environments.

---

## 🛠️ Tech Stack
* **Frameworks:** TensorFlow / Keras
* **Architectures:** ResNet50, MobileNetV2, DenseNet121, VGG16
* **Data Engineering:** ImageDataGenerator (Spatial Augmentation), GlobalAveragePooling2D
* **Optimization:** Adam Optimizer, Learning Rate Scheduling (ReduceLROnPlateau), Early Stopping
* **Evaluation:** Confusion Matrices, Precision-Recall Analysis, Weighted F1-Score

---

## 📊 Key Research Findings

### 1. The Modern Architecture Advantage
Our experiments revealed a stark divide between "Residual" networks and "Legacy" networks:
* **The Champions:** Both our **Baseline CNN** and **ResNet50** achieved a high-performance plateau of **~93.3% accuracy**. ResNet50, in particular, demonstrated superior precision (**0.934**), which is critical for minimizing false positives in a clinical triage setting.
* **The VGG Collapse:** VGG16 proved to be a "glass cannon," suffering from catastrophic mode collapse (33% accuracy). This empirically demonstrated that without internal **Batch Normalization**, legacy models cannot handle the covariate shift inherent in specialized medical datasets.

### 2. Generalization & Data Augmentation
By implementing a strictly controlled augmentation pipeline (rotation, zoom, and shifts), we successfully improved **DenseNet121's** accuracy from 74% to 75%. While the numerical gain was modest, the model's **recall for Pneumonia reached a perfect 1.00**, proving that spatial variety hardens models against false negatives—a vital requirement for medical screening.

### 3. Identity Fine-Tuning Strategy
One of the most significant technical insights was the discovery of the **"Normalization Paradox"** during fine-tuning. We found that for ResNet50, "Identity Fine-Tuning" (using a $0.0$ learning rate) was the most robust way to preserve the 93% accuracy baseline, protecting the pre-trained feature maps from being scrambled by the noisy error gradients of a specialized medical dataset.

---

## 🚀 How to Run
1. **Clone the repository:**
    ```bash
    git clone [https://github.com/WrsRosanne/Projects-in-Advanced-Machine-Learning.git](https://github.com/WrsRosanne/Projects-in-Advanced-Machine-Learning.git)
    ```
2. **Install dependencies:**
    ```bash
    pip install pandas numpy tensorflow matplotlib seaborn scikit-learn
    ```
3. **Run the Notebook:**
    Open `TeamProject2_Notebook_Final.ipynb` in Google Colab (recommended A100/L4 GPU) or Jupyter and execute all cells.

---

## 📁 Repository Structure
* `TeamProject2_Notebook_Final.ipynb`: The complete research pipeline containing the Baseline CNN, four Transfer Learning experiments, and the final 2x2 performance dashboard.
* `TeamProject2_Notebook_Final.pdf`: A professionally rendered report of the project findings.
* `TeamProject2_Notebook_Final.html`: Interactive web version of the results.

---

## 👥 Contributors
* **Rongshan Wei**
* **Xiangxinrui Shan**
* **Ziyang Yu**

*Special thanks to the QMSS program at Columbia University for providing the deep learning frameworks and datasets for this research.*



