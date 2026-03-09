# Projects-in-Advanced-Machine-Learning


# Project 1


# 🌍 Global Happiness Insights: A Machine Learning Approach
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
