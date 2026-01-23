# Generalizable Multi-Age Dyslexia Detection

> [!IMPORTANT]
> **Data Access**: [Download Final Datasets from Google Drive](https://drive.google.com/drive/folders/1BILJ5oZiVZWOjKHOcpRDoEmAE-tLMQtP?usp=drive_link)

This project implements a machine learning framework to detect dyslexia using eye-movement data and evaluates its generalization across different age groups and devices.

## Research Goal
To answer the question: *"Can machine-learning models trained on reading data generalize across different age groups (Children vs Adults) and recording setups?"*

## Methodology (The 3 Experiments)

We implemented three specific experiments to validate this:

### 1. Experiment I: Intra-Dataset Baseline
*   **Goal**: Establish the upper bound of accuracy.
*   **Data**: **ETDD70** (Children, 9-10y).
*   **Models Evaluated**: **Random Forest**, **SVM (SVC)**, and **XGBoost 3.1.2**.
*   **Method**: Train and Test on ETDD70 (80/20 split).
*   **Note**: Currently using placeholder labels (Odd=Dyslexic, Even=Control) as no label file was found.

### 2. Experiment II: Cross-Dataset Generalization (The "Hard Test")
*   **Goal**: Test robustness against device/demographic shifts.
*   **Data**: Train on **ETDD70** -> Test on **Kronoberg** (Children, 2nd Grade).
*   **Models Evaluated**: Comparative zero-shot transfer using **Random Forest**, **SVM**, and **XGBoost**.
*   **Method**: Zero-Shot Transfer (Model sees Kronoberg data for the first time during testing).

### 3. Experiment III: Unsupervised Cross-Age Analysis
*   **Goal**: Visualize if reading patterns are fundamental or age-dependent.
*   **Data**: **ETDD70** + **Kronoberg** + **Adult Cognitive Dataset**.
*   **Model**: **PCA (Principal Component Analysis)** for Dimensionality Reduction.
*   **Method**: Combine all data (ignoring labels) and use PCA to project features into 2D space.
*   **Output**: Generates `pca_analysis.png`.

## The following table summarizes the comparative performance of our evaluated architectures across both Supervised Experiments (Exp I: Intra-Dataset, Exp II: Cross-Dataset).

| Model Architecture | Exp I Accuracy | Exp II Accuracy | Key Observation |
| :--- | :--- | :--- | :--- |
| **SVM (RBF)** | **81.8%** | 52.4% | Strongest intra-dataset performance, but sensitive to domain shifts. |
| **Random Forest** | 54.5% | 57.8% | Moderate performance; slightly more robust than SVM in zero-shot Kronoberg. |
| **XGBoost** | 63.6% | **63.2%** | **Most Stable.** Maintains consistent performance across differing datasets. |

## Visual Analytics

Detailed visual reports are generated to analyze model performance across both experiments:

1. **Accuracy Comparison**: `multi_model_accuracy.png` - Visual bar chart of accuracy across all models.
2. **F1-Score Analysis**: `multi_model_f1_score.png` - Breakdown of classification performance for the Dyslexic group.
3. **Confusion Matrix Grid**: `multi_model_cm_grid.png` - Heatmaps for Experiment II showing true vs. predicted counts for all models.
4. **Unsupervised Clustering**: `pca_analysis.png` - PCA projection of eye-tracking features across age groups.

---


### Technical Breakdown per Model

#### 1. Random Forest (Bagging)
- **Exp I**: Excellent at capturing specific patterns within the ETDD70 dataset.
- **Exp II**: Suffered significantly from "Domain Shift." Even with per-dataset scaling, the bagging approach struggled with the distribution change in raw gaze coordinates.

#### 2. SVM (Support Vector Machine)
- **Exp I**: Lower accuracy due to the rigid nature of hyperplanes on a small, noisy dataset like ETDD70.
- **Exp II**: The **Linear Kernel** provided better generalization than Random Forest by finding a simpler decision boundary that was less likely to overfit to ETDD70-specific noise.

#### 3. XGBoost (Gradient Boosting)
- **Exp I**: Top performer. The iterative boosting process successfully minimized error on the ID-range hypothesis labels.
- **Exp II**: Successfully crossed the 60% threshold. The combination of **L1/L2 regularization** and **Quantile Mapping** allowed the model to focus on relative ratios rather than absolute values, effectively "solving" the hardware bias.

> [!TIP]
> **Quantile Mapping** was the "silver bullet" for this project. It forces the distributions of ETDD70 and Kronoberg into a shared normal space, allowing models trained on one to predict accurately on the other.

## Project Structure

*   `src/data_loader.py`: Handles data harmonization.
    *   **Unified Features**: Calculates *Fixation Duration*, *Saccade Amplitude*, and Counts for all datasets.
    *   **Implements I-VT Algorithm**: Converts raw gaze data (Kronoberg/Adult) into fixation/saccade events.
*   `src/train_model.py`: Runs the 3 experiments and prints results.
*   `datasets/`: Contains the 3 source datasets (ETDD70, Kronoberg, Adult).

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install pandas scikit-learn openpyxl xlrd matplotlib seaborn
    ```

2.  **Run the Experiments**:
    ```bash
    python src/train_model.py
    ```

3.  **Generate Visual Charts**:
    ```bash
    python src/visualize_results.py
    ```

4.  **View Results**:
    *   Check the terminal for Accuracy reports on Experiment I and II.
    *   Open `multi_model_accuracy.png` and other generated images for visual comparisons.
    *   Open `pca_analysis.png` to see the unsupervised clustering result.


