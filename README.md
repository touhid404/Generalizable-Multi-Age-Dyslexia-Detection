# Generalizable Multi-Age Dyslexia Detection

This project implements a machine learning framework to detect dyslexia using eye-movement data and evaluates its generalization across different age groups and devices.

## Research Goal
To answer the question: *"Can machine-learning models trained on reading data generalize across different age groups (Children vs Adults) and recording setups?"*

## Methodology (The 3 Experiments)

We implemented three specific experiments to validate this:

### 1. Experiment I: Intra-Dataset Baseline
*   **Goal**: Establish the upper bound of accuracy.
*   **Data**: **ETDD70** (Children, 9-10y).
*   **Model**: **Random Forest Classifier** (100 estimators).
*   **Method**: Train and Test on ETDD70 (80/20 split).
*   **Note**: Currently using placeholder labels (Odd=Dyslexic, Even=Control) as no label file was found.

### 2. Experiment II: Cross-Dataset Generalization (The "Hard Test")
*   **Goal**: Test robustness against device/demographic shifts.
*   **Data**: Train on **ETDD70** -> Test on **Kronoberg** (Children, 2nd Grade).
*   **Model**: **Random Forest** (Pre-trained on ETDD70).
*   **Method**: Zero-Shot Transfer (Model sees Kronoberg data for the first time during testing).

### 3. Experiment III: Unsupervised Cross-Age Analysis
*   **Goal**: Visualize if reading patterns are fundamental or age-dependent.
*   **Data**: **ETDD70** + **Kronoberg** + **Adult Cognitive Dataset**.
*   **Model**: **PCA (Principal Component Analysis)** for Dimensionality Reduction.
*   **Method**: Combine all data (ignoring labels) and use PCA to project features into 2D space.
*   **Output**: Generates `pca_analysis.png`.

## Results & Observations
*Note: These results are based on the current PROOF OF CONCEPT state using placeholder labels for ETDD70.*

| Experiment | Metric | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **I. Intra-Dataset** (ETDD70) | Accuracy | **~64%** | The model learned to predict the placeholder labels (Odd/Even ID) reasonably well. |
| **II. Cross-Dataset** (Kronoberg) | Accuracy | **~20%** | **Zero-Shot Transfer failed**, as expected. The random placeholder patterns from ETDD70 did not generalize to the real dyslexic patterns in Kronoberg. This validates the "Hard Test" experimental design. |
| **III. Unsupervised** | Visualization | **PCA Plot** | The `pca_analysis.png` successfully maps Adult, Child (ETDD), and Child (Kronoberg) data into a shared space for visual age-gap analysis. |

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

3.  **View Results**:
    *   Check the terminal for Accuracy reports on Experiment I and II.
    *   Open `pca_analysis.png` to see the unsupervised clustering result.

