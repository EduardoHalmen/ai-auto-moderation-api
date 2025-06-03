# Building a Toxic Comment Classification Models

This project focuses on the exploration toxic comments, with the goal of building robust text classification models. The primary dataset used is the **Jigsaw Unintended Bias in Toxicity Classification** dataset.

## Project Goals

- **Data Exploration:** Analyze and visualize the distribution and characteristics of toxic comments.
- **Data Processing:** Clean, preprocess, and engineer features from the raw text data.
- **Modeling:** Implement and evaluate machine learning and deep learning models for toxicity classification.
- **Bias Analysis:** Assess and mitigate unintended bias in model predictions.
- **Evaluation:** Use appropriate metrics to evaluate model performance and fairness.

## Dataset

- **Source:** [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)
- **Description:** Jigsaw curated dataset containing comments from online platforms labeled for toxicity.

## Project Structure

```
.
├── data/           # Raw and processed datasets
├── notebooks/      # Jupyter notebooks for exploration, processing, and modeling
├── src/            # Source code for data processing, modeling, and utilities
├── results/        # Model outputs, evaluation metrics, and visualizations
├── requirements.txt
└── README.md       
```

## Getting Started

1. Download the dataset from Kaggle and place it in the `data/` directory.
2. Run:
    ```
    ./setup.sh
    ```
3. Explore the notebooks in the `notebooks/`.
4. Deactivate environment
    ```
    deactivate
    ```

## Future Work

- Experiment with different NLP Text Classification models.
- Perform hyperparameter tuning and model optimization.
- Conduct in-depth bias and fairness analysis.
- Demo the best-performing model as an API.