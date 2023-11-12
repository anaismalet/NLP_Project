# NLP project: Predict TripAdvisor Reviews Rating

### Author

Anaïs Malet  
Email: anais.malet@epfedu.fr  
Date of creation: 16/10/2023

### Credits

The TripAdvisor Hotel Review Dataset file is sourced from the publication:

Alam, M. H., Ryu, W.-J., Lee, S., 2016. Joint multi-grain topic sentiment: modeling semantic aspects for online reviews. Information Sciences 339, 206–223.


### 1. Global Description

This Natural Language Processing (NLP) project aims to predict the star rating of hotel reviews from TripAdvisor. The analysis includes data exploration, the development of baseline and advanced machine learning models, and the implementation of deep learning techniques for improved predictive accuracy.

#### Notebook 1: Data Exploration
Data columns:
- **Review**: Textual conte nt of the user review.
- **Rating**: Star rating given by the user (target variable).

This notebook explores the dataset to understand its characteristics. Key insights, such as multi-class imbalance, are crucial for subsequent analysis.

#### Notebook 2: Baseline Model

This notebook focuses on preprocessing the data.
We will do those preprocessing Steps :
- Tokenization: Breaking down reviews into individual words.
- Stopword Removal: Eliminating common words without significant meaning.
- Lemmatization: Reducing words to their base or root form.

Then we will test multiple classifiers and vectorizers to establish a baseline model.

#### Notebook 3: Improve Baseline Model

In this notebook, we aim to enhance the baseline model's performance, especially in dealing with multi-class imbalance. Techniques such as class weighting, undersampling, oversampling, and hyperparameter tuning will be explored.

#### Notebook 4: Deep Learning

This notebook delves into building neural network models to capture complex relationships in the text data for improved predictive performance.

### 2. Dataset

The dataset used in this project consists of 20,000 reviews crawled from TripAdvisor. You can access the dataset through the following link: https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews

### 3. Model Performances

The table below summarizes the performance metrics of each implemented model:

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Baseline Model       | 0.64      | 0.58      | 0.59   | 0.58     |
| Improved Model       | 0.65     | 0.64      | 0.65   | 0.64     |
| Deep Learning Model  | 0.72     | 0.71      | 0.72   | 0.71     |

### 4. Installation and Execution

To install and run the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/anaismalet/NLP_Project.git
2. Navigate to the project directory:

    ```bash
    cd NLP_Project
3. Install dependencies using either requirements.txt file:

    Using requirements.txt:

    ```bash
    pip install -r requirements.txt
### 5. Additional Informations

#### Model Evaluation Metrics

- **Accuracy**: Overall correctness of the model.
- **Precision, Recall, F1-score**: Metrics to evaluate class-wise performance, crucial in the context of multi-class imbalance.

#### Project Dependencies

- Python
- Jupyter Notebooks
- Libraries: scikit-learn, TensorFlow/Keras, imbalanced-learn, pandas, matplotlib, seaborn, etc.

#### Future Work

Potential areas for future improvement and expansion include exploring advanced deep learning architectures, leveraging pre-trained embeddings, and incorporating domain-specific features.

Feel free to reach out for any further clarifications or collaboration opportunities.

