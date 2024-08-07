# Model Card


## Model Details
This is a RandomForestClassifier model using Scikit-learn. It utilizes Python 3.8 and Scikit-learn 1.0.2.
## Intended Use
This model is intended to predict if an individual's salary exceeds $50,000 per year based on census data. It could be used to understand demographic factors that influence salary levels and identify high earners for targeted marketing.
## Training Data
The census data is publicy available from the Census bureau. The training data is 80% of the original dataset and features age, workclass, Fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, Native-country, and salary.
## Evaluation Data
The evaluation data is 20% of the original dataset and the features are the same as the training data.
## Metrics
The following metrics were used to evaluate the model: precision, recall and f1 score. Precision measures the accuracy of the positive predictions. Recall measure the ability of the model to capture all the positive instances. F1 score is the harmonic mean of precision and recall.
Precision: 0.7791 | Recall: 0.6308 | F1: 0.6972
## Ethical Considerations
The model may have inherit biases in the training data, leading to unfair predictions across different demographic groups. It is important to ensure diverse and representative data. The use of personal demographic information requires strict privacy and protection measures. Model decisions should be interpretable and explainable to stakeholders.
## Caveats and Recommendations
The model's performance greatly depends on the quality and representativeness of the training data. If the data is small or not diverse, there is a risk of overfitting and needs to be regularly monitored. This model should be reviewed regularly and updated to reflect changes in the underlying data distribution.