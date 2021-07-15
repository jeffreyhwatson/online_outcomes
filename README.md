# Online Outcomes: Predicting Success in Virtual Learning

![graph0](./reports/figures/aug_neg.png)

**Author:** Jeffrey Hanif Watson
***
### Quick Links
1. [Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)
2. [Modeling Notebook](./notebooks/exploratory/modeling_eda.ipynb)
3. [Final Report](./notebooks/report/report.ipynb)
4. [Presentation Slides](./reports/presentation.pdf)
***
### Repository Structure

```
├── README.md
├── online_outcomes.yml
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
│   ├── exploratory
│   └── report
├── references
├── reports
│   └── figures
└── src
```
***
### Setup Instructions:

#### Create & Activate Environment
`cd` into the project folder and run `conda env create --file
online_outcomes.yml` in your terminal. Next, run `conda activate online_outcomes`.

#### Download & Unzip Data
Download the zipped data from [Open University Learning Analytics](https://analyse.kmi.open.ac.uk/open_dataset), unzip the data and place the csv files into the `raw` directory.

#### Create Database
Run the database creation notebook: [Database Creator](./data/db_creator.ipynb).
***
## Overview:

***
## Business Understanding
 
***
## Data Understanding
[Open University Learning Analytics dataset](https://analyse.kmi.open.ac.uk/open_dataset) 

[Schema & Data Description](https://analyse.kmi.open.ac.uk/open_dataset#description)

***
## Data Preparation
Data cleaning details for the project can be found here:
[Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)

***
# Exploring the  Data (Highlights From the EDA)
EDA for the project is detailed in the following notebook: [Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)

***
# Modeling

Details of the full modeling process can be found here:
[Modeling Notebook](./notebooks/exploratory/modeling_eda.ipynb)


## Baseline Model:

![graph8](./reports/figures/dummy.png)

<font size="4">Baseline Scores: F1 = 0, Recall = 0, Precision = 0</font>

#### Score Interpretation
F1 is a mix of both precision and recall, so the interpretation of the results is more easily given in terms of recall and precision. 

***
## First Simple Model:

<font size="4">Average Validation Scores: F1=, Recall=, Precision=</font>



![graph18](./reports/figures/baseline_cm.png)

<font size="4">Scores on Test Data: F1 =, Recall = , Precision = </font>

### Score Interpretation
Since F1 is a mix of both precision and recall, the interpretation of the results is more easily described in terms of recall and precision. 

***
#### Feature Engineering & Intermediate Models

***
## Final Model:
<font size="4"> </font>

<font size="4">Average Scores: F1=, Recall=, Precision=</font>

![graph12](./reports/figures/tuned_logreg_cm.png)

<font size="4">Scores on Test Data: F1=, Recall=, Precision=</font>

#### Score Interpretation
From the confusion matrix we see that the model still has a little more trouble classifying negatives relative to positives, but the overall performance is acceptable.


## Alternate Final Model:  
<font size="4"></font>

<font size="4">Average Scores: F1=, Recall=, Precision=</font>


![graph16](./reports/figures/tuned_rf_cm.png)

<font size="4">Scores on Test Data: F1=, Recall=, Precision=</font>

#### Score Interpretation
From the confusion matrix we see that the model still has a little more trouble classifying negatives relative to positives, but the overall performance is acceptable.

#### Notes on the Features

# Conclusion


# Next Steps

# For More Information

Please review our full analysis in our [Jupyter Notebook](./notebooks/report/report.ipynb) or our [presentation](./reports/presentation.pdf).

For any additional questions, please contact **Jeffrey Hanif Watson jeffrey.h.watson@protonmail.com**
