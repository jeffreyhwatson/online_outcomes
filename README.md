# Online Outcomes: Predicting Success in Virtual Learning

![graph0](./reports/figures/intro.jpg)

"Online Learning" by leanforward_photos is licensed under CC BY-SA 2.0

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
oo.yml` in your terminal. Next, run `conda activate oo`.
***
## Overview:

***
## Business Understanding
 
***
## Data Understanding
Citation:
Kuzilek, J., Hlosta, M. & Zdrahal, Z. Open University Learning Analytics dataset. Sci Data 4, 170171 (2017). [Article](https://doi.org/10.1038/sdata.2017.171)

Download Page:
[Open University Learning Analytics dataset](https://analyse.kmi.open.ac.uk/open_dataset)

ERD & Data Details:
[Schema & Data Description](https://analyse.kmi.open.ac.uk/open_dataset#description)

***
## Data Preparation
Data cleaning details for the project can be found here:
[Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)

***
# Exploring the  Data (Highlights From the EDA)
EDA for the project is detailed in the following notebook: [Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)

![graph1](./reports/figures/outcomes_type.png)
![graph2](./reports/figures/outcomes_imd.png)
![graph3](./reports/figures/outcomes_dis.png)
![graph4](./reports/figures/outcomes_age.png)
![graph5](./reports/figures/outcomes_edu.png)
![graph6](./reports/figures/outcomes_gen.png)
![graph7](./reports/figures/outcomes_cl.png)
![graph19](./reports/figures/outcomes_med.png)
![graph20](./reports/figures/outcomes_wa.png)
![graph21](./reports/figures/outcomes_al.png)
***
# Modeling

Details of the full modeling process can be found here:
[Modeling Notebook](./notebooks/exploratory/modeling_eda.ipynb)


## Baseline Model:

![graph8](./reports/figures/baseline.png)

<font size="4">Baseline Scores: Accuracy=.53, F1 = 0, Recall = 0, Precision = 0</font>

#### Score Interpretation

***
## First Simple Model:

<font size="4">Average Validation Scores: Accuracy=.61, F1 = .56, Recall = .53, Precision = .60</font>

![graph18](./reports/figures/fsm.png)

<font size="4">Scores on Test Data: Accuracy=.61, F1 = .56, Recall = .53, Precision = .60</font>

### Score Interpretation

***
## Feature Engineering & Intermediate Models
![graph12](./reports/figures/modeling_history.png)
***
## Final Model:
<font size="4"> </font>

<font size="4">Average Scores: F1=, Recall=, Precision=</font>

[comment]: <![graph12](./reports/figures/dummy.png)> 

<font size="4">Scores on Test Data: F1=, Recall=, Precision=</font>

#### Score Interpretation


## Alternate Final Model:  
<font size="4"></font>

<font size="4">Average Scores: F1=, Recall=, Precision=</font>


[comment]: <![graph16](./reports/figures/dummy.png)>

<font size="4">Scores on Test Data: F1=, Recall=, Precision=</font>

#### Score Interpretation


#### Notes on the Features

## Conclusion & Recommendations

## Next Steps

## For More Information

Please review our full analysis in our [Jupyter Notebook](./notebooks/report/report.ipynb) or our [presentation](./reports/presentation.pdf).

For any additional questions, please contact **Jeffrey Hanif Watson jeffrey.h.watson@protonmail.com**
