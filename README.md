# ML Life Cycle: Data Understanding and Data Preparation
This project implements the Data Understanding and Data Preparation phases of the Machine Learning life cycle using the Airbnb NYC listings dataset. The goal is to prepare a clean, structured dataset suitable for building a regression model to predict listing prices.

## Project Overview
This project focuses on the Data Understanding and Data Preparation phases of the Machine Learning life cycle, applied to the Airbnb NYC listings dataset. The objective is to prepare a clean, structured dataset suitable for training a regression model to predict Airbnb listing prices.
The lab covers:

1. Build the Data Matrix & Define the ML Problem
- Loaded and inspected the Airbnb dataset.
- Defined the label (price), cleaned currency formatting, and converted it to numeric.
- Identified relevant features and removed non-predictive columns such as URLs and identifiers.

2. Clean the Data
- Outlier handling: Applied winsorization to replace top and bottom 1% price outliers, creating a new label column label_price.
- Missing values: Detected missing data, created missingness indicator columns (*_na), and imputed numerical missing values with column means.

3. Feature Transformation
- Applied one-hot encoding to categorical variables such as host_response_time and room_type to make them suitable for modeling.

4. Exploratory Data Analysis (EDA)
- Computed correlations between features and the label.
- Identified accommodates and bedrooms as the top predictors of price.
- Created bivariate kernel density plots to visualize the relationships.

5. Insights & Recommendations
- Price increases with listing size and capacity.
- Several unstructured text fields (name, description, amenities) could add predictive value with NLP preprocessing in future work.
- Suggested transformations for date fields and categorical binary text features before modeling.
Analysis & Insights

## Skills & Techniques
- Data Cleaning: Outlier handling, missing value imputation, feature selection.
- Feature Engineering: One-hot encoding categorical features, creating indicator variables.
- Exploratory Data Analysis: Correlation analysis, visualization with Seaborn.
- Pandas & NumPy: Data wrangling and transformation.
- Visualization: Pair plots and kernel density estimation.

## Files
- DataUnderstandingAndPreparation.ipynb – Jupyter Notebook with all code, outputs, and explanations.
- DataUnderstandingAndPreparation.py – Python script version of the lab.
- DataUnderstandingAndPreparation.pdf – Exported PDF of the lab results.

Installation & Usage
1. Clone the repository:
git clone https://github.com/yourusername/Lab2-DataUnderstandingAndPreparation.git
cd Lab2-DataUnderstandingAndPreparation
2. Set up a Python environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
3. Install dependencies
pip install -r requirements.txt
If requirements.txt is not provided, install manually:
pip install pandas numpy matplotlib seaborn scipy jupyter
4. Run the notebook locally
jupyter notebook DataUnderstandingAndPreparation.ipynb
5. Run the notebook online
You can run the .ipynb file directly in Jupyter Online without local setup.
