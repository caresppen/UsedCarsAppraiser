# Carlyst: Used Cars Appraiser
![Carlyst](https://drive.google.com/uc?export=view&id=1cf6kuyI7QMg5VDDLlsbfL-Yrpi-OF_ap)
Master Thesis. Master in Data Science at KSchool (2020-2021).

**Author**: [Carlos Espejo Peña](https://www.linkedin.com/in/carlosespejopena/)

This presentation act as a guidance for Carlyst project. It contains the structure of the repository, an overall concept of the objective of this project and a guide on how to use this [app](https://carlyst.herokuapp.com/).

## Contents
1. [Repository Structure](https://github.com/caresppen/UsedCarsAppraiser#repository-structure)
2. [Introduction to the problem](https://github.com/caresppen/UsedCarsAppraiser#introduction-to-the-problem)
3. [Objective](https://github.com/caresppen/UsedCarsAppraiser#objective)
4. [Notebooks](https://github.com/caresppen/UsedCarsAppraiser#notebooks)
    1. [Data Extraction and Cleansing](https://github.com/caresppen/UsedCarsAppraiser#data-extraction-and-cleansing)
    2. [Exploratory Data Analysis (EDA)](https://github.com/caresppen/UsedCarsAppraiser#exploratory-data-analysis-eda)
    3. [Methodology: Feature Engineering and Data Modelling](https://github.com/caresppen/UsedCarsAppraiser#methodology-feature-engineering-and-data-modelling)
5. [Carlyst: Streamlit app](https://github.com/caresppen/UsedCarsAppraiser#carlyst-streamlit-app)
    1. [Running it on local](https://github.com/caresppen/UsedCarsAppraiser#running-it-on-local)
    2. [Launching the online app](https://github.com/caresppen/UsedCarsAppraiser#launching-the-online-app)
    3. [Navigation](https://github.com/caresppen/UsedCarsAppraiser#navigation)
6. [Requirements](https://github.com/caresppen/UsedCarsAppraiser#requirements)

## Repository structure
It holds the following elements, which are summarized in this [directory tree](https://htmlpreview.github.io/?https://github.com/caresppen/UsedCarsAppraiser/blob/main/tree.html):
##### Folders
* [data](https://github.com/caresppen/UsedCarsAppraiser/tree/main/data): `csv` files generated along this project.
* [modules](https://github.com/caresppen/UsedCarsAppraiser/tree/main/modules): python scripts packaged to simplify tasks on notebooks.
* [notebooks](https://github.com/caresppen/UsedCarsAppraiser/tree/main/notebooks): jupyter notebooks that elaborate every detail for each machine learning process of this project. It contains saved figures of main generated plots and stored models using bz2.
* [st-front-end](https://github.com/caresppen/UsedCarsAppraiser/tree/main/st-front-end): Carlyst python application based on streamlit.
* [txt_guidelines](https://github.com/caresppen/UsedCarsAppraiser/tree/main/txt_guidelines): general `txt` scripts to produce web scraping, a directory tree and YAML file.

##### Main Files
* [Memory](https://github.com/caresppen/UsedCarsAppraiser/blob/main/MT_Carlyst_UsedCarsAppraisser.pdf): detailed master thesis which explains every step to create this ML solution.
* [Procfile](https://github.com/caresppen/UsedCarsAppraiser/blob/main/Procfile), [runtime.txt](https://github.com/caresppen/UsedCarsAppraiser/blob/main/runtime.txt), [setup.sh](https://github.com/caresppen/UsedCarsAppraiser/blob/main/setup.sh), [requirements.txt](https://github.com/caresppen/UsedCarsAppraiser/blob/main/requirements.txt): configuration files to load the app using heroku. Contains worker execution file, python version, setup and required packages.
* [project_pkgs.txt](https://github.com/caresppen/UsedCarsAppraiser/blob/main/project_pkgs.txt): Packages installed on the conda virtual environment. This configuration was also saved in a [YAML file](https://github.com/caresppen/UsedCarsAppraiser/blob/main/env-config.yml).

## Introduction to the problem
Nowadays, there is an uncertainty on the price that should be paid for a second-hand vehicle. There are several classified ad pages that promote car prices, but how many of them are you telling which ones have a fair price?

## Objective
The aim of this project is to provide a tool which can evaluate the value of used cars. The scope of this prediction is based on the Spanish second-hand car market. In this way, both sellers and buyers can find in this tool a solution to estimate the best and most fair price for each vehicle.

This golden price could be obtained by solving this question: how do the characteristics of the vehicle impact on the final price? Therefore, the ML model will be directed by the correlation between these technical specifications and the main target, the price.

The model, which is based on a ML CatBoost algorithm, was trained using a dataset of 55,326 real second-hand cars. It can predict prices of used cars up to 100,000€ in the Spanish market. These are the features that were implemented in the model:
| Categorical Features                                                                                                | Numerical Features                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| title<br/>brand<br/>model<br/>type<br/>city<br/>gearbox<br/>color<br/>fuel_type<br/>warranty<br/>dealer<br/>chassis | year<br/>kms<br/>doors<br/>seats<br/>power<br/>co2_emiss<br/>height<br/>length<br/>width<br/>trunk_vol<br/>max_speed<br/>mixed_cons<br/>weight<br/>tank_vol<br/>acc |

## Notebooks
### Data Extraction and Cleansing
![Data Extraction](https://drive.google.com/uc?export=view&id=10XQuVjsMu5eBQ1tXgKU0vluciTlL3tII)
* [01_used_cars_scraping.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/01_used_cars_scraping.ipynb): **Do not run this notebook** (can take up to 10 hours). Web scraping process to obtain second-hand cars data in tabular format. Produces a `csv` per car type. 
* [02_renting_cars_scraping.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/02_renting_cars_scraping.ipynb): **Do not run this notebook** (can take up to 1 hour). Future work. Web scraping process to extract renting cars data into a unique `csv`.
* [03_Data_Cleansing_cars.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/03_Data_Cleansing_cars.ipynb): Merge car type files, set the correct data types for each feature, translate every value from Spanish to English, standardization of each column structure and creation of new features to obtain better explanatory variables.
* [04_Data_Cleaning_renting.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/04_Data_Cleaning_renting.ipynb): Future work. Replicate the data cleansing process, but producing new features focused on renting market.

### Exploratory Data Analysis (EDA)
* [05_EDA_cars.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/05_EDA_cars.ipynb): Evaluate the dimension of each attribute, analyze the feasibility of data distribution and study the correlation between variables.

### Methodology: Feature Engineering and Data Modelling
* [06_MVP_ML.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/06_MVP_ML.ipynb): Minimum Viable Product (MVP). It reproduces a first iteration of the project, moving along these steps: import data, data visualization, data cleansing, data modelling and model evaluation.
* [07_Feature_Engineering.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/07_Feature_Engineering.ipynb): Preprocessing module in which categorical columns are transformed to numerical to be accepted by a ML model. Explores numerical features distribution to select the best column transformation process. 
* [08_Regression_wo_CT.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/08_Regression_wo_CT.ipynb): Evaluates the performance of 6 algorithms (LR, RD, RF, XGB, GB, CB) to conformn a model to predict used cars price. This step is based on data without a column transformation.
* [09_Column_Transformation.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/09_Column_Transformation.ipynb): Analyze each feature and apply Power Transformations (Box-Cox, Yeo-Johnson) and Column Transformations (Quantile Transformer) to obtain Gaussian distributions.
* [10_Regression_w_CT.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/10_Regression_w_CT.ipynb): Ne evaluation of the 6 models after applying column trasformations.

![Model Evaluation](https://drive.google.com/uc?export=view&id=16Jffb8FT4FUm1uemgV5jvOtg6Y-xRiuG)

#### Random Forest
* [11a_Hyperparameter_Tuning-RF.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/11a_Hyperparameter_Tuning-RF.ipynb): **Do not run this notebook** (can take up to 6 hours). Through applying a Random and Grid Search CV, finds the best parameters for the Random Forest Regressor. 
* [12a_Model_Builder-RF.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/12a_Model_Builder-RF.ipynb): Evaluation of the model, residual plot and model persistence process to it in a compressed pickle.

#### CatBoost
* [11b_Hyperparameter_Tuning-CB.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/11b_Hyperparameter_Tuning-CB.ipynb): **Do not run this notebook** (can take up to 2 hours). Through applying a Random and Grid Search CV, finds the best configuration for the CatBoost Regressor. This resulted to be the best performer and most optimal model.
* [12b_Model_Builder-CB.ipynb](https://github.com/caresppen/UsedCarsAppraiser/blob/main/notebooks/12b_Model_Builder-CB.ipynb): Final model evaluation using $$R^2 score$$ and SHAP values. Generates a plot with Actual vs Predicted values. Saves the model in a compressed `bz2` file to be used in the streamlit app.

![CatBoost Model](https://drive.google.com/uc?export=view&id=1XAj6hzbKqa8tD_lBsaH0jhLQy1hY4FLb)

_The image above explains the behaviour of the machine learning model. In example, it will consider that a low manufactured year, contributes to a lower final predicted car price._

## Carlyst: Streamlit app
Behind the scenes, a CatBoost ML model is running to provide a real-time price prediction of a second-hand car. The source code of this web based application is placed in the **st-front-end** folder: [carlyst_st_app.py](https://github.com/caresppen/UsedCarsAppraiser/blob/main/st-front-end/carlyst_st_app.py)

### Running it on local
Once on the [root folder](https://github.com/caresppen/UsedCarsAppraiser), execute from command line:
```python
streamlit run st-front-end/carlyst_st_app.py
```

### Launching the online app
Carlyst has been deployed using [heroku](https://www.heroku.com/platform). This platform provides a free worker to build and execute the application on demand (- [Build log example](https://codeshare.io/vwwE6n)).

[Click here](https://carlyst.herokuapp.com/) to launch the app.

### Navigation
Interact with this app by modifying the input parameters using sliders, text and selection boxes. Each configuration will provide a new price prediction:

![Carlyst Change Params](https://drive.google.com/uc?export=view&id=1bcU05LauWxRoXQSD9aLSRjeT01K9qbCB)

To get an automatic prediction, based on a specific second-hand car in the market, provide a url from [coches.com](https://www.coches.com/coches-segunda-mano/coches-ocasion.htm). When a link is input, savings and a recommendation about the puchase appear:

![Carlyst Link](https://drive.google.com/uc?export=view&id=1mrsl6NDxDj9D_nlZdPmUFgj9nK6ZwVxf)

After the link is read, parameters can be modified to predict a new price based on the new configuration.

## Requirements
1. Install [python 3.7.10](https://www.python.org/downloads/release/python-3710/).
2. Install [anaconda](https://docs.anaconda.com/anaconda/install/index.html) distribution on OS.
3. Clone this repository:
    ```shell
    git clone https://github.com/caresppen/UsedCarsAppraiser
    ```
4. Change `cwd` to the [root folder](https://github.com/caresppen/UsedCarsAppraiser) of this project.
    ```shell
    cd UsedCarsAppraiser/
    ```
5. Create, activate, and update an environment using the provided `yml` file: [env-config.yml](https://github.com/caresppen/UsedCarsAppraiser/blob/main/env-config.yml). For further information, consult [conda user guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). This will enable the environment to run every notebook without any constraints:
    ```python
    conda create --name myenv
    conda activate myenv
    conda env update --file env-config.yml --prune
    ```
6. Deployment to Heroku. Place these config files on the root folder: [Procfile](https://github.com/caresppen/UsedCarsAppraiser/blob/main/Procfile), [runtime.txt](https://github.com/caresppen/UsedCarsAppraiser/blob/main/runtime.txt), [setup.sh](https://github.com/caresppen/UsedCarsAppraiser/blob/main/setup.sh), [requirements.txt](https://github.com/caresppen/UsedCarsAppraiser/blob/main/requirements.txt)
7. `CB_model` uploaded to [**Releases**](https://github.com/caresppen/UsedCarsAppraiser/releases/tag/cb_model_v1). In case it is needed to be directly implemented to any project.