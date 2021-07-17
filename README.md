# Carlyst: Used Cars Appraiser
![Carlyst](https://drive.google.com/uc?export=view&id=1cf6kuyI7QMg5VDDLlsbfL-Yrpi-OF_ap)
Master Thesis. Master in Data Science at KSchool (2020-2021).

**Author**: [Carlos Espejo Peña](https://www.linkedin.com/in/carlosespejopena/)

This presentation act as a guidance for Carlyst project. It contains the structure of the repository, an overall concept of the objective of this project and a guide on how to use this [app](https://carlyst.herokuapp.com/).

## Repository structure
It holds the following elements, which are summarized in this [directory tree](https://htmlpreview.github.io/?https://github.com/caresppen/UsedCarsAppraiser/blob/main/tree.html):
##### Folders
* [data](https://github.com/caresppen/UsedCarsAppraiser/tree/main/data): `csv` files generated along this project.
* [modules](https://github.com/caresppen/UsedCarsAppraiser/tree/main/modules): python scripts packaged to simplify tasks on notebooks.
* [notebooks](https://github.com/caresppen/UsedCarsAppraiser/tree/main/notebooks): jupyter notebooks that elaborate every detail for each machine learning process of this project. It contains saved figures of main generated plots and stored models using `bz2`.
* [st-front-end](https://github.com/caresppen/UsedCarsAppraiser/tree/main/st-front-end): Carlyst python application based on `streamlit`.
* [txt_guidelines](https://github.com/caresppen/UsedCarsAppraiser/tree/main/txt_guidelines): general `txt` scripts to produce web scraping, a directory tree and `YAML` file.

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

## Link to the app
The app has been deployed using [heroku](https://www.heroku.com/platform). Follow the attached link to launch the app:
https://carlyst.herokuapp.com/
