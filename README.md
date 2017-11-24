# Curb

This is a repository containing code for the paper:

> J. kim, B. Tabibian, A. Oh, B. Schölkopf, M. Gomez-Rodriguez. Leveraging the Crowd to Detect and Reduce the Spread of Fake News and Misinformation. In Proceedings of the 11th ACM International Conference on Web Search and Data Mining (WSDM), 2018.

## Pre-requisites

This code is developed under Python 3 and the following packages are required for executing the code: `numpy`, `scipy`, `matplotlib`, `pickle`, `seaborn`

## Code structure

The repository contains code for the model (`Curb` and baseline methods) execution. Also, it contains Jupyter notebook files for generating figures in the paper and user exposure data for `Twitter`, `Weibo` datasets.

 - `code` directory contains code for executing the model and baselines.
   - `generate_results.py` : Given the user exposure data (which uses exposure data in `Twitter` and `Weibo` directories) executes the models (`Curb` and baseline methods) and saves the results in `pkl` files.
   - `curb.py` : API for `Curb` and `Oracle`.
   - `flagratio.py` : API for the `Flag Ratio` baseline.
   - `baseline.py` : API for the `Exposure` baseline.
- `notebook` contains `Jupyter` notebook files for generating figures in the paper, the notebooks use results generated from the scripts in the `code` directory.
- `twitter` and `weibo`
   - `exposure_data` contains user exposure data for each story, where exposures are generated from the user sharing raw files.
   - `results` contains already computed results for the `Curb`, `Oracle`, as well as the baseline models.

## Raw data

We use the raw `Twitter` and `Weibo` data that provides users' networks and sharing logs, stories, and labels for the stories (whether the story contains misinformation or not). The reference for the data is:

> S. Kwon, M. Cha, and K Jung. 2017. Rumor detection over varying time windows. PLOS ONE 12, 1 (2017), e0168344.

and it can be downloaded from the following link:

> https://sites.google.com/site/iswgao/



