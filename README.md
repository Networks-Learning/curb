# Curb

This is a repository containing code for paper:

> J. kim, B. Tabibian, A. Oh, B. Schölkopf, M. Gomez-Rodriguez. Leveraging the Crowd to Detect and Reduce the Spread of Fake News and Misinformation. In Proceedings of the 11th ACM International Conference on Web Search and Data Mining (WSDM), 2018.

## Pre-requisites

This code is developed under python 3 and the following packages needs to be installed: `numpy`, `scipy`, `matplotlib`, `pickle`, `seaborn`

## Code structure

In the `curb` folder contains codes for the model (`Curb` and baseline methods), data generation and model execution. Also, the folder contains IPython notebook files for generating figures and twitter, weibo user exposure data.

 - `experiment_code` contains codes for model, data generation and model execution.
   - `generate_results.py` : Given the user exposure data (which uses exposure data in `Twitter` and `Weibo` directories) executes the models (`curb` and baseline methods) and saves the results in `pkl` files.
   - `curb.py` : Execution file for `Curb` and `Oracle`.
   - `flagratio.py` : Execution file for the `Flag Ratio` baseline.
   - `baseline.py` : Execution file for the `Exposure` baseline.
- `notebook` contains `Ipython` notebook files for generating figures given the file executions generated from from the files in the `experiment_code` folder.
- `twitter` and `weibo`
   - `exposure_data` contains user exposure data for each story, where exposures are generated from the user sharing raw files.
   - `results` contains the model execution results for the `Curb`, `Oracle`, as well as the baseline models.

## Raw data

We use the raw `Twitter` and `Weibo` data that provides users' networks and sharing logs, stories, and labels for the stories (whether the story contains misinformation or not). The reference for the data is:

> S. Kwon, M. Cha, and K Jung. 2017. Rumor detection over varying time windows. PLOS ONE 12, 1 (2017), e0168344.

and it can be downloaded from the following link:

> https://sites.google.com/site/iswgao/



