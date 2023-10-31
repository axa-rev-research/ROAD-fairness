#  On the Fairness ROAD: Robust Optimization for Adversarial Debiasing
Link of the arXiv paper: https://arxiv.org/abs/2310.18413

## Abstract:
In the field of algorithmic fairness, significant attention has been put on group fairness criteria, such as Demographic Parity and Equalized Odds. Nevertheless, these objectives, measured as global averages, have raised concerns about persistent local disparities between sensitive groups. In this work, we address the problem of local fairness, which ensures that the predictor is unbiased not only in terms of expectations over the whole population, but also within any subregion of the feature space, unknown at training time. To enforce this objective, we introduce ROAD, a novel approach that leverages the Distributionally Robust Optimization (DRO) framework within a fair adversarial learning objective, where an adversary tries to infer the sensitive attribute from the predictions. Using an instance-level re-weighting strategy, ROAD is designed to prioritize inputs that are likely to be locally unfair, i.e. where the adversary faces the least difficulty in reconstructing the sensitive attribute. Numerical experiments demonstrate the effectiveness of our method: it achieves Pareto dominance with respect to local fairness and accuracy for a given global fairness level across three standard datasets, and also enhances fairness generalization under distribution shift.


## How to run this code
Depending on the task, you need first to train models. For this purpose, you can run ROAD or BROAD for the local fairness objective using the following command:
- ### Compas
```bash
python3 compass.py 5 0.7 cuda:0 ROAD TEST_COMPAS_ROAD_10_200 20 10 10 10
```
- ### Law
```bash
python3 law.py 5 0.7 cuda:0 ROAD TEST_LAW_ROAD_10 20 10 10 10
```
- ### German
```bash
python3 german.py 5 0.7 cuda:0 ROAD TEST_GERMAN_ROAD_10 20 10 10 10
```
- ### Drift Experiment
For the adult drift experiment, you can run this command:
```bash
python3 adult_shifted_EO.py 5 0.7 cuda:0 ROAD_100 TEST_ADULT_ROAD_10_100_12_20 20 10 10 10
```
## Details of the hyperparameters
Here are listed the different arguments:

1) name of the selected dataset (either compass.py,  law.py, german.py or adult_shifted_EO.py)
2) maximum of the $\lambda$ hyperparameter (weight applied on the fairness component)
3) maximum of the $\tau$ hyperparameter (weight applied on the KL component)
4) name of the selected GPU (i.e., cuda:0, cuda:1, etc..)
5) maximum of the $\tau$ hyperparameter (weight applied on the KL component)
6) the desired method selected (i.e., ROAD or BROAD)
7) name of the output file
8) number of discretizations of lambda
9) number of discretizations of tau
10) number of training loops (nb_iter_s iterations) of the  training sensitive learning model $g_{w_g}$
11) number of training loops (nb_iter_r iterations) of the  training ratio $r_{w_r}$

## How to display the results
You can execute the jupyter notebook in the notebook folder: notebooks/plot_LAW.ipynb, notebooks/plot_DRO_COMPAS.ipynb, notebooks/plot_DRO_GERMAN.ipynb or plot_drifted.ipynb

