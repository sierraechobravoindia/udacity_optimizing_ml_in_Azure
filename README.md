# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The dataset provided contains  various properties of individuals like age, occupation etc. as features and a binary yes/no value  as the label that I assume is something like "individual becoming a customer" or "individual taking out a loan" or "individual buying a particular service or product". Hence the task at hand is a binary classification over multiple features. 



## Scikit-learn Pipeline
The pipeline ingests the data from a csv-file, cleans and one-hot encodes categorical features, then splits the data in a test and a training set. The estimator used is Logistic Regression with two parameters "C" the regularization strength and "max_iter" the maximum number of iterations.
I picked a RandomParameterSampling method with an early termination policy (BanditPolicy). I did not put in any effort to pick the parameter space to sample from or the specific BanditPolicy parameters as I was more interested in the engineering part than the machine learning part.

Random sampling was chosen because it offers in general a good performance with much less compute time when compared to a full grid search.

Early termination policies like the BanditPolicy chosen reduce compute without much risk of losing performance.

## AutoML
The AutoML pipeline not only optimizes over the hyperparamters of one given model but also tries many different estimators for the classification task. In this run, not surprisingly a Voting Ensemble scored best.

## Pipeline comparison
The accuracy of the hyperparameter optimization run w/ logistic regression was 0.9139 whereas the AutoML run achieved with 0.9182 accuracy. This comparison is not conclusive, as the parameters chosen for the Hyperdrive run were not optimized and the AutoML run was time limited to 30 Minutes.

## Future work
One obvious first step would be to increase the timeout time to give the Auto ML algorithm more time to explore different algorithms. 
For the Hyperdrive Search more effort could be placed on the parameter space and the BanditPolicy parameters.

## Proof of cluster clean up
The compute cluster is taken down in the Jupyter notebook via the .delete() method
