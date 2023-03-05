# summarization_aws

Example of using aws sagemaker studio to finetune BART on financial data in text summarization task.
Model was trained using huggingface framework

## Usage

Repo contains 2 files: summarization.ipynb and train.py

- summarization.ipynb is notebook which is starting sagemaker training jobs.

- train.py is script of an actual preprocessing and model finetuning part.

## Model and dataset

The main objective of this module is to fine-tune and evaluate a model on financial data. 
Finetuning is improving the performance of the model on domain specific tasks (financial data in our example). 
The pre-trained models can be finetuned on a number of downstream tasks based on their architecture.
We are using BART model in large configuration finetuned on CNN-Daily-Mail.
Our financial data is the Trade the Event dataset.
It contains more than 300000 articles from 2020 and 2021 year.
