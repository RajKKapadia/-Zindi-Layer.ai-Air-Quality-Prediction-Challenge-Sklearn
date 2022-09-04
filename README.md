# Zindi - Layer.ai Air Quality Prediction Challenge
Hello, this is my attempt to [this](https://zindi.africa/competitions/layerai-air-quality-prediction-challenge) competition hosted by [Zindi](https://zindi.africa/).

I have used the following things here:

* cleaning the dataset
* training model
* generating submission

This strategy did not work well on the leadrborad but the take away here is that I learned a completely new technology.

## Installation
First create a new python virtual environment, activate that virtual environment and install the required packages

> pip install -r requirements.txt

## Run
You can run the `main.py` file, this will do the following things:

> python main.py

1. read and clean the dataset [prepare_dataset.py]
2. train ML model, for this make sure to have a look at `model_configuration.json`, this step will also save all the models, best model, preprocessing object, a `score.json` file [make sure to have a look at `configuration.py` file.]
3. generate submission file for each model and an ensembled submission file, the weights for the ensembled submission file is calculated from the `score.json` file.

# About me

I am `Raj Kapadia`, I am passionate about `AI/ML/DL` and their use in different domains, I also love to build `chatbots` using `Google Dialogflow ES/CX`, I have backend development experience with Python[Flask], and NodeJS[Express] For any work, you can reach out to me at...

* [LinkedIn](https://www.linkedin.com/in/rajkkapadia/)
* [Fiverr](https://www.fiverr.com/rajkkapadia​)
* [Upwork](https://www.upwork.com/freelancers/~0176aeacfcff7f1fc2)
* [Youtube](https://www.youtube.com/channel/UCOT01XvBSj12xQsANtTeAcQ)
