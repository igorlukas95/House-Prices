# House Price Prediction 

This project focuses on creating a predicting house price model using machine learning techniques.
The goal is to build a model that can predict the value of properties based on dataset features.
The project is designed to be in production, ensuring that the model is easily deployed, monitored 
and maintained in the real world.  

## Dataset
The dataset used in this project is available on Hugging Face:

https://huggingface.co/datasets/AnDrEw0203/advanced_house_prices



## Installation 
To start the projects, you need to have Python 3.12+ and also run the following scripts:
1. Clone git repository 
```
git clone https://github.com/KamilAndrzejewski-dev/House-Prices
```
2. Move to the cloned repository
```
cd .\House-Prices\
```
3. Install necessary packages
```
pip install -r requirements.txt
```

## Model Evaluation
Before prediction, you need to train the model running the following script:
```
python -m house_price_model.train_pipeline
```
After that, you can test model predictions:
```
python -m house_price_model.predict
```

## Testing
To test repository code, you need to run this commands:
```
python -m pytest
```




