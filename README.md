# Car Purchase Prediction using Neural Network

This repository contains code for predicting car purchase amount using a neural network model. The dataset used is 'Car_Purchasing_Data.csv'. The neural network is implemented using TensorFlow/Keras.

## Files

- `car_purchase_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and prediction.
- `Car_Purchasing_Data.csv`: CSV file containing the car purchasing data.
- `README.md`: This file.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow/Keras

## Usage

1. Make sure you have all the dependencies installed.
2. Clone or download the repository to your local machine.
3. Run the Jupyter Notebook `car_purchase_prediction.ipynb`.
4. Follow the instructions in the notebook to preprocess the data, train the neural network model, and make predictions.

## Description

- The notebook first loads the car purchasing data from the CSV file.
- It performs exploratory data analysis (EDA) using seaborn's pairplot to visualize the relationships between different features.
- Features such as 'Customer Name', 'Customer e-mail', 'Country' are dropped as they are not relevant for prediction.
- The target variable 'Car Purchase Amount' is separated from the input features.
- The input features are scaled using MinMaxScaler.
- A neural network model is defined using TensorFlow/Keras Sequential API.
- The model architecture consists of three dense layers with ReLU activation for hidden layers and linear activation for the output layer.
- The model is compiled with Adam optimizer and mean squared error loss function.
- The data is split into training and testing sets using train_test_split.
- The model is trained on the training data for a specified number of epochs.
- Training and validation losses are plotted to visualize the model's performance.
- Finally, the trained model is used to make predictions on new data.



