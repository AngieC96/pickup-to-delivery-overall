# pickup-to-delivery-overall

This repository contains the code for the pickup-to-delivery-overall (PDO) Data Science toy problem.

## Problem Statement

We will try to estimate the time it takes for a delivery to be completed, from the pick-up point to the delivery point (therefore, we will have $y = PD_time$).


We selected the following features to predict the delivery time:
- `country_code`: The country code for the order
- `city_code`: The city code for the order
- `order_id`: The unique identifier for the order
- `courier_id`: The unique identifier for the courier
- `activation_time`: The time the order was activated
- `transport`: The mode of transport used for the delivery
- `pickup_time`: The time the package was picked up
- `delivery_time`: The time it takes to deliver the package
- `pickup_latitude`: The latitude of the pickup location
- `pickup_longitude`: The longitude of the pickup location
- `delivery_latitude`: The latitude of the dropoff location
- `delivery_longitude`: The longitude of the dropoff location
- `delivery_entering_time`: The time the courier entered the delivery location
- `time_zone`: The time zone of the delivery location

We will compute the 
- `distance`: The distance between the pickup and dropoff locations

## Data

The data is retrieved from the database using a query.

## Deliverables

The deliverables for this project are: 
- A Jupyter notebook containing the code for the data cleaning, feature engineering, and model building

## Evaluation

The model will be evaluated using the Mean Absolute Error (MAE) metric.
We will develop a baseline model and compare the performance of the baseline model with the performance of the model we build.

The **baseline model** will be a heuristic model that predicts the delivery time per transport type as the distance divided by the speed of the transport type for all the orders.

## Model

As the first model, we will us e a **Linear Regression model** to predict the delivery time.
- Drop the rows with missing values, as we have enough values to work with
- Convert the categorical columns to numerical columns
- Split the data into training and testing sets
- Train a Linear Regression model on the training set
- Evaluate the model on the testing set
- Compute the Mean Absolute Error (MAE) of the model
- Compare the MAE of the model with the MAE of the baseline model
- Save the model

As the second model, we will use a **Random Forest Regressor** to predict the delivery time.
- Drop the rows with missing values, as we have enough values to work with
- Convert the categorical columns to numerical columns
- Split the data into training and testing sets
- Train a Random Forest Regressor on the training set
- Evaluate the model on the testing set
- Compute the Mean Absolute Error (MAE) of the model
- Compare the MAE of the model with the MAE of the linear regression model
- Save the model

As the third model, we will use a **Boosted Trees Regressor** to predict the delivery time.
- Drop the rows with missing values, as we have enough values to work with
- Convert the categorical columns to numerical columns
  - One-hot encode the categorical columns
  - Scale the numerical columns
  - Concatenate the one-hot encoded categorical columns with the scaled numerical columns
- Split the data into training and testing sets
- Train a Boosted trees model on the training set
- Evaluate the model on the testing set
- Compute the Mean Absolute Error (MAE) of the model
- Compare the MAE of the model with the MAE of the random forest model
- Save the model

## Future Work

In the future, we can try to improve the model by:
- Tuning the hyperparameters of the model
- Using a different feature engineering technique

