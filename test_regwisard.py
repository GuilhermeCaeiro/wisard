import pandas as pd
import numpy as np
import math
from Wisard import RegressionWisard
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import datasets, linear_model


def thermometer_encoding(value, thermometer_size, min_value, max_value):
    ones = math.ceil(value / ((max_value - min_value) / thermometer_size))
    #zeros = thermometer_size - ones
    if ones > thermometer_size:
        ones = thermometer_size
    elif ones < 0:
        ones = 0

    thermometer = [1 if bit < ones else 0 for bit in range(thermometer_size)] #"1" * ones + 0 * zeros

    return thermometer


## Simple regression task, using the dateset from
## https://www.kaggle.com/tanuprabhu/linear-regression-dataset
def test_1():
    dataset = pd.read_csv("sample_data/simplelinearregression.csv", sep=",")
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    training_set, test_set = train_test_split(dataset, test_size=0.3)

    tuple_sizes = [size for size in range(2,21)]
    thermometer_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Test 1.1
    ## References
    regressor = linear_model.LinearRegression()
    regressor.fit(dataset["X"].to_numpy().reshape(-1, 1), dataset["Y"].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(dataset["X"].to_numpy().reshape(-1, 1))
    print("LR MAE:", mean_absolute_error(predictions, dataset["Y"]), "LR RMSE:", np.sqrt(mean_squared_error(predictions, dataset["Y"])))
    
    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in dataset.iterrows():
                encoded_observations.append(thermometer_encoding(row["X"], thermometer_size, dataset["X"].min(), dataset["X"].max()))

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "mean", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, dataset["Y"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, dataset["Y"])
            error_rmse = np.sqrt(mean_squared_error(predictions, dataset["Y"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

            #break

    #return
    
    # Test 1.2
    ## References
    regressor = linear_model.LinearRegression()
    regressor.fit(training_set["X"].to_numpy().reshape(-1, 1), training_set["Y"].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(training_set["X"].to_numpy().reshape(-1, 1))
    print("LR Ein MAE:", mean_absolute_error(predictions, training_set["Y"]), "LR Ein RMSE:", np.sqrt(mean_squared_error(predictions, training_set["Y"])))
    predictions = regressor.predict(test_set["X"].to_numpy().reshape(-1, 1))
    print("LR Eout MAE:", mean_absolute_error(predictions, test_set["Y"]), "LR Eout RMSE:", np.sqrt(mean_squared_error(predictions, test_set["Y"])))

    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in training_set.iterrows():
                encoded_observations.append(thermometer_encoding(row["X"], thermometer_size, training_set["X"].min(), training_set["X"].max()))

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "mean", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, training_set["Y"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, training_set["Y"])
            error_rmse = np.sqrt(mean_squared_error(predictions, training_set["Y"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

            encoded_observations = []

            for index, row in test_set.iterrows():
                encoded_observations.append(thermometer_encoding(row["X"], thermometer_size, training_set["X"].min(), training_set["X"].max()))

            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, test_set["Y"])
            error_rmse = np.sqrt(mean_squared_error(predictions, test_set["Y"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "Out of sample (mae):", error_mae, "Out of sample (rmse):", error_rmse)


## Simple regression task, using the dateset from
## https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression
def test_2():
    pass

## Simple regression task, using the dateset from
## https://www.kaggle.com/quantbruce/real-estate-price-prediction
def test_3():
    pass



test_1()



















"""
dataset = pd.read_csv("sample_data/titanic_train.csv", sep=",")
dataset = dataset.sample(frac=1).reset_index(drop=True)
training_set, test_set = train_test_split(dataset, test_size=0.3)

# preprocessing
nan_age = dataset["Age"].mean()
training_set["Age"] = dataset["Age"].fillna(nan_age)
test_set["Age"] = dataset["Age"].fillna(nan_age)



min_max = {
    "age": {
        "min": math.floor(training_set["Age"].min()),
        "max": math.floor(training_set["Age"].max()),
    },
    "pclass": {
        "min": math.floor(training_set["Pclass"].min()),
        "max": math.floor(training_set["Pclass"].max()),
    },
    "sex": {
        "min": math.floor(training_set["Sex"].min()),
        "max": math.floor(training_set["Sex"].max()),
    },
    "siblings_spouse": {
        "min": math.floor(training_set["SibSp"].min()),
        "max": math.floor(training_set["SibSp"].max()),
    },
    "siblings_spouse": {
        "min": math.floor(training_set["SibSp"].min()),
        "max": math.floor(training_set["SibSp"].max()),
    },
    "parents": {
        "min": math.floor(training_set["Parch"].min()),
        "max": math.floor(training_set["Parch"].max()),
    },
    "fare": {
        "min": math.floor(training_set["Fare"].min()),
        "max": math.floor(training_set["Fare"].max()),
    },
}

# apply binarizaion to the dataset
def encode(dataset):
    encoded_observation = []

    for index, row in dataset.iterrows():
        age = thermometer_encoding(row["Age"])
        pclass = row["Pclass"]
        sex = row["Sex"]
        siblings_spouse = row["SibSp"]
        parents = row["Parch"]
        fare = row["Fare"]

        encoded_observation.append([age + pclass + sex + siblings_spouse + parents + fare])

    return encoded_observations

encoded_training_set_observations = encode(training_set)
encoded_test_set_observations = encode(test_set)

# train
wsd = Wisard(tuple_size = 2, mean_type = "mean", seed = 0, shuffle_observations = True, type_mem_alloc = "dalloc")
wsd.train(encoded_training_set_observations, training_set["Survived"].tolist())

# evaluate
"""