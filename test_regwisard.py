import pandas as pd
import numpy as np
import math
import time
from Wisard import RegressionWisard
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
import bitstring


"""def thermometer_encoding(value, thermometer_size, min_value, max_value):
    ones = math.ceil(value / ((max_value - min_value) / thermometer_size))
    #zeros = thermometer_size - ones
    if ones > thermometer_size:
        ones = thermometer_size
    elif ones < 0:
        ones = 0

    thermometer = [1 if bit < ones else 0 for bit in range(thermometer_size)] #"1" * ones + 0 * zeros

    return thermometer"""


def encode_value(value, thermometer_size, min_value, max_value):
    value = float(value)
    min_value = float(min_value)
    max_value = float(max_value)
    #thermometer_size = training_experiment["binary_feature_size"]

    clamped_value = max(min_value, min(value, max_value))

    ones = int(math.floor(((clamped_value - min_value) / (max_value - min_value)) * thermometer_size))
    zeros = int(thermometer_size - ones)
    
    encoded_value = ([1] * ones) + ([0] * zeros)

    return encoded_value

def encode_value_non_completely_zeros(value, thermometer_size, min_value, max_value):
    value = float(value)
    min_value = float(min_value)
    max_value = float(max_value)

    clamped_value = max(min_value, min(value, max_value))

    ones = int(min(int(math.floor(((clamped_value - min_value) / (max_value - min_value)) * thermometer_size)) + 1, max_value))
    zeros = int(thermometer_size - ones)
    
    encoded_value = ([1] * ones) + ([0] * zeros)

    #print("-", zeros, ones, clamped_value)

    return encoded_value


def encode_value_circular_thermometer(value, thermometer_size, min_value, max_value):
    value = float(value)
    min_value = float(min_value)
    max_value = float(max_value)
    num_ones = int(thermometer_size / 2)

    clamped_value = max(min_value, min(value, max_value))

    ones = int(math.floor(((clamped_value - min_value) / (max_value - min_value)) * thermometer_size))
    zeros = int(thermometer_size - ones)

    starting_zeros = min(ones, thermometer_size - 1)
    expected_size = starting_zeros + num_ones
    remainder_ones = max(0, expected_size - thermometer_size)

    #encoded_value = ([1] * ones) + ([0] * zeros)
    #print(thermometer_size, expected_size, remainder_ones, starting_zeros)
    encoded_value = ([1] * remainder_ones) + ([0] * (starting_zeros - remainder_ones))  + ([1] * (num_ones - remainder_ones)) + ([0] * max(0, thermometer_size - expected_size))

    return encoded_value


def float_binary_encoding(value, length=32):
    binary_representation = bitstring.BitArray(float=value, length=length).bin
    return [int(bit) for bit in binary_representation]


## Simple regression task, using the dateset from
## https://www.kaggle.com/tanuprabhu/linear-regression-dataset
def test_1():
    dataset = pd.read_csv("sample_data/simplelinearregression.csv", sep=",")
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    training_set, test_set = train_test_split(dataset, test_size=0.3)
    print(len(training_set), len(test_set))

    tuple_sizes = [size for size in range(2,21)] +[50, 75, 100]
    thermometer_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Test 1.1
    ## References
    """regressor = linear_model.LinearRegression()
    regressor.fit(dataset["X"].to_numpy().reshape(-1, 1), dataset["Y"].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(dataset["X"].to_numpy().reshape(-1, 1))
    print("LR MAE:", mean_absolute_error(predictions, dataset["Y"]), "LR RMSE:", np.sqrt(mean_squared_error(predictions, dataset["Y"])))
    
    start_time = time.time()
    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in dataset.iterrows():
                encoded_observations.append(thermometer_encoding(row["X"], thermometer_size, dataset["X"].min(), dataset["X"].max()))

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "power_7", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, dataset["Y"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            #print(type(predictions))
            #print(len(predictions), list(predictions), len(dataset["Y"]), list(dataset["Y"]))
            
            error_mae = mean_absolute_error(predictions, dataset["Y"])
            error_rmse = np.sqrt(mean_squared_error(predictions, dataset["Y"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

            #break
        #break

    print("Time taken:", time.time() - start_time, "s.")"""

    #return
    
    # Test 1.2
    ## References
    regressor = linear_model.LinearRegression()
    regressor.fit(training_set["X"].to_numpy().reshape(-1, 1), training_set["Y"].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(training_set["X"].to_numpy().reshape(-1, 1))
    print("LR Ein MAE:", mean_absolute_error(predictions, training_set["Y"]), "LR Ein RMSE:", np.sqrt(mean_squared_error(predictions, training_set["Y"])))
    predictions = regressor.predict(test_set["X"].to_numpy().reshape(-1, 1))
    print("LR Eout MAE:", mean_absolute_error(predictions, test_set["Y"]), "LR Eout RMSE:", np.sqrt(mean_squared_error(predictions, test_set["Y"])))

    start_time = time.time()
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

    print("Time taken:", time.time() - start_time, "s.")


## Simple regression task, using the dateset from
## https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression
def test_2():
    dataset = pd.read_csv("sample_data/Salary_Data.csv", sep=",")
    dataset = dataset.sample(frac=1, random_state=3356).reset_index(drop=True)
    training_set, test_set = train_test_split(dataset, test_size=0.3, random_state=3356)

    tuple_sizes = [size for size in range(2,21)]
    thermometer_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Test 2.1
    ## References
    regressor = linear_model.LinearRegression()
    regressor.fit(dataset["YearsExperience"].to_numpy().reshape(-1, 1), dataset["Salary"].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(dataset["YearsExperience"].to_numpy().reshape(-1, 1))
    print("LR MAE:", mean_absolute_error(predictions, dataset["Salary"]), "LR RMSE:", np.sqrt(mean_squared_error(predictions, dataset["Salary"])))
    
    start_time = time.time()
    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in dataset.iterrows():
                encoded_observations.append(thermometer_encoding(row["YearsExperience"], thermometer_size, dataset["YearsExperience"].min(), dataset["YearsExperience"].max()))

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "mean", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, dataset["Salary"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, dataset["Salary"])
            error_rmse = np.sqrt(mean_squared_error(predictions, dataset["Salary"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

            #break

    print("Time taken:", time.time() - start_time, "s.")

    #return
    
    # Test 2.2
    ## References
    regressor = linear_model.LinearRegression()
    regressor.fit(training_set["YearsExperience"].to_numpy().reshape(-1, 1), training_set["Salary"].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(training_set["YearsExperience"].to_numpy().reshape(-1, 1))
    print("LR Ein MAE:", mean_absolute_error(predictions, training_set["Salary"]), "LR Ein RMSE:", np.sqrt(mean_squared_error(predictions, training_set["Salary"])))
    predictions = regressor.predict(test_set["YearsExperience"].to_numpy().reshape(-1, 1))
    print("LR Eout MAE:", mean_absolute_error(predictions, test_set["Salary"]), "LR Eout RMSE:", np.sqrt(mean_squared_error(predictions, test_set["Salary"])))

    start_time = time.time()
    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in training_set.iterrows():
                encoded_observations.append(thermometer_encoding(row["YearsExperience"], thermometer_size, training_set["YearsExperience"].min(), training_set["YearsExperience"].max()))

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "mean", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, training_set["Salary"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, training_set["Salary"])
            error_rmse = np.sqrt(mean_squared_error(predictions, training_set["Salary"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

            encoded_observations = []

            for index, row in test_set.iterrows():
                encoded_observations.append(thermometer_encoding(row["YearsExperience"], thermometer_size, training_set["YearsExperience"].min(), training_set["YearsExperience"].max()))

            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, test_set["Salary"])
            error_rmse = np.sqrt(mean_squared_error(predictions, test_set["Salary"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "Out of sample (mae):", error_mae, "Out of sample (rmse):", error_rmse)

    print("Time taken:", time.time() - start_time, "s.")

## Simple regression task, using the dateset from
## https://www.kaggle.com/quantbruce/real-estate-price-prediction
def test_3():
    dataset = pd.read_csv("sample_data/Real_estate.csv", sep=",")
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    training_set, test_set = train_test_split(dataset, test_size=0.3)

    tuple_sizes = [size for size in range(2,21)]
    thermometer_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    observation_features = [
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude",
    ]

    target = [
        "Y house price of unit area"
    ]

    # Test 1.1
    ## References
    regressor = linear_model.LinearRegression()
    regressor.fit(
        dataset[observation_features],#.to_numpy().reshape(-1, 1), 
        dataset[target].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(dataset[observation_features])#.to_numpy().reshape(-1, 1))
    print("LR MAE:", mean_absolute_error(predictions, dataset[target]), "LR RMSE:", np.sqrt(mean_squared_error(predictions, dataset[target])))

    """
    start_time = time.time()
    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in dataset.iterrows():
                trasaction_date = thermometer_encoding(row["X1 transaction date"], thermometer_size, dataset["X1 transaction date"].min(), dataset["X1 transaction date"].max())
                house_age = thermometer_encoding(row["X2 house age"], thermometer_size, dataset["X2 house age"].min(), dataset["X2 house age"].max())
                distance_mrt_station = thermometer_encoding(row["X3 distance to the nearest MRT station"], thermometer_size, dataset["X3 distance to the nearest MRT station"].min(), dataset["X3 distance to the nearest MRT station"].max())
                num_conv_stores = thermometer_encoding(row["X4 number of convenience stores"], thermometer_size, dataset["X4 number of convenience stores"].min(), dataset["X4 number of convenience stores"].max())
                latitute = thermometer_encoding(row["X5 latitude"], thermometer_size, dataset["X5 latitude"].min(), dataset["X5 latitude"].max())
                longitute = thermometer_encoding(row["X6 longitude"], thermometer_size, dataset["X6 longitude"].min(), dataset["X6 longitude"].max())
                #print("-", trasaction_date, house_age, distance_mrt_station, num_conv_stores, latitute, longitute)
                #print("+", trasaction_date + house_age + distance_mrt_station + num_conv_stores + latitute + longitute)
                encoded_observations.append(trasaction_date + house_age + distance_mrt_station + num_conv_stores + latitute + longitute)

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "mean", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, dataset["Y house price of unit area"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, dataset[target])
            error_rmse = np.sqrt(mean_squared_error(predictions, dataset[target]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

    print("Time taken:", time.time() - start_time, "s.")
    """


    regressor = linear_model.LinearRegression()
    regressor.fit(training_set[observation_features], training_set[target].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(training_set[observation_features])
    print("LR Ein MAE:", mean_absolute_error(predictions, training_set[target]), "LR Ein RMSE:", np.sqrt(mean_squared_error(predictions, training_set[target])))
    predictions = regressor.predict(test_set[observation_features])
    print("LR Eout MAE:", mean_absolute_error(predictions, test_set[target]), "LR Eout RMSE:", np.sqrt(mean_squared_error(predictions, test_set[target])))


    start_time = time.time()
    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in training_set.iterrows():
                trasaction_date = thermometer_encoding(row["X1 transaction date"], thermometer_size, training_set["X1 transaction date"].min(), training_set["X1 transaction date"].max())
                house_age = thermometer_encoding(row["X2 house age"], thermometer_size, training_set["X2 house age"].min(), training_set["X2 house age"].max())
                distance_mrt_station = thermometer_encoding(row["X3 distance to the nearest MRT station"], thermometer_size, training_set["X3 distance to the nearest MRT station"].min(), training_set["X3 distance to the nearest MRT station"].max())
                num_conv_stores = thermometer_encoding(row["X4 number of convenience stores"], thermometer_size, training_set["X4 number of convenience stores"].min(), training_set["X4 number of convenience stores"].max())
                latitute = thermometer_encoding(row["X5 latitude"], thermometer_size, training_set["X5 latitude"].min(), training_set["X5 latitude"].max())
                longitute = thermometer_encoding(row["X6 longitude"], thermometer_size, training_set["X6 longitude"].min(), training_set["X6 longitude"].max())

                encoded_observations.append(trasaction_date + house_age + distance_mrt_station + num_conv_stores + latitute + longitute)

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "mean", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, training_set["Y house price of unit area"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, training_set[target])
            error_rmse = np.sqrt(mean_squared_error(predictions, training_set[target]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

            encoded_observations = []

            for index, row in test_set.iterrows():
                trasaction_date = thermometer_encoding(row["X1 transaction date"], thermometer_size, training_set["X1 transaction date"].min(), training_set["X1 transaction date"].max())
                house_age = thermometer_encoding(row["X2 house age"], thermometer_size, training_set["X2 house age"].min(), training_set["X2 house age"].max())
                distance_mrt_station = thermometer_encoding(row["X3 distance to the nearest MRT station"], thermometer_size, training_set["X3 distance to the nearest MRT station"].min(), training_set["X3 distance to the nearest MRT station"].max())
                num_conv_stores = thermometer_encoding(row["X4 number of convenience stores"], thermometer_size, training_set["X4 number of convenience stores"].min(), training_set["X4 number of convenience stores"].max())
                latitute = thermometer_encoding(row["X5 latitude"], thermometer_size, training_set["X5 latitude"].min(), training_set["X5 latitude"].max())
                longitute = thermometer_encoding(row["X6 longitude"], thermometer_size, training_set["X6 longitude"].min(), training_set["X6 longitude"].max())

                encoded_observations.append(trasaction_date + house_age + distance_mrt_station + num_conv_stores + latitute + longitute)

            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, test_set[target])
            error_rmse = np.sqrt(mean_squared_error(predictions, test_set[target]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "Out of sample (mae):", error_mae, "Out of sample (rmse):", error_rmse)

    print("Time taken:", time.time() - start_time, "s.")



def test_4():
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

    scaler = MinMaxScaler()
    scaler.fit(dataset["X"].to_numpy().reshape(-1, 1))
    
    start_time = time.time()
    for tuple_size in tuple_sizes:
        for thermometer_size in thermometer_sizes:
            encoded_observations = []
            
            for index, row in dataset.iterrows():
                #print(float_binary_encoding(10))
                #print(type(float_binary_encoding(10)))
                #return 
                encoded_observations.append(float_binary_encoding(scaler.transform([[row["X"]]]), 32))

            #print(encoded_observations)
            wisard = RegressionWisard(tuple_size = tuple_size, mean_type = "mean", seed = 3356, shuffle_observations = True, type_mem_alloc = "dalloc")
            wisard.train(encoded_observations, dataset["Y"].tolist())
            
            predictions = wisard.predict(encoded_observations)
            
            error_mae = mean_absolute_error(predictions, dataset["Y"])
            error_rmse = np.sqrt(mean_squared_error(predictions, dataset["Y"]))

            print("Tuple size:", tuple_size, "Thermometer size:", thermometer_size, "In-sample (mae):", error_mae, "In-sample (rmse):", error_rmse)

            #break

    print("Time taken:", time.time() - start_time, "s.")

    return
    
    # Test 1.2
    ## References
    regressor = linear_model.LinearRegression()
    regressor.fit(training_set["X"].to_numpy().reshape(-1, 1), training_set["Y"].to_numpy().reshape(-1, 1))
    predictions = regressor.predict(training_set["X"].to_numpy().reshape(-1, 1))
    print("LR Ein MAE:", mean_absolute_error(predictions, training_set["Y"]), "LR Ein RMSE:", np.sqrt(mean_squared_error(predictions, training_set["Y"])))
    predictions = regressor.predict(test_set["X"].to_numpy().reshape(-1, 1))
    print("LR Eout MAE:", mean_absolute_error(predictions, test_set["Y"]), "LR Eout RMSE:", np.sqrt(mean_squared_error(predictions, test_set["Y"])))

    start_time = time.time()
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

    print("Time taken:", time.time() - start_time, "s.")


thermometer_encoding = encode_value

test_1()

print("\n\n\n\n\n\n\n\n\n")

#test_2()

print("\n\n\n\n\n\n\n\n\n")

#test_3()

#test_4()
















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