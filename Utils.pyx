import random
import math
import copy
import numpy as np
import bitstring

cdef class DataPreprocessor:

    cdef public:
        cdef int seed, tuple_size, observation_length, number_of_rams, additional_zeros
        cdef bint shuffle_observations

    def __init__(self, int tuple_size, double ram_size, bint shuffle_observations, int seed):
        self.seed = seed
        self.tuple_size = tuple_size
        self.observation_length = 0
        #self.ram_size = ram_size # unused, and received as double to avoid problems with the size of the number received
        self.shuffle_observations = shuffle_observations
        self.number_of_rams = 0 
        self.additional_zeros = 0


    def random_mapping(self, list observation):
        """
        Prepares the observation by appling to it the random mapping of the bits in the input sequence, 
        based on the seed provided during the class creation. 

        :param observation: list of binary values (lists with zeros and ones as integers).
        
        Returns: 
            -> A shuffled version of "observation" if "self.shuffle_observations" is 
            True, or the unmodified "observation" otherwise.
        """

        if self.shuffle_observations:
            observation = copy.deepcopy(observation)
            
            random.seed(self.seed)
            random.shuffle(observation)

        return observation


    def prepare_observations(self, list observations, str caller):
        cdef list transformed_observations
        cdef int observation_length
        
        transformed_observations = []

        for observation in observations:

            if caller == "train" and self.observation_length == 0:
                #if ((observation_length % self.tuple_size) != 0):
                #    raise Exception("Observation length MUST be multiple of tuple size.")
                observation_length = len(observation)

                if ((observation_length % self.tuple_size) != 0):
                    self.additional_zeros = ((observation_length // self.tuple_size) + 1) * self.tuple_size - observation_length
                    print("Adding %s zeros." % (str(self.additional_zeros)))

                self.observation_length = observation_length + self.additional_zeros
                self.number_of_rams = self.observation_length / self.tuple_size
            
            observation = self.random_mapping(observation + [0 for i in range(self.additional_zeros)])
            observation_length = len(observation)

            if observation_length != self.observation_length:
                raise Exception("Observation length MUST be %s." % (str(self.observation_length)))

            #if self.type_mem_alloc == "dalloc":
            #    #observation = self.get_observation_as_bin_strings(observation)
            #    observation = self.get_observation_as_ints(observation) # string consumes way, waaay too much memory here.
            #elif self.type_mem_alloc == "palloc":
            #    observation = self.get_observation_as_ints(observation)

            observation = self.get_observation_as_ints(observation)
            transformed_observations.append(observation)

        return transformed_observations


    # eats a lot of memory
    def get_observation_as_bin_strings(self, observation):
        observation_as_bin_strings = []

        for i in range(self.number_of_rams):
            address = observation[i * self.tuple_size: (i * self.tuple_size) + self.tuple_size]
            address = "".join(str(k) for k in address) 
            observation_as_bin_strings.append(address)

        return observation_as_bin_strings


    def get_observation_as_ints(self, list observation):
        observation_as_ints = []

        for i in range(self.number_of_rams): 
            observation_as_ints.append(
                self.get_address_as_int(
                    observation, 
                    i * self.tuple_size, (i * self.tuple_size) + self.tuple_size
                )
            )

        return observation_as_ints


    def get_address_as_int(self, list pattern, int start, int end):
        cdef list address
        cdef int int_address

        address = pattern[start: end]
        int_address = int("".join(str(i) for i in address), 2) 
        return int_address



class Mean:

    @classmethod
    def mean(self, counters, partial_ys):
        #print(counters, partial_ys)
        #exit(1)
        #print(len(counters), len(partial_ys), sum(partial_ys))
        return sum(partial_ys) / sum(counters)


    @classmethod
    def median(self, counters, partial_ys):
        return np.median(np.array(partial_ys) / np.array(counters))


    @classmethod
    def harmonic(self, counters, partial_ys):
        return self.power(counters, partial_ys, -1.0)
        #return np.power((np.sum(np.power(partial_ys), -1.0) / float(len(partial_ys))), -1.0)


    # https://undergroundmathematics.org/glossary/power-mean
    @classmethod
    def power(self, counters, partial_ys, power_value):
        partial_ys = np.asarray(partial_ys)
        counters = np.asarray(counters)
        power_value = float(power_value)
        return np.power((sum(np.power(partial_ys / counters, power_value)) / len(partial_ys)), 1.0 / power_value)
        #return np.power((np.sum(np.power(np.array(partial_ys), np.array(counters, dtype="float32"))) / float(len(partial_ys))), np.sum(counters))

    @classmethod
    def harmonic_power(self, counters, partial_ys):
        return float(len(partial_ys)) / np.sum(np.array(counters) / np.array(partial_ys))


    @classmethod
    def geometric(self, counters, partial_ys):
        return np.power(np.prod(np.array(partial_ys) / np.array(counters)), 1.0 / len(partial_ys))


    @classmethod
    def exponential(self, counters, partial_ys):
        return np.log(np.sum(np.exp(np.array(partial_ys) / np.array(counters))) / float(len(partial_ys)))



class Binarizer:

    @classmethod
    def thermometer(value, thermometer_size, min_value, max_value):
        return thermometer_v1(value, thermometer_size, min_value, max_value)

    @classmethod
    def thermometer_v1(value, thermometer_size, min_value, max_value):
        value = float(value)
        min_value = float(min_value)
        max_value = float(max_value)
        #thermometer_size = training_experiment["binary_feature_size"]

        clamped_value = max(min_value, min(value, max_value))

        ones = int(math.floor(((clamped_value - min_value) / (max_value - min_value)) * thermometer_size))
        zeros = int(thermometer_size - ones)
        
        encoded_value = ([1] * ones) + ([0] * zeros)

        return encoded_value

    @classmethod
    def thermometer_v2(value, thermometer_size, min_value, max_value):
        value = float(value)
        min_value = float(min_value)
        max_value = float(max_value)

        clamped_value = max(min_value, min(value, max_value))

        ones = int(min(int(math.floor(((clamped_value - min_value) / (max_value - min_value)) * thermometer_size)) + 1, max_value))
        zeros = int(thermometer_size - ones)
        
        encoded_value = ([1] * ones) + ([0] * zeros)

        #print("-", zeros, ones, clamped_value)

        return encoded_value

    @classmethod
    def circular_thermometer(value, thermometer_size, min_value, max_value):
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

    @classmethod
    def float_binary(value, length=32):
        binary_representation = bitstring.BitArray(float=value, length=length).bin
        return [int(bit) for bit in binary_representation]