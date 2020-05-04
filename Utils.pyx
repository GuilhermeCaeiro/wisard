import random
import math
import copy

cdef class DataPreprocessor:

    cdef public:
        cdef int seed, tuple_size, observation_length, ram_size, number_of_rams
        cdef bint shuffle_observations

    def __init__(self, int tuple_size, int ram_size, bint shuffle_observations, int seed):
        self.seed = seed
        self.tuple_size = tuple_size
        self.observation_length = 0
        self.ram_size = ram_size
        self.shuffle_observations = shuffle_observations
        self.number_of_rams = 0 


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
            observation_length = len(observation)

            if caller == "train" and self.observation_length == 0:
                if ((observation_length % self.tuple_size) != 0):
                    raise Exception("Observation length MUST be multiple of tuple size.")

                self.observation_length = observation_length
                self.number_of_rams = self.observation_length / self.tuple_size
            
            if observation_length != self.observation_length:
                raise Exception("Observation length MUST be %s." % (str(self.observation_length)))

            observation = self.random_mapping(observation)

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
