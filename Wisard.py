"""
Module that implements the WiSARD classifier.
"""

import random
import math
import copy
import time
import pickle


class Discriminator:
    """
    Represents a WiSARD discriminator, which contains a set of random access memories.      
    """


    def __init__(self, input_class, input_length, tuple_size, bleaching = False, type_mem_alloc = "dalloc"):
        """
        Constructor for the Discriminator class.

        :param input_class: a string that identifies the class being represented by the discriminator.
        :param input_length: a number (integer) of bits in an observation.
        :param tuple_size: a number (integer) that defines the address size to be used by the memories.
        :param bleaching: a boolean indicating if the bleaching functionality is active or not.
        :param type_mem_alloc: a string indicating what type of memory allocation shoud be used. The 
        accepted values are "dalloc" (for dynamically allocation) and "palloc" (for pre allocation).
        The consequences of that choice are that "dalloc" is expected to consume less memory and be 
        slower than "palloc", while "palloc" is expected to consume more memory and be faster. The 
        default value is "dalloc".

        """
        self.input_class = input_class
        self.input_length = input_length
        self.tuple_size = tuple_size
        self.ram_size = int(math.pow(2, self.tuple_size))
        self.number_of_rams = int(self.input_length / self.tuple_size)
        self.bleaching = bleaching
        self.memory = None
        self.type_mem_alloc = type_mem_alloc

        if type_mem_alloc == "dalloc":
            self.memory = {}
            self.write = self.write_dinamyc_alloc
            self.evaluate = self.evaluate_dinamyc_alloc
            self.generate_mental_image = self.generate_mental_image_dalloc
        elif type_mem_alloc == "palloc":
            self.memory = [0] * int(self.number_of_rams * self.ram_size) #np.zeros((int(self.number_of_rams * self.ram_size)))
            self.write = self.write_pre_alloc_array
            self.evaluate = self.evaluate_pre_alloc_array
            self.generate_mental_image = self.generate_mental_image_palloc
        else:
            raise Exception("\"type_mem_alloc\" only accepts the values \"dalloc\", for dynamic allocation and \"palloc\", for pre-allocation.")

    def write_dinamyc_alloc(self, addresses):
        """
        Writes a pattern to the RAMs.
        :param pattern: binary pattern to be learned.
        """
        
        for i in range(self.number_of_rams):
            if i not in self.memory:
                self.memory[i] = {}

            address = addresses[i]

            if address not in self.memory[i]:
                self.memory[i][address] = 1
            elif address in self.memory[i] and self.bleaching:
                self.memory[i][address] = self.memory[i][address] + 1


    def evaluate_dinamyc_alloc(self, addresses, bleaching_threshold = 0):
        """
        Evaluates a pattern and returns its score.
        :param pattern: pattern to be evaluated.
        :param bleaching_threshold: threshold to be used to solve draws.
        Returns:
            -> discriminator score.
        """

        score = 0

        for i in range(self.number_of_rams):
            address = addresses[i]

            if i in self.memory and address in self.memory[i]:
                if self.bleaching and self.memory[i][address] >= bleaching_threshold:
                    score = score + 1
                elif self.bleaching and self.memory[i][address] < bleaching_threshold:
                    continue
                else:
                    score = score + 1
                
        return score
            

    def write_pre_alloc_array(self, addresses):
        """
        Writes a pattern to the RAMs.
        :param pattern: binary pattern to be learned.
        """

        #pattern = self.get_observation_as_ints(pattern) 

        for i in range(self.number_of_rams): 
            ram_position = addresses[i]
            ram = i * self.ram_size + ram_position
            self.memory[ram] = self.memory[ram] + 1
        
            
    def evaluate_pre_alloc_array(self, addresses, bleaching_threshold = 0):
        """
        Evaluates a pattern and returns its score.
        :param pattern: pattern to be evaluated.
        :param bleaching_threshold: threshold to be used to solve draws.
        Returns:
            -> discriminator score.
        """
        #pattern = self.get_observation_as_ints(pattern)
        score = 0

        for i in range(self.number_of_rams):
            ram_position = addresses[i]
            position = i * self.ram_size + ram_position

            if self.bleaching:
                if self.memory[position] > bleaching_threshold:
                    score = score + 1
            else:
                if self.memory[position] > 0:
                    score = score + 1
                
        return score

    def generate_mental_image_dalloc(self):
        pixels = []

        for ram in sorted(self.memory):
            chunk = [0] * self.tuple_size

            for i in range(self.tuple_size):
                for address in self.memory[ram]:
                    if address[i] == "1" and self.memory[ram][address] > 0:
                        chunk[i] = chunk[i] + self.memory[ram][address]
                
            pixels = pixels + chunk

        return pixels

    def generate_mental_image_palloc(self):
        pixels = []

        for ram in range(self.number_of_rams):
            chunk = [0] * self.tuple_size

            for i in range(self.tuple_size):
                for address in range(self.ram_size):
                    bin_address = ("{0:0" + str(self.tuple_size) + "b}").format(address)
                    position = int(ram * self.ram_size + address)

                    if bin_address[i] == "1" and self.memory[position] > 0:
                        chunk[i] = chunk[i] + self.memory[position]
            
            pixels = pixels + chunk

        return pixels


class Wisard:
    """
    WiSARD "neural network". A weightless, RAM-based classifier.
    """

    def __init__(self, tuple_size = 2, bleaching = False, seed = 0, shuffle_observations = True, type_mem_alloc = "dalloc"):
        """
        Constructor for the WiSARD class.
        
        :param tuple_size: a number (integer) that defines the address size to be used by the memories.
        :param bleaching: a boolean indicating if the bleaching functionality is active or not. It defaults to False.
        :param seed: a integer to be used as seed for the random number generator, to allow reproducibility. It defaults to 0.
        :param shuffle_observations: a boolean to activate the shuffling of an observation, based on the param "seed". It 
        defaults to True.
        :param type_mem_alloc: a string indicating what type of memory allocation shoud be used. The 
        accepted values are "dalloc" (for dynamically allocation) and "palloc" (for pre allocation).
        The consequences of that choice are that "dalloc" is expected to consume less memory and be 
        slower than "palloc", while "palloc" is expected to consume more memory and be faster. The 
        default value is "dalloc".
        """

        self.seed = seed
        self.shuffle_observations = shuffle_observations
        self.bleaching = bleaching
        self.tuple_size = tuple_size
        self.observation_length = 0
        self.ram_size = math.pow(2, self.tuple_size)
        self.number_of_rams = 0 
        self.discriminators = {}
        self.type_mem_alloc = type_mem_alloc


    def random_mapping(self, observation):
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


    def train_bulk(self, observations_and_classes):
        """
        Trains the WiSARD classifier based on the provide inputs and its expected outputs. 

        :param observations_and_classes: a matrix of binary input sequences 
        (lists with zeros and ones as integers) and their expected outputs.
        """

        for observation_and_classe in observations_and_classes:
            self.train([observation_and_classe[0]], [observation_and_classe[1]])

    def train(self, observations, classes):
        """
        Trains the WiSARD classifier based on the provide inputs and its expected outputs. 
        :param observations: list of binary input sequences (lists with zeros and ones as integers).
        :param classes: list of expected outputs (preferrably as strings)
        """

        #check lists sizes
        if(len(observations) != len(classes)):
            raise Exception("Lengths of \"observations\" and \"classes\" must be equal.")

        transformed_observations = self.prepare_observations(observations, "train")

        for i in range(len(transformed_observations)):
            observation = transformed_observations[i]
            observation_class = classes[i]
            discriminator = None

            if observation_class not in self.discriminators:
                discriminator = Discriminator(observation_class, self.observation_length, 
                                    self.tuple_size, self.bleaching, self.type_mem_alloc)   
                self.discriminators[observation_class] = discriminator
            else:
                discriminator = self.discriminators[observation_class] 

            discriminator.write(observation)


    def predict(self, observations, detailed = False):
        """
        Evaluates an observation and returns its predicted class. 
        :param observation: binary input sequence (list with zeros and ones as integers).

        Returns: the class that returned the biggest discriminator response.
        """
        
        predictions = []
        transformed_observations = self.prepare_observations(observations, "predict")

        for observation in transformed_observations:                
            result_achieved = False
            predicted_classes = []
            last_scores = []
            confidence = 0             # can be optional
            current_threshold = 0 
            bleaching_actions = 0      # can be removed

            discriminators_to_evaluate = self.discriminators.keys() 
            previously_evaluated_discriminators = [] # can be removed
            
            while not result_achieved:
                predicted_classes = [{"discriminator": None, "score": 0}]

                for discriminator in sorted(discriminators_to_evaluate):
                    score = self.discriminators[discriminator].evaluate(observation, current_threshold)
                    last_scores.append(score)

                    if score > predicted_classes[0]["score"] or predicted_classes[0]["discriminator"] == None:
                        predicted_classes = [{"discriminator": self.discriminators[discriminator], "score": score}]
                    elif score == predicted_classes[0]["score"]:
                        predicted_classes.append({"discriminator": self.discriminators[discriminator], "score": score})

                exit_condition = None

                if not self.bleaching:
                    exit_condition = 1
                    result_achieved = True
                    confidence = self.calculate_confidence(last_scores)
                elif self.bleaching and len(predicted_classes) > 1:
                    exit_condition = 2
                    if predicted_classes[0]["score"] == 0:
                        result_achieved = True
                    else:
                        bleaching_actions = bleaching_actions + 1
                        current_threshold = current_threshold + 1

                        previously_evaluated_discriminators = discriminators_to_evaluate
                        last_scores = []
                        discriminators_to_evaluate = []
                        for predicted_class in predicted_classes:
                            discriminators_to_evaluate.append(predicted_class["discriminator"].input_class)

                elif self.bleaching and len(predicted_classes) == 1:
                    exit_condition = 3
                    result_achieved = True
                    confidence = self.calculate_confidence(last_scores)
                else:
                    raise Exception("Unable to reach valid stopping criteria.")
                    break

            # If the method ends with more than one class as possible, it just returns the first one.
            predictions.append({
                "class": predicted_classes[0]["discriminator"].input_class, 
                "score": predicted_classes[0]["score"], 
                "confidence": confidence,
                "bleaching_actions": bleaching_actions,
                "draw": True if len(predicted_classes) > 1 else False
            })

        if not detailed:
            self.simplify_predictions(predictions)
            
        return predictions

    
    def prepare_observations(self, observations, caller):
        transformed_observations = []

        for observation in observations:

            if caller == "train" and self.observation_length == 0:
                observation_length = len(observation)
                
                if ((observation_length % self.tuple_size) != 0):
                    raise Exception("Observation length MUST be multiple of tuple size.")

                self.observation_length = observation_length
                self.number_of_rams = int(self.observation_length / self.tuple_size)
            
            if len(observation) != self.observation_length:
                raise Exception("Observation length MUST be %s." % (str(self.observation_length)))

            observation = self.random_mapping(observation)

            if self.type_mem_alloc == "dalloc":
                observation = self.get_observation_as_bin_strings(observation)
            elif self.type_mem_alloc == "palloc":
                observation = self.get_observation_as_ints(observation)

            transformed_observations.append(observation)

        return transformed_observations


    def get_observation_as_bin_strings(self, observation):
        observation_as_bin_strings = []

        for i in range(self.number_of_rams):
            address = observation[i * self.tuple_size: (i * self.tuple_size) + self.tuple_size]
            address = "".join(str(k) for k in address) 
            observation_as_bin_strings.append(address)

        return observation_as_bin_strings

    def get_observation_as_ints(self, observation):
        observation_as_ints = []

        for i in range(self.number_of_rams): 
            observation_as_ints.append(
                self.get_address_as_int(
                    observation, 
                    i * self.tuple_size, (i * self.tuple_size) + self.tuple_size
                )
            )

        return observation_as_ints

    def get_address_as_int(self, pattern, start, end):
        address = pattern[start: end]
        address = int("".join(str(i) for i in address), 2) 
        return address


    def simplify_predictions(self, predictions):
        for i in range(len(predictions)):
            predictions[i] = predictions[i]["class"]


    def calculate_confidence(self, scores):
        ordered_scores = sorted(scores, reverse = True)
        if len(ordered_scores) < 2 or ordered_scores[0] == 0:
            return 0.0
        elif len(ordered_scores) >= 2:
            return (ordered_scores[0] - ordered_scores[1]) / ordered_scores[0]


    def deactivate_bleaching(self):
        """
        Dectivates bleaching, if that functionality is active. Does nothing otherwise.
        """
        self.bleaching = False

    def activate_bleaching(self):
        """
        Activates bleaching, if that functionality is deactivated. 
        It is only relevant if the model was trained with bleaching activated, 
        having that functionality deactivated only afterwards.
        """
        self.bleaching = True

    def get_mental_images(self, desired_class = None):
        """
        Generates a mental image for each discriminator. Each mental imagem is 
        a 1D list that can be converted to an array and reshaped to the desired
        dimensions. If "desired_class" is provided, only the mental image of the
        discriminator for the desired class will be generated.

        :param desired_class: class of a given discriminator.
        
        Returns: a dictionare of mental images indexed by discriminator's classes.
        """

        shuffled_mental_images = {}
        unshuffled_mental_images = {}

        for discriminator_class in self.discriminators:
            if desired_class != None and discriminator_class != desired_class:
                continue
            
            pixels = self.discriminators[discriminator_class].generate_mental_image()
            shuffled_mental_images[discriminator_class] = pixels

        original_positions = list(range(0,self.observation_length))

        if self.shuffle_observations:
            random.seed(self.seed)
            random.shuffle(original_positions)

        for mental_image_class in sorted(shuffled_mental_images):
            shuffled_mental_image = shuffled_mental_images[mental_image_class]
            unshuffled_mental_image = [0] * len(shuffled_mental_image)
            
            for i in range(len(original_positions)):
                unshuffled_mental_image[original_positions[i]] = shuffled_mental_image[i]

            unshuffled_mental_images[mental_image_class] = unshuffled_mental_image

        return unshuffled_mental_images

    def save(self, file_name):
        """
        Saves the object to disk.

        :param file: file URI.
        """
        
        output_file = open(file_name, "wb")
        pickle.dump(self, output_file)
        output_file.close()

    @classmethod
    def load(cls, file_name):
        """
        Loads object from disk.

        :param file: file URI.

        Returns: a Wisard object loaded from disk.
        """
        
        input_file = open(file_name, "rb")
        loaded_object = pickle.load(input_file)
        input_file.close()

        return loaded_object
        
