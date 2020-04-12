"""
Module that implements the WiSARD classifier.
"""

import random
import math
import copy

class Discriminator:
    """
    Represents a WiSARD discriminator, which encapsulates a set of random access memories.      
    """


    def __init__(self, input_class, input_length, tuple_size, bleaching = False):
        """
        Constructor for the Discriminator class.
        :param input_class: string that identifies the class being represented by the discriminator.
        :param input_length: number (integer) of bits in a WiSARD input.
        :param tuple_size: number (integer) that defines the address size to be used by the memories.
        :param bleaching: boolean indicating if the bleaching functionality is active or not.
        """
        self.input_class = input_class
        self.input_length = input_length
        self.tuple_size = tuple_size
        self.ram_size = math.pow(2, self.tuple_size) # Unused
        self.number_of_rams = int(self.input_length / self.tuple_size)
        self.bleaching = bleaching
        self.memory = {} # This can be a list
    
    def write(self, pattern):
        """
        Writes a pattern to the RAMs.
        :param pattern: binary pattern to be learned.
        """

        for i in range(self.number_of_rams):
            if i not in self.memory:
                self.memory[i] = {}
            address = []

            for j in range(i * self.tuple_size, (i * self.tuple_size) + self.tuple_size):
                 address.append(pattern[j])

            address = "".join(str(i) for i in address) # address becomes a string

            if address not in self.memory[i]:
                self.memory[i][address] = 1
            elif address in self.memory[i] and self.bleaching:
                self.memory[i][address] = self.memory[i][address] + 1
            
    def evaluate(self, pattern, bleaching_threshold = 0):
        """
        Evaluates a pattern and returns its score.
        :param pattern: pattern to be evaluated.
        :param bleaching_threshold: threshold to be used to solve draws.
        Returns:
            -> discriminator score.
        """

        score = 0
        evaluation_pattern = ""

        for i in range(self.number_of_rams):
            address = []

            for j in range(i * self.tuple_size, (i * self.tuple_size) + self.tuple_size):
                 address.append(pattern[j])

            address = "".join(str(i) for i in address) # address becomes a string

            if i in self.memory and address in self.memory[i]:
                if self.bleaching and self.memory[i][address] >= bleaching_threshold:
                    score = score + 1
                    evaluation_pattern = evaluation_pattern + "1"
                elif self.bleaching and self.memory[i][address] < bleaching_threshold:
                    evaluation_pattern = evaluation_pattern + "2"
                    continue
                else:
                    evaluation_pattern = evaluation_pattern + "3"
                    score = score + 1
            else:
                evaluation_pattern = evaluation_pattern + "4"
                
        return score, evaluation_pattern, address

class Wisard:
    """
    WiSARD "neural network". A weightless, RAM-based classifier.
    """

    def __init__(self, tuple_size = 2, bleaching = False, seed = 0, shuffle_observations = True):
        """
        Constructor for the WiSARD class.
        :param tuple_size: number (integer) that defines the address size to be used by the memories.
        :param bleaching: boolean indicating if the bleaching functionality is active or not.
        :param seed: integer to be used as seed in the random number generator, to allow reproducibility.
        :param shuffle_observations: boolean to activate the shuffling of an observation, based on the param "seed".
        """

        self.seed = seed
        self.shuffle_observations = shuffle_observations
        self.bleaching = bleaching
        self.tuple_size = tuple_size
        self.discriminators = {}


    def random_mapping(self, observation):
        """
        Prepares the observation by appling to it the random mapping of the bits in the input sequence, based on the seed provided during the class creation. 
        :param observation: list of binary values (lists with zeros and ones as integers).
        Returns: 
            -> A shuffled version of "observation"
        """

        # shuffles observation
        observ_copy = copy.deepcopy(observation)
        random.seed(self.seed)
        random.shuffle(observ_copy)

        return observ_copy


    def train_bulk(self, observations_and_classes):
        """
        Trains the WiSARD classifier based on the provide inputs and its expected outputs. 
        :param observations_and_classes: list of binary input sequences (lists with zeros and ones as integers) and expected outputs.
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

        for i in range(len(observations)):
            observation = observations[i]
            observation_class = classes[i]
            discriminator = None
            
            if self.shuffle_observations:
                observation = self.random_mapping(observation)

            if observation_class not in self.discriminators:
                observation_length = len(observation)
                discriminator = Discriminator(observation_class, observation_length, 
                                    self.tuple_size, self.bleaching)   
                self.discriminators[observation_class] = discriminator
            else:
                discriminator = self.discriminators[observation_class]

            discriminator.write(observation)


    def predict(self, observation):
        """
        Evaluates an observation and returns its predicted class. 
        :param observation: binary input sequence (list with zeros and ones as integers).
        Returns: the class that returned the biggest discriminator response.
        """

        if self.shuffle_observations:
            observation = self.random_mapping(observation)

        result_achieved = False

        predicted_classes = []
        current_threshold = 0

        discriminators_to_evaluate = self.discriminators.keys()
        previously_evaluated_discriminators = []

        while not result_achieved:
            predicted_classes = [{"discriminator": None, "score": 0}]

            for discriminator in sorted(discriminators_to_evaluate):
                score, evaluation_pattern, address = self.discriminators[discriminator].evaluate(observation, current_threshold)
                last_score = score

                if score > predicted_classes[0]["score"] or predicted_classes[0]["discriminator"] == None:
                    predicted_classes = [{"discriminator": self.discriminators[discriminator], "score": score}]
                elif score == predicted_classes[0]["score"]:
                    predicted_classes.append({"discriminator": self.discriminators[discriminator], "score": score})

            exit_condition = None

            if not self.bleaching:
                exit_condition = 1
                result_achieved = True
            elif self.bleaching and len(predicted_classes) > 1:
                exit_condition = 2
                if predicted_classes[0]["score"] == 0:
                    result_achieved = True
                else:
                    current_threshold = current_threshold + 1

                    previously_evaluated_discriminators = discriminators_to_evaluate
                    discriminators_to_evaluate = []
                    for predicted_class in predicted_classes:
                        discriminators_to_evaluate.append(predicted_class["discriminator"].input_class)

            elif self.bleaching and len(predicted_classes) == 1:
                exit_condition = 3
                result_achieved = True
            else:
                print("Error predicting class.")
                break

        # If the method ends with more than one class as possible, it just returns the first one.
        # TODO: Change the following line to return a random class if there is still a draw between
        # two or more classes.
        return {"class": predicted_classes[0]["discriminator"].input_class, "score": predicted_classes[0]["score"]}, True if len(predicted_classes) > 1 else False

    def deactivate_bleaching(self):
        """
        Dectivates bleaching, if that functionality is active. Does nothing otherwise.
        """
        self.bleaching = False

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
        input_length = 0


        for discriminator_class in self.discriminators:
            if desired_class != None and discriminator_class != desired_class:
                continue

            rams = self.discriminators[discriminator_class].memory
            # not a problem if called multiple time, because all discriminators 
            # should have the same value for input_length
            input_length = self.discriminators[discriminator_class].input_length 
            pixels = []

            for ram in sorted(rams):
                num_bits = len(list(rams[ram])[0]) #self.tuple_size
                chunk = [0] * num_bits

                for i in range(num_bits):
                    for address in rams[ram]:
                        #if i >= len(address):
                        #    continue

                        if address[i] == "1" and rams[ram][address] > 0:
                            chunk[i] = chunk[i] + rams[ram][address]
                    
                pixels = pixels + chunk

            shuffled_mental_images[discriminator_class] = pixels

        original_positions = list(range(0,input_length))
        random.seed(self.seed)
        random.shuffle(original_positions)

        for mental_image_class in sorted(shuffled_mental_images):
            shuffled_mental_image = shuffled_mental_images[mental_image_class]
            unshuffled_mental_image = [0] * len(shuffled_mental_image)
            
            for i in range(len(original_positions)):
                unshuffled_mental_image[original_positions[i]] = shuffled_mental_image[i]

            unshuffled_mental_images[mental_image_class] = unshuffled_mental_image

        return unshuffled_mental_images#, shuffled_mental_images

    def save_network_to_disk(self):
        """
        Not implemented yet.
        """
        pass

    def load_network_from_disk(self):
        """
        Not implemented yet.
        """
        pass

        
