"""
Module that implements the WiSARD classifier.
"""

import pyximport; pyximport.install()
from Discriminator import DiscriminatorWisard, DiscriminatorRegressionWisard 
from Utils import DataPreprocessor, Mean
import random
import math
import copy
import time
import pickle
import multiprocessing


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
        
        self.data_preprocessor = DataPreprocessor(
            self.tuple_size, 
            self.ram_size, 
            self.shuffle_observations,
            self.seed
        )


    """
    def random_mapping(self, observation):
        "
        Prepares the observation by appling to it the random mapping of the bits in the input sequence, 
        based on the seed provided during the class creation. 

        :param observation: list of binary values (lists with zeros and ones as integers).
        
        Returns: 
            -> A shuffled version of "observation" if "self.shuffle_observations" is 
            True, or the unmodified "observation" otherwise.
        "

        if self.shuffle_observations:
            observation = copy.deepcopy(observation)
            
            random.seed(self.seed)
            random.shuffle(observation)

        return observation
    """


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

        transformed_observations = self.data_preprocessor.prepare_observations(observations, "train")

        if self.observation_length == 0:
            self.observation_length = self.data_preprocessor.observation_length
            self.number_of_rams = self.data_preprocessor.number_of_rams
        
        for i in range(len(transformed_observations)):
            observation = transformed_observations[i]
            observation_class = classes[i]
            discriminator = None

            if observation_class not in self.discriminators:
                discriminator = DiscriminatorWisard(observation_class, self.observation_length, 
                                    self.tuple_size, self.bleaching, self.type_mem_alloc)   
                self.discriminators[observation_class] = discriminator
            else:
                discriminator = self.discriminators[observation_class] 

            discriminator.write(observation)


    def predict_single_proc(self, observations, detailed = False):
        """
        Evaluates an observation and returns its predicted class. 
        :param observation: binary input sequence (list with zeros and ones as integers).

        Returns: the class that returned the biggest discriminator response.
        """
        
        predictions = []
        start_time = time.time()
        transformed_observations = self.data_preprocessor.prepare_observations(observations, "predict")
        print("Time taken to prepare observations:", time.time() - start_time)

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

    def predict_multi_proc(self, observations, detailed, process_id, manager_dict):
        predictions = self.predict_single_proc(observations, detailed)
        manager_dict[process_id] = predictions


    def predict(self, observations, detailed = False, multi_proc = False, num_proc = 1):
        """
        Evaluates an observation and returns its predicted class. 
        :param observation: binary input sequence (list with zeros and ones as integers).

        Returns: the class that returned the biggest discriminator response.
        """

        predictions = []

        if not multi_proc:
            predictions = self.predict_single_proc(observations, detailed)
        else:
            if num_proc <= 0:
                raise Exception("\"num_proc\" must be an integer >= 1.")

            num_observations = len(observations)
            observations_per_chunk = math.ceil(num_observations / num_proc)
            processes = {}
            processes_predictions = multiprocessing.Manager().dict()

            for i in range(num_proc):
                observations_chunk = []
                start_position = i * observations_per_chunk
                end_position = start_position + observations_per_chunk
                
                if i < (num_proc - 1):
                    observations_chunk = observations[start_position: end_position]
                else:
                    observations_chunk = observations[start_position:]

                process = multiprocessing.Process(
                    target = self.predict_multi_proc, 
                    args=(
                        observations_chunk,
                        detailed,
                        i,
                        processes_predictions
                    )
                )

                process.start()
                processes[i] = process

            for process in processes:
                processes[process].join()

            for i in range(num_proc):
                predictions = predictions + processes_predictions[i]  
            
        return predictions


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
        