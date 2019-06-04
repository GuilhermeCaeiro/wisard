"""
Module that implements the WiSARD classifier.
"""

import random
import math
import numpy as np
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
        #self.memory = np.zeros(self.number_of_rams, self.ram_size)
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

    def __init__(self, tuple_size = 2, bleaching = False, seed = 0):
        """
        Constructor for the WiSARD class.

        :param tuple_size: number (integer) that defines the address size to be used by the memories.
        :param bleaching: boolean indicating if the bleaching functionality is active or not.
        :param seed: integer to be used as seed in the random number generator, to allow reproducibility.
        """

        #self.input_list = []
        #self.expected_output_list = []
        self.seed = seed
        self.bleaching = bleaching
        self.tuple_size = tuple_size
        self.discriminators = []

    def process_input(self, mode, input_list, expected_output_list = []):
        """
        Prepares the input by appling to it the random mapping of the bits in the input sequence, based on the seed provided during the class creation. 

        :param mode: string identifying its mode of operation as "training" or "prediction".
        :param input_list: list of binary input sequences (lists with zeros and ones as integers).
        :param expected_output_list: list of expected outputs.

        Returns: 
            -> For "training" mode: a dictionary containing the input classes and their respective transformed training observations.
            -> For "prediction" mode: the transformed observation.
        """

        # separates classes
        input_classes = {}

        if mode == "training":
            for i in range(len(input_list)):
                if expected_output_list[i] not in input_classes:
                    input_classes[expected_output_list[i]] = []
                
                # shuffles input list
                input_item = copy.deepcopy(input_list[i])
                
                random.seed(self.seed)
                random.shuffle(input_item)

                input_classes[expected_output_list[i]].append(input_item)

            return input_classes
        elif mode == "prediction":
            input_item = copy.deepcopy(input_list[0])

            random.seed(self.seed)
            random.shuffle(input_item)
            
            return input_item
        else:
            raise ValueError("Invalid mode identifier.")

    def train(self, input_list, expected_output_list):
        """
        Trains the WiSARD classifier based on the provide inputs and its expected outputs. 

        :param input_list: list of binary input sequences (lists with zeros and ones as integers).
        :param expected_output_list: list of expected outputs (preferrably as strings)
        """

        input_classes = self.process_input("training", input_list, expected_output_list)
        number_of_classes = len(input_classes)
        class_identifiers = list(input_classes.keys())
        class_identifiers.sort() # sorts the classes
        #class_identifiers = class_identifiers[::-1] # reverses the classes' list
        
        print("Number of classes being trained: " + str(number_of_classes))
        print(input_classes.keys())
        print(class_identifiers)

        # TODO Change to allow online training.
        for input_class in class_identifiers:
            print("Number of training samples for class " + str(input_class) + ": " + str(len(input_classes[input_class])))

            input_data_length = len(input_classes[input_class][0])
            discriminator = Discriminator(input_class, input_data_length, self.tuple_size, self.bleaching)

            for training_sample in input_classes[input_class]:
                discriminator.write(training_sample)

            self.discriminators.append(discriminator)

    def predict(self, rawinput):
        """
        Evaluates a binary input sequence and returns its class. 

        :param rawinput: binary input sequence (list with zeros and ones as integers).

        Returns: the class that returned the biggest discriminator response.
        """

        processed_input = self.process_input("prediction", [rawinput])

        result_achieved = False

        predicted_classes = []
        current_threshold = 0

        discriminators_to_evaluate = self.discriminators
        previously_evaluated_discriminators = []

        while not result_achieved:
            predicted_classes = [{"discriminator": None, "score": 0}]

            for discriminator in discriminators_to_evaluate:
                score, evaluation_pattern, address = discriminator.evaluate(processed_input, current_threshold)
                last_score = score

                if score > predicted_classes[0]["score"] or predicted_classes[0]["discriminator"] == None:
                    predicted_classes = [{"discriminator": discriminator, "score": score}]
                elif score == predicted_classes[0]["score"]:
                    predicted_classes.append({"discriminator": discriminator, "score": score})


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
                        discriminators_to_evaluate.append(predicted_class["discriminator"])

            elif self.bleaching and len(predicted_classes) == 1:
                exit_condition = 3
                result_achieved = True
            else:
                print("Error predicting class.")
                break

            #
            #if predicted_classes[0]["discriminator"] is None or predicted_classes[0]["score"] is 0:
            #    print("Last score: ", last_score)
            #    print("Evaluation pattern: ", evaluation_pattern)
            #    print("Last address: ", address)
            #    print("Exit condition: ", exit_condition)
            #    print("Discriminators: ", discriminators_to_evaluate)
            #    print("Discriminators: ", previously_evaluated_discriminators)
            #    print("Prediction failed (score equals 0). Picking the first discriminator's.")
                

        # If the method ends with more than one class as possible, it just returns the first one.
        # TODO: Change the following line to return a random class if there is still a draw between
        # two or more classes.
        return {"class": predicted_classes[0]["discriminator"].input_class, "score": predicted_classes[0]["score"]}, True if len(predicted_classes) > 1 else False

    def deactivate_bleaching(self):
        """
        Dectivates bleaching, if that functionality is active. Does nothing otherwise.
        """
        self.bleaching = False

    def show_mental_map(self):
        """
        Not implemented yet.
        """
        pass

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
        
