import math
import numpy as np

cdef class DiscriminatorWisard:
    """
    Represents a WiSARD discriminator, which contains a set of random access memories.      
    """
    cdef public:
        cdef str input_class
        cdef int input_length
        cdef int tuple_size
        cdef int ram_size
        cdef int number_of_rams
        cdef bint bleaching
        #cdef dict memory_dict
        #cdef list memory_list
        cdef object memory
        cdef str type_mem_alloc
        cdef object write
        cdef object evaluate
        cdef object generate_mental_image



    def __init__(self, str input_class, int input_length, int tuple_size, bint bleaching = False, str type_mem_alloc = "dalloc"):
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
        #self.memory_dict = None
        #self.memory_list = None
        self.memory = None
        self.type_mem_alloc = type_mem_alloc

        if type_mem_alloc == "dalloc":
            self.memory = {}
            #self.memory_dict = {}
            self.write = self.write_dinamyc_alloc
            self.evaluate = self.evaluate_dinamyc_alloc
            self.generate_mental_image = self.generate_mental_image_dalloc
        elif type_mem_alloc == "palloc":
            self.memory = [0] * int(self.number_of_rams * self.ram_size) #np.zeros((int(self.number_of_rams * self.ram_size)))
            #self.memory_list = [0] * int(self.number_of_rams * self.ram_size)
            #self.memory = np.zeros(self.number_of_rams * self.ram_size, dtype=np.int)
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
                    bin_address = ("{0:0" + str(self.tuple_size) + "b}").format(address) # remove if address is string

                    if bin_address[i] == "1" and self.memory[ram][address] > 0: # change bin_address to address if address is string
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




class DiscriminatorRegressionWisard:
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
        self.memory = None              # counters
        self.memory_y = None            # partial ys
        self.type_mem_alloc = type_mem_alloc

        if type_mem_alloc == "dalloc":
            self.memory = {}
            self.memory_y = {}
            self.write = self.write_dinamyc_alloc
            self.evaluate = self.evaluate_dinamyc_alloc
            self.generate_mental_image = self.generate_mental_image_dalloc
        elif type_mem_alloc == "palloc":
            self.memory = [0] * int(self.number_of_rams * self.ram_size) #np.zeros((int(self.number_of_rams * self.ram_size)))
            self.memory_y = [0] * int(self.number_of_rams * self.ram_size)
            self.write = self.write_pre_alloc_array
            self.evaluate = self.evaluate_pre_alloc_array
            self.generate_mental_image = self.generate_mental_image_palloc
        else:
            raise Exception("\"type_mem_alloc\" only accepts the values \"dalloc\", for dynamic allocation and \"palloc\", for pre-allocation.")

    def write_dinamyc_alloc(self, addresses, y):
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
                self.memory_y[i][address] = y
            elif address in self.memory[i] and self.bleaching:
                self.memory[i][address] = self.memory[i][address] + 1
                self.memory_y[i][address] = self.memory_y[i][address] + y


    def evaluate_dinamyc_alloc(self, addresses, bleaching_threshold = 0):
        """
        Evaluates a pattern and returns its score.
        :param pattern: pattern to be evaluated.
        :param bleaching_threshold: threshold to be used to solve draws.
        Returns:
            -> discriminator score.
        """

        counters = []
        partial_ys = []

        for i in range(self.number_of_rams):
            address = addresses[i]

            if i in self.memory and address in self.memory[i]:
                counters.append(self.memory[i][address])
                partial_ys.append(self.memory_y[i][address])

        return counters, partial_ys
            

    def write_pre_alloc_array(self, addresses, y):
        """
        Writes a pattern to the RAMs.
        :param pattern: binary pattern to be learned.
        """

        for i in range(self.number_of_rams): 
            ram_position = addresses[i]
            ram = i * self.ram_size + ram_position
            self.memory[ram] = self.memory[ram] + 1
            self.memory_y[ram] = self.memory_y[ram] + y
        
            
    def evaluate_pre_alloc_array(self, addresses, bleaching_threshold = 0):
        """
        Evaluates a pattern and returns its score.
        :param pattern: pattern to be evaluated.
        :param bleaching_threshold: threshold to be used to solve draws.
        Returns:
            -> discriminator score.
        """
        
        counters = []
        partial_ys = []

        for i in range(self.number_of_rams):
            ram_position = addresses[i]
            position = i * self.ram_size + ram_position

            if self.self.memory[position] != 0:
                counters.append(self.memory[position])
                partial_ys.append(self.memory_y[position])
                
        return counters, partial_ys


    def generate_mental_image_dalloc(self):
        pixels = []

        for ram in sorted(self.memory):
            chunk = [0] * self.tuple_size

            for i in range(self.tuple_size):
                for address in self.memory[ram]:
                    bin_address = ("{0:0" + str(self.tuple_size) + "b}").format(address) # remove if address is string

                    if bin_address[i] == "1" and self.memory[ram][address] > 0: # change bin_address to address if address is string
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