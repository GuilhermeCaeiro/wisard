from Wisard import Wisard
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score

# Just a very simple test to see if things are working.

def print_discriminators_info(wann):
    for discriminator_class in wann.discriminators:
        discriminator = wann.discriminators[discriminator_class]

        print(discriminator.input_class)
        print(discriminator.input_length)
        print(discriminator.tuple_size)
        print(discriminator.memory)

if __name__ == "__main__":    

    training_set = [
                        [[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], "H"],
                        [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "H"],
                        [[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], "H"],
                        [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "A"],
                        [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], "O"],
                    ]

    test_set = [
                        [[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], "H"],
                        [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "H"],
                        [[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], "H"],
                        [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "A"],
                        [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], "O"],
                        [[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], "H"],
                        [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], "A"],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "0"],
                    ]

    training_patterns = [row[0] for row in training_set]
    expected_outputs = [row[1] for row in training_set]
    test_patterns = [row[0] for row in test_set]
    test_expected_outputs = [row[1] for row in test_set]

    wann = Wisard(2, False, 3546, type_mem_alloc="dalloc")
    wann.train(training_patterns, expected_outputs)
    print(wann.get_mental_images())
    print_discriminators_info(wann)
    print("\n\n\n")
    for pattern in test_set:
        print("Evaluating pattern ", pattern)
        print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")

    ##########

    wann = Wisard(2, True, 3546, type_mem_alloc="palloc")
    wann.train(training_patterns, expected_outputs)
    print(wann.get_mental_images("H"))
    print_discriminators_info(wann)
    print("\n\n\n")
    for pattern in test_set:
        print("Evaluating pattern ", pattern)
        print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")

    ##########

    wann = Wisard(2, True, 3546, type_mem_alloc="dalloc")
    wann.train(training_patterns, expected_outputs)
    print(wann.get_mental_images())
    print_discriminators_info(wann)
    print("\n\n\n")
    wann.deactivate_bleaching()
    for pattern in test_set:
        print("Evaluating pattern ", pattern)
        print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")

    mental_images = wann.get_mental_images()
    for mental_image_class in mental_images:
        mental_image = mental_images[mental_image_class]
        mental_image = np.asarray(mental_image).reshape((4,3))

        print(mental_image)
        plt.imshow(mental_image, cmap='hot', interpolation='nearest')
        plt.show()

    ##########

    print("TRAINING NETWORK.")

    wann = Wisard(2, True, 3546, type_mem_alloc="dalloc")
    wann.train(training_patterns, expected_outputs)
    for pattern in test_set:
        print("Evaluating pattern ", pattern)
        print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")

    print("PRINTING MENTAL IMAGE.")
    mental_images = wann.get_mental_images()
    for mental_image_class in mental_images:
        mental_image = mental_images[mental_image_class]
        mental_image = np.asarray(mental_image).reshape((4,3))
        plt.imshow(mental_image, cmap='hot', interpolation='nearest')
        plt.show()

    print("SAVING NETWORK.")
    wann.save("wann.pkl")

    print("LOADING NETWORK.")
    wann = Wisard.load("wann.pkl")
    for pattern in test_set:
        print("Evaluating pattern ", pattern)
        print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")

    print("PRINTING MENTAL IMAGE.")
    mental_images = wann.get_mental_images()
    for mental_image_class in mental_images:
        mental_image = mental_images[mental_image_class]
        mental_image = np.asarray(mental_image).reshape((4,3))
        plt.imshow(mental_image, cmap='hot', interpolation='nearest')
        plt.show()

    print("TRAINING ADDITIONAL PATTERNS.")
    wann.train([test_patterns[0]], [test_expected_outputs[0]])
    wann.train([test_patterns[6]], [test_expected_outputs[6]])

    print("PRINTING MENTAL IMAGE.")
    mental_images = wann.get_mental_images()
    for mental_image_class in mental_images:
        mental_image = mental_images[mental_image_class]
        mental_image = np.asarray(mental_image).reshape((4,3))
        plt.imshow(mental_image, cmap='hot', interpolation='nearest')
        plt.show()

    ########################################################################

    NUM_SAMPLES = 100000
    TEST_TIME = False

    def print_avg_time_type_malloc(type_malloc):
        wann = Wisard(2, True, 3546, type_mem_alloc=type_malloc)
        wann.train(training_patterns, expected_outputs)
        for i in range(NUM_SAMPLES):
            for pattern in test_set:
                wann.predict([pattern[0]])
        #wann.print_time_data()

    if TEST_TIME:
        for malloc_type in ("dalloc", "palloc"):
            print(malloc_type)
            print_avg_time_type_malloc(malloc_type)

    import sys
    print("Python version")
    print (sys.version)
    print("Version info.")
    print (sys.version_info)

""" Test with mnist """

def encode(a_file):
    mnist_file = open(a_file, "r")

    encoded_observ = []
    targets = []

    for line in mnist_file:
        values = line.split(",")
        target = values[0]
        features = [1 if int(number) > 64 else 0 for number in values[1:]]

        encoded_observ.append(features)
        targets.append(target)

    return encoded_observ, targets

if __name__ == "__main__":    
	# Dataset files can be found at "https://www.kaggle.com/vikramtiwari/mnist-ml-crash-course"
    training_observations, training_targets = encode("sample_data/mnist_train_small.csv")
    test_observations, test_targets = encode("sample_data/mnist_test.csv")

    print(len(training_observations), len(training_targets))
    print(len(test_observations), len(test_targets))

    wann = Wisard(2, True, 3546, type_mem_alloc="palloc")
    wann.train(training_observations, training_targets)

    mental_images = wann.get_mental_images()
    for mental_image_class in mental_images:
        mental_image = mental_images[mental_image_class]
        mental_image = np.asarray(mental_image).reshape((28,28))

        #print(mental_image)
        plt.imshow(mental_image, cmap='hot', interpolation='nearest')
        plt.show()

    print("In-sample error")
    start_time = time.time()
    predictions = wann.predict(training_observations, detailed = False, multi_proc = True, num_proc = 2)
    print(time.time() - start_time)
    #wann.print_time_data()

    print(len(training_targets), accuracy_score(training_targets, predictions))

    print("Out of sample error")
    start_time = time.time()
    predictions = wann.predict(test_observations, detailed = False, multi_proc = True, num_proc = 2)
    print(time.time() - start_time)
    #wann.print_time_data()

    print(len(test_targets), accuracy_score(test_targets, predictions))

