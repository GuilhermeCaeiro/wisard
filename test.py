from Wisard import Wisard
import matplotlib.pyplot as plt
import numpy as np

# Just a very simple test to see if things are working.

def print_discriminators_info(wann):
    for discriminator_class in wann.discriminators:
        discriminator = wann.discriminators[discriminator_class]

        print(discriminator.input_class)
        print(discriminator.input_length)
        print(discriminator.tuple_size)
        print(discriminator.memory)
    

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

wann = Wisard(2, False, 3546)
wann.train(training_patterns, expected_outputs)
print(wann.get_mental_images())
print_discriminators_info(wann)
print("\n\n\n")
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")
wann.print_time_data()

##########

wann = Wisard(2, True, 3546)
wann.train(training_patterns, expected_outputs)
print(wann.get_mental_images("H"))
print_discriminators_info(wann)
print("\n\n\n")
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")
wann.print_time_data()

##########

wann = Wisard(2, True, 3546)
wann.train(training_patterns, expected_outputs)
print(wann.get_mental_images())
print_discriminators_info(wann)
print("\n\n\n")
wann.deactivate_bleaching()
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict([pattern[0]]), "Expected: ", pattern[1], "\n")
wann.print_time_data()

mental_images = wann.get_mental_images()
for mental_image_class in mental_images:
    mental_image = mental_images[mental_image_class]
    mental_image = np.asarray(mental_image).reshape((4,3))

    print(mental_image)
    plt.imshow(mental_image, cmap='hot', interpolation='nearest')
    plt.show()

########################################################################

NUM_SAMPLES = 100000

def print_avg_time_type_malloc(type_malloc):
    wann = Wisard(2, True, 3546, type_mem_alloc=type_malloc)
    wann.train(training_patterns, expected_outputs)
    for i in range(NUM_SAMPLES):
        for pattern in test_set:
            wann.predict([pattern[0]])
    wann.print_time_data()

for i in range(0,4):
    print(i)
    print_avg_time_type_malloc(i)
