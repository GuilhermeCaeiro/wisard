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


