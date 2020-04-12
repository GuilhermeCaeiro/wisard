from Wisard import Wisard
import matplotlib.pyplot as plt
import numpy as np

# Just a very simple test to see if things are working.

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

wann = Wisard(2, False, 3546)
wann.train(training_patterns, expected_outputs)

print(wann.get_mental_images())

for discriminator_class in wann.discriminators:
    print(wann.discriminators[discriminator_class].input_class)
    print(wann.discriminators[discriminator_class].input_length)
    print(wann.discriminators[discriminator_class].tuple_size)
    print(wann.discriminators[discriminator_class].memory)


print("\n\n\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")


wann = Wisard(2, True, 3546)
wann.train(training_patterns, expected_outputs)

print(wann.get_mental_images("H"))

for discriminator in wann.discriminators:
    print(wann.discriminators[discriminator_class].input_class)
    print(wann.discriminators[discriminator_class].input_length)
    print(wann.discriminators[discriminator_class].tuple_size)
    print(wann.discriminators[discriminator_class].memory)


print("\n\n\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")



wann = Wisard(2, True, 3546)
wann.train(training_patterns, expected_outputs)

for discriminator in wann.discriminators:
    print(wann.discriminators[discriminator_class].input_class)
    print(wann.discriminators[discriminator_class].input_length)
    print(wann.discriminators[discriminator_class].tuple_size)
    print(wann.discriminators[discriminator_class].memory)


wann.deactivate_bleaching()

print("\n\n\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")


mental_images = wann.get_mental_images()
for mental_image_class in mental_images:
    mental_image = mental_images[mental_image_class]
    mental_image = np.asarray(mental_image).reshape((4,3))

    print(mental_image)
    plt.imshow(mental_image, cmap='hot', interpolation='nearest')
    plt.show()
