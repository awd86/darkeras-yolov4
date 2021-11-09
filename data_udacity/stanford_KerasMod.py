#############################################################################
# Alex Denton, 9 Nov 2021, AE4824 @ NPS
#
# This is an implimenation of the code and method found here:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#
#############################################################################

# call as "python3 stanford_KerasMod.py

import numpy as np

from keras.models import Sequential
from stanford_DA import DataGenerator  # changed to reflect my file
#from my_classes import DataGenerator

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = # IDs
labels = # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture  # this is where you would input your model...
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

# As you can see, we called from model the fit_generator method instead of fit, where we just had to give our training generator as one of the arguments. Keras takes care of the rest!
#
#Note that our implementation enables the use of the multiprocessing argument of fit_generator, where the number of threads specified in workers are those that generate batches in parallel. A high enough number of workers assures that CPU computations are efficiently managed, i.e. that the bottleneck is indeed the neural network's forward and backward operations on the GPU (and not data generation).
