# Getting started


First define your Keras Neural Network and convert it into its `decomon` version
using the `clone` method. You can then call `get_upper_box` and `get_lower_box` to
respectively obtain certified upper and lower bounds for the network's outputs
within a box domain.

````python
# Imports
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from decomon import get_lower_box, get_upper_box
from decomon.models import clone

# Toy example with a Keras model:
model = Sequential([Dense(10, activation="relu", input_dim=2)])

# Create a fake box with the right shape
x_min = np.zeros((1, 2))
x_max = np.ones((1, 2))

# Convert into a decomon neural network
decomon_model = clone(model)

# Get lower and upper bounds
lower_bound = get_lower_box(decomon_model, x_min, x_max)
upper_bound = get_upper_box(decomon_model, x_min, x_max)

print(lower_bound)
print(upper_bound)
````

Other types of domains are possible and illustrated
in our [tutorials](tutorials).
