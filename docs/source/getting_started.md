# Getting started


First define your Keras Neural Network and convert it into its `decomon` version
using the `clone` method. You can then call `get_upper_box` and `get_lower_box` to
respectively obtain certified upper and lower bounds for the network's outputs
within a box domain.

````python
# Imports
import keras
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential

# from decomon import get_lower_box, get_upper_box
from decomon import clone

# Toy example with a Keras model:
model = Sequential([Input((2,)), Dense(10, activation="relu")])

# Convert into a decomon neural network
decomon_model = clone(
    model,
    final_ibp=True, # keep constant bounds
    final_affine=False  # drop affine bounds
)

# Create a fake box with the right shape
x_min = np.zeros((1, 2))
x_max = np.ones((1, 2))
x_box = np.concatenate([x_min[:, None], x_max[:, None]], axis=1)

# Get lower and upper bounds
lower_bound, upper_bound = decomon_model.predict_on_single_batch_np(x_box)  # more efficient than predict on very small batch

print(f"lower bound: {lower_bound}")
print(f"upper bound: {upper_bound}")
````

Other types of domains are possible and illustrated
in our [tutorials](tutorials).
