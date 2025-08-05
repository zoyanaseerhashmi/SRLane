import numpy as np
import tensorflow as tf

saved_model_path = "/media/data/ayman/code/srlane_copy/work_dirs/sr_mtv/5006_3/exported"
saved_model = tf.saved_model.load(saved_model_path)

input_ndarray = np.zeros((1, 448, 800, 3 )) # this is just the ndarray from the input
model = saved_model.signatures["serving_default"]
input_tensor = tf.convert_to_tensor(input_ndarray, dtype=tf.float32) # also tried not specifying the type
result = model(input_tensor)
print(result)