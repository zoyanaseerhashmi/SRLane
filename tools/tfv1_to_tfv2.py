import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def convert_to_tf1_graph(model, output_dir):
    # Create a concrete function from the model's serving signature
    concrete_func = model.signatures["serving_default"]

    # Convert the concrete function to a TensorFlow v1 graph
    # frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants_v2(concrete_func)
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_graph_def = frozen_func.graph.as_graph_def()

    # Save the frozen graph to a `.pb` file
    pb_model_path = f"{output_dir}/model_tf1.pb"
    with tf.io.gfile.GFile(pb_model_path, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
    print(f"TensorFlow v1-compatible model saved at {pb_model_path}")

# Example usage
model = tf.saved_model.load("/media/data/ayman/code/srlane_copy/work_dirs/sr_mtv/5006_3/exported/")
output_dir = "/media/data/ayman/code/srlane_copy/work_dirs/sr_mtv/5006_3/exported"
convert_to_tf1_graph(model, output_dir)