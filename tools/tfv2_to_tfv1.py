import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def remove_tf2_nodes_attributes(inputGraph):
    for i in range(0, len(inputGraph.node) - 1):
        if inputGraph.node[i].op == "FusedBatchNormV3":
            del inputGraph.node[i].attr["exponential_avg_factor"]
        if inputGraph.node[i].op in ["DepthwiseConv2dNative", "MaxPool"]:
            del inputGraph.node[i].attr["explicit_paddings"]
        if inputGraph.node[i].op == "ResizeNearestNeighbor":
            del inputGraph.node[i].attr["half_pixel_centers"]
        if inputGraph.node[i].op.endswith("V2"):
            inputGraph.node[i].op = inputGraph.node[i].op[:-2]
    return inputGraph

graph_def = tf.compat.v1.GraphDef()
model_path = "/media/data/ayman/code/srlane_copy/work_dirs/sr_mtv/5006_3/exported/model_simplified_float32.pb"

with tf.io.gfile.GFile(model_path, "rb") as f:
    graph_def.ParseFromString(f.read())

modified_graph_def = remove_tf2_nodes_attributes(graph_def)

# # output_node_names = ["output"]
# # graphdef = tf.graph_util.convert_variables_to_constants(
# #     sess, graph_def, output_node_names
# # )

# # Import the graph into the current default graph
# with tf.Graph().as_default() as graph:
#     tf.import_graph_def(graph_def, name="")

# # Define the output node name
# output_node_names = ["output"]

# # Convert variables to constants
# frozen_func = convert_variables_to_constants_v2(
#     tf.function(lambda: None).get_concrete_function(),
#     lower_control_flow=False
# )

# # Rename the output node
# for node in frozen_func.graph.as_graph_def().node:
#     if node.name == output_node_names[0]:
#         node.name = "output"

# modified_graph_def = remove_tf2_nodes_attributes(graph_def)

# Save the modified graph
output_path = "/media/data/ayman/code/srlane_copy/work_dirs/sr_mtv/5006_3/exported/model_modified_float32.pb"
with tf.io.gfile.GFile(output_path, "wb") as f:
    f.write(modified_graph_def.SerializeToString())