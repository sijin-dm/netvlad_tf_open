import netvlad_tf.nets as nets

import os
import cv2
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import graph_io

input_node_name = 'input'
output_node_name = 'output'
frozen_graph_output_name = 'model.graphdef'
onnx_output_name = 'model.onnx'
placeholder_shape = [None, 720, 1280, 3]
input_shape = [1, 720, 1280, 3]


def replace(checkpoint_dir, replace_from, replace_to, save_name=None):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    assert None not in [replace_from, replace_to], 'must specify replace_from and replace_to'
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            new_name = new_name.replace(replace_from, replace_to)

            print('%-50s ==> %-50s' % (var_name, new_name))
            # Rename the variable
            var = tf.Variable(var, name=new_name)

        if save_name is not None:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, save_name)
            print("replace checkpoint was save to {}".format(save_name))


def gen_serving_model(checkpoint_dir=nets.defaultCheckpoint(), save_result=False, is_onnx_model=False):
    tf.reset_default_graph()
    net_in = tf.placeholder(dtype=tf.float32, shape=placeholder_shape, name='input')
    net_out = nets.vgg16NetvladPca(net_in)
    saver = tf.compat.v1.train.Saver()
    input_node_name = net_in.name[:-2]  # delete :0
    output_node_name = net_out.name[:-2]  # delete :0
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_dir)
        if save_result:
            batch = np.ones(input_shape, dtype=np.float32)
            result = sess.run(net_out, feed_dict={net_in: batch})
            with open("tf_python_checkpoint_output.txt", "w") as f:
                for r in result.flatten():
                    f.write("%.8f\n" % r)
        # for node in sess.graph_def.node:
        #     print(node.name)
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                    sess.graph_def,
                                                                    output_node_names=[output_node_name])
        frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
    if is_onnx_model:
        import tf2onnx
        model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(frozen_graph,
                                                                              input_names=[net_in.name],
                                                                              output_names=[net_out.name],
                                                                              opset=11,
                                                                              output_path=onnx_output_name)
        print('Onnx InputName: [{}] \t->\t OutputName: [{}]'.format(input_node_name, output_node_name))
    else:
        graph_io.write_graph(frozen_graph, '', frozen_graph_output_name, as_text=False)
        print('Frozen graph InputName: [{}] \t->\t OutputName: [{}]'.format(input_node_name, output_node_name))


def load_frozen_graph():
    with tf.io.gfile.GFile(frozen_graph_output_name, 'rb') as f:
        frozen_graph = tf.compat.v1.GraphDef()
        frozen_graph.ParseFromString(f.read())
    G = tf.Graph()
    with tf.Session(graph=G) as sess:
        output = tf.import_graph_def(frozen_graph, return_elements=[output_node_name + ':0'], name='')
        # [print(n.name) for n in sess.graph_def.node]
        inputs = G.get_tensor_by_name(input_node_name + ':0')
        # inim = cv2.imread('example.jpg')
        # batch = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        batch = np.ones(input_shape, dtype=np.float32)
        result_frozen_graph = sess.run(output, feed_dict={inputs: batch})
        with open("tf_python_frozen_graph_output.txt", "w") as f:
            for r in result_frozen_graph[0].flatten():
                f.write("%.8f\n" % r)
        print(result_frozen_graph[0].shape)


if __name__ == '__main__':
    checkpoint = nets.defaultCheckpoint() + '_strip'
    # replace(nets.defaultCheckpoint(), 'vgg16_netvlad_pca/', '', save_name=checkpoint)
    gen_serving_model(checkpoint_dir=checkpoint, is_onnx_model=True)
    # load_frozen_graph()
