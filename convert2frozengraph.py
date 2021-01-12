import netvlad_tf.nets as nets

import os
import cv2
import numpy as np
import tensorflow as tf
import time

from tensorflow.python.framework import graph_io

output_node_name = ['vgg16_netvlad_pca/l2_normalize_3']
output_graph_name = './netvlad_frozen_model_dynamic_shape_nnn3.pb'
input_node_name = 'import/Placeholder:0'
placeholder_shape = [None, None, None, 3]
input_shape = [1,360,640,3]


def gen_frozen_graph():
    tf.reset_default_graph()
    image_batch = tf.placeholder(dtype=tf.float32, shape=placeholder_shape)
    net_out = nets.vgg16NetvladPca(image_batch)
    print('-------------------', net_out.name)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, nets.defaultCheckpoint())
        batch = np.ones(input_shape, dtype=np.float32)
        result = sess.run(net_out, feed_dict={image_batch: batch})

        with open("tf_python_checkpoint_output.txt", "w") as f:
            for r in result.flatten():
                f.write("%.8f\n"%r)

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names=output_node_name)
        frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(
            frozen_graph)

        graph_io.write_graph(frozen_graph,
                             '',
                             output_graph_name,
                             as_text=False)

    print('gen_frozen_graph done.')


def load_frozen_graph():
    with tf.io.gfile.GFile(output_graph_name, 'rb') as f:
        frozen_graph = tf.compat.v1.GraphDef()
        frozen_graph.ParseFromString(f.read())
    G = tf.Graph()
    with tf.Session(graph=G) as sess:
        output = tf.import_graph_def(
            frozen_graph, return_elements=[output_node_name[0] + ':0'])

        # [print(n.name) for n in sess.graph_def.node]
        inputs = G.get_tensor_by_name(input_node_name)
        inim = cv2.imread('example.jpg')
        batch = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        batch = np.ones(input_shape, dtype=np.float32)
        result_frozen_graph = sess.run(output, feed_dict={inputs: batch})
        with open("tf_python_frozen_graph_output.txt", "w") as f:
            for r in result_frozen_graph[0].flatten():
                f.write("%.8f\n"%r)
        print(result_frozen_graph)

if __name__ == '__main__':
    gen_frozen_graph()
    load_frozen_graph()
