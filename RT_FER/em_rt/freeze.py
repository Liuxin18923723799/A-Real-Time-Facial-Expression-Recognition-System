import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#dir = os.path.dirname(os.path.realpath('freeze.py'))
dir = '/home/joy/Desktop/YU/slim_fer2013/em_rt'
input_checkpoint = 'model.ckpt-603764'

#def freeze_graph(model_folder):
def freeze_graph(input_checkpoint):
    # We retrieve our checkpoint fullpath
    '''
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    print checkpoint
    input_checkpoint = checkpoint.model_checkpoint_path
    '''

    '''
    # We precise the file fullname of our frozen graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_graph.pb"
    '''
    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes


    #output_node_names = "MobilenetV1/Predictions/Reshape_1"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    print input_checkpoint + '.meta'
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    print "start a session"
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        print "restore"
    # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names=['MobilenetV1/Predictions/Reshape_1']
            #output_node_names.split(",")# We split on comma for convenience
        )

    # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile('frozen_decay20_center0.001.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    freeze_graph(input_checkpoint)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default=dir, type=str, help="Model folder to export")
    args = parser.parse_args()
    print args.model_folder
    '''
    #print dir
    #freeze_graph(args.model_folder)
    #freeze_graph("output")