
r"""Converts checkpoint variables into Const ops in a standalone GraphDef file.

This script is designed to take a GraphDef proto, a SaverDef proto, and a set of
variable values stored in a checkpoint file, and output a GraphDef with all of
the variable ops converted into const ops containing the values of the
variables.

It's useful to do this when we need to load a single file in C++, especially in
environments like mobile or embedded where we may not have access to the
RestoreTensor ops and file loading calls that they rely on.

An example of command-line usage is:
bazel build tensorflow/python/tools:freeze_graph && \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=some_graph_def.pb \
--input_checkpoint=model.ckpt-8361242 \
--output_graph=/tmp/frozen_graph.pb --output_node_names=softmax

You can also look at freeze_graph_test.py for an example of how to use it.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from google.protobuf import text_format
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import saver as saver_lib

#FLAGS = None
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_graph', './export_in_graph/ur_directory.pb', 'The address of TensorFlow \'GraphDef\' file to load.')

tf.app.flags.DEFINE_string(
    'input_checkpoint',
    '/home/ur_directory',
    'The address of TensorFlow variables file to load.')

tf.app.flags.DEFINE_string(
    'output_node_names', 'MobilenetV1/Predictions/Reshape_1', 'The name of the output nodes, comma separated.')


tf.app.flags.DEFINE_string(
    'output_graph',
    './output_frozen_graph/ur_directory.pb',
    'The address of Output \'GraphDef\' file name writing to.')

tf.app.flags.DEFINE_boolean(
    'input_binary', True,
    'Whether the input files are in binary format.')

tf.app.flags.DEFINE_string(
    'input_saver', '', 'TensorFlow saver file to load.'
)

tf.app.flags.DEFINE_string(
    'restore_op_name', 'save/restore_all', 'The name of the master restore operator. Deprecated, unused by updated \
      loading code.'
)

tf.app.flags.DEFINE_string(
    'filename_tensor_name', 'save/Const:0', 'The name of the tensor holding the save path. Deprecated, unused by \
      updated loading code.'
)

tf.app.flags.DEFINE_boolean(
    'clear_devices', True, 'Whether to remove device specifications.'
)

tf.app.flags.DEFINE_string(
    'initializer_nodes', '', 'Comma separated list of initializer nodes to run before freezing.'
)

tf.app.flags.DEFINE_string(
    'variable_names_whitelist', '', 'Comma separated list of variables to convert to constants. If specified, \
      only those variables will be converted to constants.'
)

tf.app.flags.DEFINE_string(
    'variable_names_blacklist', '', 'Comma separated list of variables to skip converting to constants.'
)

tf.app.flags.DEFINE_string(
    'input_meta_graph', '', 'TensorFlow \'MetaGraphDef\' file to load.'
)

tf.app.flags.DEFINE_string(
    'input_saved_model_dir', '', 'Path to the dir with TensorFlow \'SavedModel\' file and variables.'
)

tf.app.flags.DEFINE_string(
    'saved_model_tags', 'serve', 'Group of tag(s) of the MetaGraphDef to load, in string format'
)

tf.app.flags.DEFINE_integer(
    'checkpoint_version', saver_pb2.SaverDef.V2, 'Tensorflow variable file format'
)

def freeze_graph_with_def_protos(input_graph_def,
                                 input_saver_def,
                                 input_checkpoint,
                                 output_node_names,
                                 restore_op_name,
                                 filename_tensor_name,
                                 output_graph,
                                 clear_devices,
                                 initializer_nodes,
                                 variable_names_whitelist="",
                                 variable_names_blacklist="",
                                 input_meta_graph_def=None,
                                 input_saved_model_dir=None,
                                 saved_model_tags=None,
                                 checkpoint_version=saver_pb2.SaverDef.V2):
  """Converts all variables in a graph and checkpoint into constants."""
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if (not input_saved_model_dir and
      not saver_lib.checkpoint_exists(input_checkpoint)):
    print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
    return -1

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    if input_meta_graph_def:
      for node in input_meta_graph_def.graph_def.node:
        node.device = ""
    elif input_graph_def:
      for node in input_graph_def.node:
        node.device = ""

  if input_graph_def:
    _ = importer.import_graph_def(input_graph_def, name="")
  with session.Session() as sess:
    if input_saver_def:
      saver = saver_lib.Saver(saver_def=input_saver_def,
                              write_version=checkpoint_version)
      saver.restore(sess, input_checkpoint)
    elif input_meta_graph_def:
      restorer = saver_lib.import_meta_graph(
          input_meta_graph_def, clear_devices=True)
      restorer.restore(sess, input_checkpoint)
      if initializer_nodes:
        sess.run(initializer_nodes.split(","))
    elif input_saved_model_dir:
      if saved_model_tags is None:
        saved_model_tags = []
      loader.load(sess, saved_model_tags, input_saved_model_dir)
    else:
      var_list = {}
      reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        try:
          tensor = sess.graph.get_tensor_by_name(key + ":0")
        except KeyError:
          # This tensor doesn't exist in the graph (for example it's
          # 'global_step' or a similar housekeeping element) so skip it.
          continue
        var_list[key] = tensor
      saver = saver_lib.Saver(var_list=var_list,
                              write_version=checkpoint_version)
      saver.restore(sess, input_checkpoint)
      if initializer_nodes:
        sess.run(initializer_nodes.split(","))

    variable_names_whitelist = (variable_names_whitelist.split(",")
                                if variable_names_whitelist else None)
    variable_names_blacklist = (variable_names_blacklist.split(",")
                                if variable_names_blacklist else None)

    if input_meta_graph_def:
      output_graph_def = graph_util.convert_variables_to_constants(
          sess,
          input_meta_graph_def.graph_def,
          output_node_names.split(","),
          variable_names_whitelist=variable_names_whitelist,
          variable_names_blacklist=variable_names_blacklist)
    else:
      output_graph_def = graph_util.convert_variables_to_constants(
          sess,
          input_graph_def,
          output_node_names.split(","),
          variable_names_whitelist=variable_names_whitelist,
          variable_names_blacklist=variable_names_blacklist)

  # Write GraphDef to file if output path has been given.
  if output_graph:
    with gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())

  return output_graph_def


def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def


def _parse_input_meta_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into MetaGraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input meta graph file '" + input_graph + "' does not exist!")
    return -1
  input_meta_graph_def = MetaGraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_meta_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_meta_graph_def)
  print("Loaded meta graph file '" + input_graph)
  return input_meta_graph_def


def _parse_input_saver_proto(input_saver, input_binary):
  """Parser input tensorflow Saver into SaverDef proto."""
  if not gfile.Exists(input_saver):
    print("Input saver file '" + input_saver + "' does not exist!")
    return -1
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_saver, mode) as f:
    saver_def = saver_pb2.SaverDef()
    if input_binary:
      saver_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), saver_def)
  return saver_def


def freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_whitelist="",
                 variable_names_blacklist="",
                 input_meta_graph=None,
                 input_saved_model_dir=None,
                 saved_model_tags=tag_constants.SERVING,
                 checkpoint_version=saver_pb2.SaverDef.V2):
  """Converts all variables in a graph and checkpoint into constants."""
  input_graph_def = None
  if input_saved_model_dir:
    input_graph_def = saved_model_utils.get_meta_graph_def(
        input_saved_model_dir, saved_model_tags).graph_def
  elif input_graph:
    input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
  input_meta_graph_def = None
  if input_meta_graph:
    input_meta_graph_def = _parse_input_meta_graph_proto(
        input_meta_graph, input_binary)
  input_saver_def = None
  if input_saver:
    input_saver_def = _parse_input_saver_proto(input_saver, input_binary)
  freeze_graph_with_def_protos(
      input_graph_def, input_saver_def, input_checkpoint, output_node_names,
      restore_op_name, filename_tensor_name, output_graph, clear_devices,
      initializer_nodes, variable_names_whitelist, variable_names_blacklist,
      input_meta_graph_def, input_saved_model_dir,
      saved_model_tags.split(","), checkpoint_version=checkpoint_version)



def main(_):
    freeze_graph(FLAGS.input_graph, FLAGS.input_saver, FLAGS.input_binary,
                 FLAGS.input_checkpoint, FLAGS.output_node_names,
                 FLAGS.restore_op_name, FLAGS.filename_tensor_name,
                 FLAGS.output_graph, FLAGS.clear_devices, FLAGS.initializer_nodes,
                 FLAGS.variable_names_whitelist, FLAGS.variable_names_blacklist,
                 FLAGS.input_meta_graph, FLAGS.input_saved_model_dir,
                 FLAGS.saved_model_tags,FLAGS.checkpoint_version)



if __name__ == "__main__":
   
    tf.app.run()