import keras.backend as K
import jieba
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)

path = 'model'
jieba.load_userdict('{}/dic.txt'.format(path))

model = load_model('{}/bank_classification.h5'.format(path))


export_path = 'bank_clf_export/1'

# I want the full prediction tensor out, not classification. This format: {"image": Resnet50model.input} took me a while to track down
prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"input": model.input},
                                                                                {"prediction":model.output})

# export_path is a directory in which the model will be created
builder = saved_model_builder.SavedModelBuilder(export_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')


# Initialize global variables and the model
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature,
      },
      legacy_init_op=legacy_init_op)
# save the graph
builder.save()
print('done!')
