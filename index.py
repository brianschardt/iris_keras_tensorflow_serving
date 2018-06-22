import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(0)  # all new operations will be in test mode from now on

model_version = "2" #Change this to export different model versions, i.e. 2, ..., 7
epoch = 100 ## the higher this number is the more accurate the prediction will be 5000 is a good number to set it at just takes a while to train

seed = 7
np.random.seed(seed)

dataframe = pd.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
one_hot_labels = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(8, input_dim=4, activation="relu"))
model.add(Dense(3, activation="softmax"))

serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'x': tf.FixedLenFeature(shape=[4], dtype=tf.float32),}
tf_example = tf.parse_example(serialized_tf_example, feature_configs)

x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
y = model(x)

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(X, one_hot_labels, epochs=epoch, batch_size=32)

labels = []
for label in Y:
    if label not in labels:
        labels.append(label)

values, indices = tf.nn.top_k(y, len(labels))
table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(labels))
prediction_classes = table.lookup(tf.to_int64(indices))

classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

classification_signature = tf.saved_model.signature_def_utils.classification_signature_def(
    examples=serialized_tf_example,
    classes=prediction_classes,
    scores=values
)

prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"inputs": x}, {"prediction":y})

valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
valid_classification_signature = tf.saved_model.signature_def_utils.is_valid_signature(classification_signature)

if(valid_prediction_signature == False):
    raise ValueError("Error: Prediction signature not valid!")

if(valid_classification_signature == False):
    raise ValueError("Error: Classification signature not valid!")

builder = saved_model_builder.SavedModelBuilder('./'+model_version)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           'predict-iris':
               prediction_signature,
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)
# save the graph
builder.save()

