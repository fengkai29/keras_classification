import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import os


def save_model_to_serving(model, export_version, export_path='prod_models'):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': model.input},
        outputs={'classification': model.output}
    )
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'keras_classification': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()


if __name__ == '__main__':
    emotion_model_path = '/Users/fengkai/PycharmProjects/keras_classification/output/resnet50.03-0.9864.hdf5'
    export_path = "/Users/fengkai/PycharmProjects/keras_classification/output"
    emotion_model = load_model(emotion_model_path)
    save_model_to_serving(emotion_model, "1", export_path)