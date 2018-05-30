# pip install tensorflow-serving-api-python3

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server',
                           'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def do_inference(hostport):
    """Tests PredictionService with concurrent requests.
    Args:
    hostport: Host:port address of the Prediction Service.
    Returns:
    pred values, ground truth label
    """
    # create connection
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # initialize a request
    request = predict_pb2.PredictRequest()
    # request.model_spec.name = 'text_clf_model'
    request.model_spec.name = 'mnist_serving'
    # request.model_spec.signature_name = 'prediction'

    # Randomly generate some test data
    # temp_data = numpy.random.randn(10, 3).astype(numpy.float32)
    # x = numpy.array([[0, 0, 0, 0, 0, 0, 0, 622, 27, 153, 1, 1, 1, 1, 1]])
    x = np.random.random_integers(low=0, high=256, size=(28*28, ))

    # request.inputs['input'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(x, shape=x.shape, dtype=tf.float32))

    request.inputs['x'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x, shape=x.shape, dtype=tf.float32))

    # predict
    result = stub.Predict(request, 1)  # 5 seconds
    return result


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return

    print('start inference')
    start = time.time()
    result = do_inference(FLAGS.server)
    print('Result is: ', result)
    print('used time: {}'.format(time.time() - start))

    print('start inference')
    start = time.time()
    result = do_inference(FLAGS.server)
    print('Result is: ', result)
    print('used time: {}'.format(time.time() - start))

    print('start inference')
    start = time.time()
    result = do_inference(FLAGS.server)
    print('Result is: ', result)
    print('used time: {}'.format(time.time() - start))


if __name__ == '__main__':
    tf.app.run()