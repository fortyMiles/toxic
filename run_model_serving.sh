#!/usr/bin/env bash

# To stop: docker stop simple && docker rm simple

docker run -it -p 9000:9000 --name keras_model_test --entrypoint=/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server -v $(pwd)/bank_clf_export/:/models/ zekka/tensorflow-serving-devel:bazel-py2-py3 --port=9000 --model_name=text_clf_model --model_base_path=/models

# the --model_name is name serving
# the --name is the name of container
