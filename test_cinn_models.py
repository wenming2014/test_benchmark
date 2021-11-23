import os
import logging
logger = logging.getLogger(__name__)
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s : %(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(syslog)
import numpy as np
import time
import tensorflow as tf
import argparse
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.client import device_lib
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def infer(model, inputs, num_epoches):
    w_epoches, i_epoches = num_epoches
    start = time.perf_counter()
    for i in range(w_epoches):
        model(inputs)
        logger.info("Warmup-%d: %.3f ms" % (i, (time.perf_counter() - start) * 1000))
    logger.info("Warmup performance:[num_epoches:%d|elapsed_time: %.3f ms]" % (
        w_epoches, (time.perf_counter() - start) * 1000 / w_epoches))
    infer_start = time.perf_counter()
    for i in range(i_epoches):
        model(inputs)
    logger.info("Infer performance:[num_epoches:%d|elapsed_time: %.3f ms]" % (
        i_epoches, (time.perf_counter() - infer_start) * 1000 / i_epoches))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_gpu", dest="disable_gpu",  action='store_true')
    parser.add_argument("--use_xla", dest="use_xla",  action='store_true')
    parser.add_argument("--debug_xla", dest="debug_xla",  action='store_true')
    parser.add_argument("--model_name", type=str, choices=['ResNet18', 'ResNet50', 'MobileNetV1', 'MobileNetV2', 'EfficientNet', 'SqueezeNet', 'FaceDet'],
            default='ResNet18', help="the test model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epoches", nargs=2, type=int, default=[10, 1000],
            help="the epoches of warmup and inference seperately")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info("Args setting: {}".format(args))
    if args.use_xla:
        os.environ['TF_CPP_VMODULE'] = 'xla_compilation_cache=1'
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        if args.debug_xla:
            xla_dump_path = os.path.join("./xla_debug", args.model_name)
            os.environ['XLA_FLAGS'] = '--xla_dump_to={} --xla_dump_hlo_as_html --xla_dump_hlo_pass_re=.*'.format(xla_dump_path)
    saved_path = os.path.join("./cinn_tf_pb/", args.model_name)
    print("saved_path", saved_path)
    run_device = '/cpu:0' if args.disable_gpu else '/gpu:0'
    with tf.device(run_device):
        logger.info("Loading model-{} from: {}".format(args.model_name, saved_path))
        input_data = np.random.random([args.batch_size, 3, 224, 224]).astype("float32")
        # input_data = np.random.random([args.batch_size, 3, 227, 227]).astype("float32")
        # input_data = np.random.random([args.batch_size, 3, 240, 320]).astype("float32")
        loaded = tf.saved_model.load(saved_path, tags=[tag_constants.SERVING])
        fn = loaded.signatures["serving_default"]
        if args.use_xla:
            fn = tf.function(fn, experimental_compile=True)
        infer(fn, tf.cast(input_data, dtype="float"), args.epoches)
