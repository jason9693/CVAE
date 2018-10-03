import tensorflow as tf


def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    try:
        plus = x * tf.cast(x >= 0, tf.float32)
        minus = x * tf.cast(x < 0, tf.float32) * alpha
    except:
        plus = x * tf.cast(x >= 0, tf.float64)
        minus = x * tf.cast(x < 0, tf.float64) * alpha
    return plus + minus

def gpu_mode(gpu: bool):
    device_string = 'device:{}:0'
    if gpu:
        device_string = device_string.format('GPU')
    else:
        device_string = device_string.format('CPU')
    return device_string