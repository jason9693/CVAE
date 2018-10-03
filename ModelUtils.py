import tensorflow as tf
import tensorflow.contrib.layers as layers


def residual(cycle,layer,tensor,dropout,net = 'encoder'):
        tensor = tf.layers.dropout(tensor,rate=dropout)
        conv = tf.layers.conv2d(
            inputs=tensor,
            filters= 2 ** (3+cycle),
            kernel_size=[3, 3],
            strides=[1, 1],
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_res1_'+net+str(cycle)+'_'+str(layer),
            activation=tf.nn.relu,
            padding='same',
        )
        conv = tf.layers.conv2d(
            inputs=conv,
            filters= 2 ** (3+cycle),
            kernel_size=[3, 3],
            strides=[1, 1],
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_res2_'+net+str(cycle)+'_'+str(layer),
            padding='same',
        )
        try:
            return conv + tensor
        except ValueError:
            tr_tens = tf.transpose(tensor, perm=[3, 0, 1, 2])
            tr_conv = tf.transpose(conv, perm=[3, 0, 1, 2])

            return tf.transpose(
                tf.concat([tr_tens, tr_tens], axis=0) + tr_conv, perm=[1, 2, 3, 0]
            )
        #return conv + tensor

def affine(tensor,input_shape,output_shape,num=0,net='encoder',activation=None):
    w = tf.get_variable(
        "W_"+net+str(num),
        shape=[input_shape, output_shape],
        dtype=tf.float32,
        initializer=layers.xavier_initializer()
    )
    b = tf.get_variable(
        "b_"+net+str(num),
        shape=[output_shape],
        dtype=tf.float32,
        initializer=layers.xavier_initializer()
    )

    out = tf.matmul(tensor,w) + b
    if activation is None:
        return out
    else:
        return activation(out)

