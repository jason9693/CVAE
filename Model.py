import tensorflow as tf
import tensorflow.contrib.layers as layers
import ModelUtils as mu
import tf_utils
import params as par

class Model:

    #TODO : replace model to CNN

    def __init__(self,labels=None,input_shape = None,dropout=None):
        self.labels =labels
        self.input_dims = input_shape
        self.dropout = dropout

    def zy_concat(self,z,y,output_dim = 10):
        # W = tf.get_variable(shape=[y.shape[1],z.shape[1]])
        # b = tf.get_variable(shape=z.shape[1])
        #return tf.concat([z,mu.affine(y,y.shape[1],z.shape[1],0,'concat',activation=tf.nn.tanh)],1)
        # y = tf.argmax(y,1)
        # y = tf.cast(y,tf.float32)
        # y = tf.expand_dims(y,axis=1)

        return tf.concat([z,y],1)


    def encoder(self,X,name='encoder',sess=None):
        print('encoder')
        affine_iter = 0
        with tf.variable_scope(name) and tf.device(tf_utils.gpu_mode(par.gpu_mode)):
            #encoder_x = tf.reshape(X,[1,-1])
            encoder_x = tf.layers.flatten(X)

            h0 = mu.affine(encoder_x,encoder_x.shape[1],encoder_x.shape[1] // 2,affine_iter,name,activation=tf.nn.tanh)
            affine_iter += 1
            h1 = mu.affine(h0,h0.shape[1],h0.shape[1],affine_iter,name,activation=tf.nn.leaky_relu)
            affine_iter += 1
            #lay1 = tf.reshape(h0, shape=[tf.shape(encoder_x)[0], 32, 32, 1], name="reshape1_encoder")
            #
            # conv0 = tf.layers.conv2d(
            #     inputs=lay1,
            #     filters=8,
            #     kernel_size=[7, 7],
            #     strides=[2, 2],
            #     name="conv0_encoder",
            #     activation=tf.nn.relu,
            #     padding='same'
            # )
            # # print('conv0:',conv0.shape)
            # pool0 = tf.layers.max_pooling2d(
            #     inputs=conv0,
            #     pool_size=2,
            #     strides=1,
            #     padding='same',
            #     name="max_pool0_encoder"
            # )
            #
            # cycle = 3
            # conv = pool0
            # for i in range(cycle):
            #     for j in range(i + 3):
            #         conv = mu.residual(i, j, conv, self.dropout,name)
            #
            #
            # ap1 = tf.layers.average_pooling2d(
            #     inputs=conv,
            #     pool_size=2,
            #     strides=1,
            #     padding='valid',
            #     name="max_pool1_encoder"
            # )
            #
            # ap_flat = tf.layers.flatten(ap1)

            self.fc_layer = mu.affine(h1,h1.shape[1],self.labels * 2,affine_iter,name)
            self.lay_out = self.fc_layer#tf.nn.relu(self.fc_layer)

            mean = self.lay_out[:,:self.labels]
            std_dev = 1e-6 + tf.nn.softplus(self.lay_out[:,self.labels:])

        return mean,std_dev


    def decoder(self,Z,name='decoder',sess=None):
        affine_iter = 0
        with tf.variable_scope(name) and tf.device(tf_utils.gpu_mode(par.gpu_mode)):
            self.Z = Z

            h0 = mu.affine(self.Z, Z.shape[1], Z.shape[1] * 2, affine_iter,name,activation=tf.nn.elu)
            affine_iter += 1
            h1 = mu.affine(h0, h0.shape[1], h0.shape[1], affine_iter, name, activation=tf.nn.relu)
            affine_iter += 1
            #
            # lay1 = tf.reshape(h0, shape=[tf.shape(self.Z)[0], 32, 32, 1], name="reshape1_decoder")
            #
            # conv0 = tf.layers.conv2d(
            #     inputs=lay1,
            #     filters=8,
            #     kernel_size=[7, 7],
            #     strides=[2, 2],
            #     name="conv0_decoder",
            #     activation=tf.nn.relu,
            #     padding='same'
            # )
            # # print('conv0:',conv0.shape)
            # pool0 = tf.layers.max_pooling2d(
            #     inputs=conv0,
            #     pool_size=2,
            #     strides=1,
            #     padding='same',
            #     name="max_pool0_decoder"
            # )
            #
            # cycle = 3
            # conv = pool0
            # for i in range(cycle):
            #     for j in range(i + 3):
            #         conv = mu.residual(i, j, conv, self.dropout,name)
            #
            # ap1 = tf.layers.average_pooling2d(
            #     inputs=conv,
            #     pool_size=2,
            #     strides=1,
            #     padding='valid',
            #     name="max_pool1_decoder"
            # )
            #
            # ap_flat = tf.layers.flatten(ap1)

            self.affined_decoder = mu.affine(h1,h1.shape[1],self.input_dims[1],affine_iter,name)
            self.out =tf.nn.sigmoid(self.affined_decoder) #(tf.tanh(self.affined_decoder)+1) / 2
        return self.out


