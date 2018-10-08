import tensorflow as tf
import Model as model
import numpy as np
class VAE:
    def __init__(self,input_shape,label_size=10,z_dim = 16,learning_rate=1e-2,sess=None , path=None, alpha = 1):
        self.labels = z_dim
        self.learning_rate = learning_rate
        self.y_depth = label_size
        self.input_shape = input_shape
        self.alpha = alpha
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=(),name="batch_size")
        self.shape = [None] + input_shape
        self.model = model.Model(labels=self.labels,input_shape = self.shape)
        self.__build_net__()
        self.sess = sess
        if path == None:
            self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, path)


    def __build_net__(self):
      with tf.variable_scope('vae',reuse=tf.AUTO_REUSE) :
        self.dropout = tf.placeholder(dtype=tf.float16,shape=(),name="dropout")
        self.X = tf.placeholder(dtype=tf.float32,shape=self.shape)
        self.y = tf.placeholder(dtype= tf.uint8,shape=[None,])

        self.s = tf.one_hot(self.y,depth=self.y_depth,axis=-1)

        z_mean, z_std_dev = self.model.encoder(self.X, name="train_encoder")
        self.KLD = tf.reduce_mean(
            tf.reduce_sum(
                #0.5*
                tf.square(z_std_dev) + tf.square(z_mean) - 1 - tf.log(tf.square(z_std_dev))
                #z_var + tf.square(z_mean) -1 - tf.log(z_var+ 1e-8)
                , axis=1
            ),
            axis=0
        )

        self.Z = tf.placeholder(dtype=tf.float32,shape=[None, int(self.labels)])
        tf.set_random_seed(777)
        dec_z = z_std_dev * tf.random_normal([self.batch_size, int(self.labels)],0,1,dtype=tf.float32) + z_mean

        self.pred_X = self.model.decoder(self.model.zy_concat(self.Z,self.s))   #using inference time
        decoded_X =self.model.decoder(self.model.zy_concat(dec_z,self.s))       #using training time

        self.likelihood = tf.reduce_mean(
            tf.reduce_sum(
                self.X * tf.log(decoded_X + 1e-8) + (1 - self.X) * (tf.log(1e-8 + (1 - decoded_X)))
                , axis=1
            )
        )
        self.loss = self.KLD - self.likelihood  # reverse sequence for minimize. ( argmax -> argmin )
        self.GSNN = - self.likelihood

        self.loss = self.alpha * self.loss + (1-self.alpha) * self.GSNN
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self,X,y,dropout=0):
        return self.sess.run([self.optim,self.loss,self.KLD,self.likelihood],
                        feed_dict={
                                   self.X:X,
                                   self.y:y,
                                   self.batch_size: len(y),
                                   self.dropout:dropout}
                        )

    def predict(self, Z, y):
        return self.sess.run(self.pred_X,feed_dict={
            self.Z:Z, self.y:y ,self.dropout:0, self.batch_size: len(y)
        })