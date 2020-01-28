import tensorflow as tf

class Discriminator(tf.keras.Model):

  def __init__(self):
    super().__init__()

    # since discriminator is for classification it should be robust, thus, add
    # additional regularization like dropout to prevent from pixel attacks
    self.trainable = False
    self.image_encoder = tf.keras.Sequential([
        # input 5,4,24,84
        # 64,4,32,28
        tf.keras.layers.Conv3D(64, (1, 3, 3), (1, 3, 3), input_shape=(5, 4, 96, 84), data_format='channels_first'),
     #   tf.keras.layers.BatchNormalization(axis=1),
        #tf.keras.layers.LeakyReLU(),
        # 128,2,16,14
        # conv with stride
        tf.keras.layers.Conv3D(128, (1, 2, 2), (1, 2, 2), 'same', data_format='channels_first'),
      #  tf.keras.layers.BatchNormalization(axis=1),
       # tf.keras.layers.LeakyReLU(),
        # 256,2,8,7
        tf.keras.layers.Conv3D(256, (2, 2, 2), (2, 2, 2), 'same', data_format='channels_first'),
       # tf.keras.layers.BatchNormalization(axis=1),
      #  tf.keras.layers.LeakyReLU(),
        # #
        # # # tf.keras.layers.Dropout(0.3),
        # # # # flatten + hidden layer
        tf.keras.layers.Flatten(),
        #         tf.keras.layers.Dense(64),
        #         tf.keras.layers.BatchNormalization(axis=1),
        #         tf.keras.layers.ReLU(),
        #         tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

  def call(self, images, training):
    return self.image_encoder(images)

class Generator(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.noise_decoder = tf.keras.Sequential([
        # flat
        tf.keras.layers.Dense(5*2*8*7,input_shape=(100,)),
       # tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.ReLU(),
        tf.keras.layers.Reshape([5,2,8,7]),
       # # conv without stride (7x7)
        tf.keras.layers.Conv3D(512, 5, 1, 'same',data_format='channels_first'),
        #tf.keras.layers.BatchNormalization(axis=1),
        #tf.keras.layers.ReLU(),
        # # # t_conv with stride (14x14)
        tf.keras.layers.Conv3DTranspose(256, (2, 2, 1), (2, 2, 1),  'same',data_format='channels_first'),
        #tf.keras.layers.BatchNormalization(axis=1),
        #tf.keras.layers.ReLU(),
        #

        tf.keras.layers.Conv3DTranspose(128, (1, 1, 12), (1, 1, 12), 'same', data_format='channels_first'),
        #tf.keras.layers.BatchNormalization(axis=1),
        #tf.keras.layers.ReLU(),

        tf.keras.layers.Conv3D(64, 5, 1, 'same',data_format='channels_first'),
        #tf.keras.layers.BatchNormalization(axis=1),
        #tf.keras.layers.ReLU(),
        #
        tf.keras.layers.Conv3DTranspose(32, (1, 2, 1), (1, 2, 1), 'same', data_format='channels_first'),
        #tf.keras.layers.BatchNormalization(axis=1),
       # tf.keras.layers.ReLU(),
        #
        tf.keras.layers.Conv3DTranspose(16, (1, 3, 1), (1, 3, 1), 'same', data_format='channels_first'),
        #tf.keras.layers.BatchNormalization(axis=1),
      #  tf.keras.layers.ReLU(),
        # #
        tf.keras.layers.Conv3D(5, 5, 1, 'same',data_format='channels_first')
    ])

  def call(self, noise, training):
    return self.noise_decoder(noise)

class NoiseGenerator(tf.keras.layers.Layer):

  def __init__(self, num_classes, distribution_size):
    super().__init__()
    self.distribution_size = distribution_size
    # self.data_distributions = self.add_weight(shape=(num_classes, distribution_size), trainable=True)
    # self.data_distributions = tf.tile(tf.range(0, num_classes, dtype=tf.float32)[:, tf.newaxis], [1, distribution_size])
    # TODO:

  def call(self, inputs):
    # dists = tf.nn.embedding_lookup(self.data_distributions, inputs)
    # dists += tf.random.uniform(tf.shape(dists), -0.35, 0.35)
    # return dists
    # TODO
    return tf.random.uniform([tf.shape(inputs)[0], self.distribution_size])

  def diverse_distributions_loss(self):
    # TODO
    return None