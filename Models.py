import tensorflow as tf

class Discriminator(tf.keras.Model):

  def __init__(self):
    super().__init__()

    # since discriminator is for classification it should be robust, thus, add
    # additional regularization like dropout to prevent from pixel attacks
    self.image_encoder = tf.keras.Sequential([
        # conv with stride (out = 14x14)
        tf.keras.layers.Conv3D(64, 5, 2, 'same',input_shape=(5,4,24,84), data_format='channels_first'),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        #2,2,12
        # conv with steide (out = 7x7)
        # tf.keras.layers.Conv3D(128, 3, 2, 'same'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ReLU(),
        # tf.keras.layers.Dropout(0.3),
        # # flatten + hidden layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        # prediction (LOGITS!)
        tf.keras.layers.Dense(1)
    ])

  def call(self, images, training):
    return self.image_encoder(images)

class Generator(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.noise_decoder = tf.keras.Sequential([
        # flat
        tf.keras.layers.Dense(5*2*12*42),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Reshape([5,2,12,42]),
        # conv without stride (7x7)
        tf.keras.layers.Conv3D(128, 5, 1, 'same',data_format='channels_first'),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU(),
        # # t_conv with stride (14x14)
        tf.keras.layers.Conv3DTranspose(64, 5, 2, 'same',data_format='channels_first'),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU(),
        # # conv without stride (14x14)
        # # tf.keras.layers.Conv3D(32, 5, 1, 'same'),
        # # tf.keras.layers.BatchNormalization(),
        # # tf.keras.layers.ReLU(),
        # # t_conv with stride (28x28)
        # # tf.keras.layers.Conv3DTranspose(32, 5, 2, 'same'),
        # # tf.keras.layers.BatchNormalization(),
        # # tf.keras.layers.ReLU(),
        # # conv without stride
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