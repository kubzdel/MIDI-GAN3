import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import os.path
import pypianoroll as piano
import numpy as np

from Models import Generator, Discriminator, NoiseGenerator


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)




bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def min_max_discriminator_loss(real_out, gen_out):
    real_loss = bce(tf.ones_like(real_out), real_out)
    gen_loss = bce(tf.zeros_like(gen_out), gen_out)
    return real_loss + gen_loss


def min_max_generator_loss(gen_out):
    return - min_max_discriminator_loss(tf.ones_like(gen_out), gen_out)


def w_discriminator_loss(real_out, gen_out):
    return - (tf.reduce_mean(real_out) - tf.reduce_mean(gen_out))


def w_generator_loss(gen_out):
    return - tf.reduce_mean(gen_out)


def train_step_template(generator, discriminator, noise, d_optim, g_optim, d_loss_f, g_loss_f):
    @tf.function
    def _train_step_template(images, labels):
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            real_out = discriminator(images, True)
            gen_out = discriminator(generator(noise(labels), True), True)

            d_loss = d_loss_f(real_out, gen_out)
            g_loss = g_loss_f(gen_out)

        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables + noise.trainable_variables)

        d_optim.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        g_optim.apply_gradients(zip(g_grads, generator.trainable_variables + noise.trainable_variables))

    return _train_step_template

def gen_step_template(generator, noise):

  @tf.function
  def _gen_step_template(labels):
    return tf.clip_by_value(generator(noise(labels), False), -1, 1)

  return _gen_step_template

def ds(images, labels, buffer_size, batch_size):
  images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
  images = (images - 127.5) / 127.5
  labels = labels.astype('int32')
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))\
    .shuffle(buffer_size)\
    .batch(batch_size)
  return dataset

def show_images(images):
  fig = plt.figure(figsize=(12, 12 * 10))

  for i in range(images.shape[0]):
      plt.subplot(10, 10, i+1)
      plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.show()


def train(train_step, gen_step, epochs, batch_size):
    train_data, _ = tf.keras.datasets.mnist.load_data()
    train_ds = ds(*train_data, 60000, batch_size)

    for epoch in range(epochs):

        for images, labels in train_ds:
            train_step(images, labels)

        print('Epoch {0}/{1}'.format(epoch, epochs))

        images = gen_step([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        show_images(images)


def divide_into_bars(track,resolution,length,values):
    if track.size:
        bars = np.vsplit(track,track.shape[0]/resolution)
    else:
        empty_track = np.zeros((length,values))
        bars = np.vsplit(empty_track,empty_track.shape[0]/resolution)
    return bars

def load_data(folder):
    data = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in [f for f in filenames if f.endswith(".npz")]:
            pianoroll = piano.Multitrack(os.path.join(dirpath, filename))
            duration = max(roll.pianoroll.shape[0] for roll in pianoroll.tracks)
            values =  max(roll.pianoroll.shape[1] for roll in pianoroll.tracks)
            multitrack_bar = []
            for track in sorted(pianoroll.tracks, key=lambda x: x.name):
                #print(track.pianoroll.shape)
                multitrack_bar.append(divide_into_bars(track.pianoroll,pianoroll.beat_resolution,duration,values))
            multitrack_bar = np.asarray(multitrack_bar)
            multitrack_bar = multitrack_bar.transpose((1,0,2,3))
            data.extend(multitrack_bar)
    return np.asarray(data)
# rolls = load_data('test2')
# fig = piano.plot(rolls[1])
# plt.show()
d = load_data("test")
print(np.asarray(d).shape)



