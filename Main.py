import tensorflow as tf
import os
import matplotlib.pyplot as plt
import os.path
import pypianoroll as piano
import numpy as np
from tensorflow_core.python.keras.utils import plot_model
import pydot

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


from Models import Generator, Discriminator, NoiseGenerator

allow_memory_growth()

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


def show_images(images):
    fig = plt.figure(figsize=(12, 12 * 10))

    for i in range(images.shape[0]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()


def ds(images, labels, buffer_size, batch_size):
    images = images.reshape(images.shape[0], 5, 4, 96, 84).astype('float32')
    images = (images - 0.5) / 0.5
    labels = labels.astype('int32')
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)) \
        .shuffle(buffer_size) \
        .batch(batch_size)
    return dataset


def train(data, labels, train_step, gen_step, epochs, batch_size):
    train_ds = ds(data, labels, 1, batch_size)
    for epoch in range(epochs):

        for images, labels in train_ds:
            train_step(images, labels)

        print('Epoch {0}/{1}'.format(epoch, epochs))

        images = gen_step([0, 1])
    #  show_images(images)


def divide_into_bars(track, resolution, length, values):
    track[track > 0] = 1
    if track.size:
        bars = np.vsplit(track, track.shape[0] / resolution)
    else:
        empty_track = np.zeros((length, 84))
        bars = np.vsplit(empty_track, empty_track.shape[0] / resolution)
    return np.asarray(divide_into_phrase((bars)))


def divide_into_phrase(bars):
    phrases = []
    phrase = []
    n = 0
    for i in range(len(bars)):
        phrase.append(bars[i])
        n += 1
        if (n % 4 == 0):
            phrases.append(np.vstack(phrase))
            phrase = []
            n = 0
    units = []
    unit = []
    n = 0
    for i in range(len(phrases)):
        unit.append(phrases[i])
        n += 1
        if (n % 4 == 0):
            units.append(np.array(unit))
            unit = []
            n = 0
    return np.array(units)


def map_instrument_name(program, isdrum):
    if program in range(1, 9):
        return "Piano"
    if program in range(25, 33):
        return "Guitar"
    if program in range(33, 41):
        return "Bass"
    if program in range(41, 49):
        return "Strings"
    if isdrum == True:
        return "Drums"
    return "XXX"


def load_data(folder):
    data = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in [f for f in filenames if f.endswith(".npz")]:
            pianoroll = piano.Multitrack(os.path.join(dirpath, filename))
            pianoroll.remove_empty_tracks()
            duration = max(roll.pianoroll.shape[0] for roll in pianoroll.tracks)
            values = max(roll.pianoroll.shape[1] for roll in pianoroll.tracks)
            multitrack_bar = []
            i = 0
            exit = 0
            lineup = {'Guitar': 0, 'Piano': 0, 'Bass': 0, 'Strings': 0, 'Drums': 0}
            for track in sorted(pianoroll.tracks, key=lambda x: map_instrument_name(x.program, x.is_drum)):
                name = map_instrument_name(track.program, track.is_drum)
                if name not in ['Guitar', 'Piano', 'Bass', 'Strings', 'Drums']:
                    continue
                if lineup[name] == 1:
                    continue
                if len(pianoroll.tracks) < 5:
                    exit = True
                    break
                lineup[name] = 1
                i += 1
                # print(track.pianoroll.shape)
                phrases = divide_into_bars(track.pianoroll[:, 0:84], pianoroll.beat_resolution, duration, values)
                multitrack_bar.append(phrases)
                if i == 5:
                    break
            if len(multitrack_bar) != 5:
                break
            if exit == True:
                break
            multitrack_bar = np.asarray(multitrack_bar)
            multitrack_bar = multitrack_bar.transpose((1, 0, 2, 3, 4))
            data.append(multitrack_bar)
    data = np.vstack(data)
    return data


# rolls = load_data('test2')
# fig = piano.plot(rolls[1])
# plt.show()
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# data = load_data("test6")
#
# np.save('datatest',data)
# print(np.asarray(data).shape)
# print(sum(data))
data = load_data('C')
np.save("train_sample",data[2])
print(data.shape)
labels = np.ones(np.asarray(data).shape[0])
generator = Generator()
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
generator.build((256,100))
print(generator.summary())
plot_model(generator.noise_decoder, to_file='gen_plot.png', show_shapes=True, show_layer_names=True,dpi=1000)
discriminator = Discriminator()
discriminator.build((256, 5, 4, 96, 84))
print(discriminator.summary())

plot_model(discriminator.image_encoder, to_file='dis_plot.png', show_shapes=True, show_layer_names=True,dpi=1000)

# noise = NoiseGenerator(2, 100)
# d_optim = tf.optimizers.Adam(1e-4)
# g_optim = tf.optimizers.Adam(1e-4)
#
# train_step = train_step_template(
#     generator=generator,
#     discriminator=discriminator,
#     noise=noise,
#     d_optim=d_optim,
#     g_optim=g_optim,
#     d_loss_f=w_discriminator_loss,
#     g_loss_f=w_generator_loss,
# )
#
# gen_step = gen_step_template(
#     generator=generator,
#     noise=noise
# )
#
# train(
#     data=data,
#     labels=labels,
#     train_step=train_step,
#     gen_step=gen_step,
#     epochs=30,
#     batch_size=256
# )
