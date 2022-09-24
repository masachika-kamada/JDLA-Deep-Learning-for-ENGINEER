import numpy as np


@tf.function
def update_discriminator(self, noize, real_data):
    """
    :param self:
    :param noize: Gの入力乱数ベクトル[バッチサイズ:x, ベクトル長:100]
    :param real_data: Dの入力に用いる学習データ[バッチサイズ:x, H:64, W:64, RGB:3]
    :return:
    """
    fake_data = self.G(noize)
    with tf.GradientTape() as d_tape:
        real_pred = self.D(real_data)
        fake_pred = self.D(fake_data)
        real_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(real_pred), real_pred
        )
        real_loss = tf.math.reduce_mean(real_loss)
        fake_loss = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(fake_pred), fake_pred
        )
        adv_loss = real_loss + fake_loss
    d_grad = d_tape.gradient(adv_loss, source=self.D.trainable_cariables)
    self.d_optimizer.apply_gradients(zip(d_grad, self.D.trainable_cariables))


def train(self, train_data):
    """GANの学習を行う関数
    :param self:
    :param train_data: 学習データ[データ数:x>バッチサイズ, H:64, W:64, RGB:3]
    :return:
    """
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    for _ in range(100):
        shuffled_train_data = train_dataset.shuffle(len(train_data[0]))
        epoch_train_data = shuffled_train_data.batch(128)
        for batch_train_data in epoch_train_data:
            noise = tf.random.uniform((128, 100))
            self.update_discriminator(noise, batch_train_data)
            self.update_generator(noise)


class DCGAN_Generator(object):
    def __init__(self, batch_size, noise_dim=100):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.w_init = RandomNormal(mean=0.0, stddev=0.02)

    def build(self):
        noise = Input(batch_shape=(self.batch_size, self.noise_dim))
        densed = Dense(4 * 4 * 1024, "relu", kernel_initializer=self.w_init)(noise)
        densed = BatchNormalization()(densed)
        reshaped = Reshape((4, 4, 1024))(densed)
        conv1 = Conv2DTranspose(512, (5, 5), (2, 2), "same", activation="relu",
                                kernel_initializer=self.w_init)(reshaped)
        conv1 = BatchNormalization()(conv1)
        conv2 = Conv2DTranspose(256, (5, 5), (2, 2), "same", activation="relu",
                                kernel_initializer=self.w_init)(conv1)
        conv2 = BatchNormalization()(conv2)
        conv3 = Conv2DTranspose(128, (5, 5), (2, 2), "same", activation="relu",
                                kernel_initializer=self.w_init)(conv2)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2DTranspose(64, (5, 5), (2, 2), "same", activation="relu",
                                kernel_initializer=self.w_init)(conv3)
        generator = Model(inputs=noise, outputs=conv4)
        return generator

