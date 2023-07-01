from keras.optimizers import Adam

from sample_pack_generator import *
from model import IAFVAEModel
import tensorflow as tf


if __name__ == '__main__':
    # generator = SamplePackGenerator('')
    model = IAFVAEModel(util.load_params('params.json'))

    model.compile(
        loss=tf.losses.kl_divergence,
        optimizer=Adam(
            learning_rate=1e-3,
            epsilon=1e-4
        )
    )

    model.fit(None, None, epochs=10, batch_size=1)
