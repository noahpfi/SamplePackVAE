import tensorflow as tf
from keras.layers import Conv2D, Dense, Conv2DTranspose, MaxPooling2D, Reshape, Flatten

from audio import to_mel_spectrogram


class IAFVAEModel(tf.keras.Model):
    def __init__(self,
                 params,
                 keep_prob=1.0,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.encoder_conv = []
        for i in range(len(params['CONV_CHANNELS'])):
            if i == 0:
                self.encoder_conv.append(Conv2D(
                    input_dim=1,
                    filters=params['CONV_CHANNELS'][i],
                    kernel_size=(3, 3),
                    activation='elu',
                    padding='same'
                ))
            else:
                self.encoder_conv.append(Conv2D(
                    filters=params['CONV_CHANNELS'][i],
                    kernel_size=(3, 3),
                    activation='elu',
                    padding='same'
                ))
            self.encoder_conv.append(MaxPooling2D(pool_size=params['MAX_POOLING'][i]))

        self.encoder_fc = Dense(units=params['DIM_LATENT'], activation='elu')
        self.encoder_mean = Dense(units=params['DIM_LATENT'])
        self.encoder_logvar = Dense(units=params['DIM_LATENT'])

        self.iaf_flows = []
        for _ in range(params['IAF_FLOW_LENGTH']):
            flow_layers = []
            for _ in range(params['DIM_LATENT']):
                flow_layers.append(Dense(units=params['DIM_AUTOREGRESSIVE_NL'], activation='elu'))
                flow_layers.append(Dense(units=2))
            self.iaf_flows.append(flow_layers)

        deconv_shape = params['DECONV_SHAPE']
        self.decoder_fc = Dense(units=deconv_shape[0][1] * deconv_shape[0][2] * deconv_shape[0][3], activation='elu')
        self.decoder_reshape = Reshape(target_shape=(deconv_shape[0][1], deconv_shape[0][2], deconv_shape[0][3]))

        self.decoder_deconv = []
        for i in range(len(deconv_shape) - 1):
            self.decoder_deconv.append(Conv2DTranspose(
                filters=deconv_shape[i + 1][3],
                kernel_size=(3, 3),
                activation='elu',
                padding='same'
            ))
        self.decoder_output = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='elu', padding='same')

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for conv_layer in self.encoder_conv:
            x = conv_layer(x)

        x = Flatten()(x)
        x = self.encoder_fc(x)

        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        z = self.sample_z(mean, logvar)

        for flow_layers in self.iaf_flows:
            m, s = self.compute_m_s(z, flow_layers)
            z = self.transform_z(z, m, s)

        x = self.decoder_fc(z)
        x = self.decoder_reshape(x)

        for deconv_layer in self.decoder_deconv:
            x = deconv_layer(x)

        reconstructed_output = self.decoder_output(x)
        return reconstructed_output, mean, logvar

    def sample_z(self, mean, logvar):
        epsilon = tf.random.normal(shape=mean.shape)
        z = mean + tf.exp(0.5 * logvar) * epsilon
        return z

    def compute_m_s(self, z, flow_layers):
        m, s = [], []
        for i, layer in enumerate(flow_layers):
            if i == 0:
                input_ = z
            else:
                input_ = tf.concat([z] + m[:i], axis=-1)
            output = layer(input_)
            m.append(output[:, 0])
            s.append(tf.nn.softplus(output[:, 1]))
        return m, s

    def transform_z(self, z, m, s):
        transformed_z = z
        for i in range(len(m)):
            u = (transformed_z - m[i]) / s[i]
            transformed_z = u * tf.exp(-s[i]) + m[i]
        return transformed_z

    def embedding_from_audio(self, audio, params):
        spec = to_mel_spectrogram(audio, params)
        # TODO
        # return self.embedding_from_feature(spec)[0]
        pass

    def embedding_from_feature(self, spec):
        # TODO
        pass