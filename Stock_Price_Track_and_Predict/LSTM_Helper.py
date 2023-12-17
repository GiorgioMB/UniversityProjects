from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
class ParabolicVGPLayer(Layer):
    def __init__(self, amplitude=None, length_scale=None, **kwargs):
        super().__init__(**kwargs)
        self._amplitude = amplitude
        self._length_scale = length_scale

    @property
    def kernel(self):
        return tfp.math.psd_kernels.Parabolic(
            amplitude=self._amplitude, length_scale=self._length_scale)
