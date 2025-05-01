from keras_core import ops
from keras_core.layers import (
    Layer,
    Dense,
    Reshape,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Activation,
    Add,
    Multiply,
    Concatenate,
    Conv2D,
)
from keras_core.utils import register_keras_serializable


@register_keras_serializable()
class CBAMBlock(Layer):
    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel = input_shape[-1]

        self.shared_dense_1 = Dense(
            self.channel // self.ratio, activation="relu", use_bias=True
        )
        self.shared_dense_2 = Dense(self.channel, use_bias=True)
        self.reshape = Reshape((1, 1, self.channel))

        self.concat = Concatenate(axis=-1)
        self.conv2d = Conv2D(
            filters=1, kernel_size=7, strides=1, padding="same", activation="sigmoid"
        )

    def call(self, input_feature):
        avg_pool = GlobalAveragePooling2D()(input_feature)
        max_pool = GlobalMaxPooling2D()(input_feature)

        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
        cbam_feature = Activation("sigmoid")(Add()([avg_out, max_out]))
        cbam_feature = self.reshape(cbam_feature)
        channel_refined = Multiply()([input_feature, cbam_feature])

        avg_pool = ops.mean(channel_refined, axis=3, keepdims=True)
        max_pool = ops.max(channel_refined, axis=3, keepdims=True)
        concat = self.concat([avg_pool, max_pool])
        cbam_feature = self.conv2d(concat)
        refined_feature = Multiply()([channel_refined, cbam_feature])

        return refined_feature
