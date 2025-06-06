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
    """
    Convolutional Block Attention Module (CBAM)

    Paper: "CBAM: Convolutional Block Attention Module" by Woo et al.

    Args:
        reduction_ratio (int): Channel reduction ratio for MLP. Default: 16
        kernel_size (int): Kernel size for spatial attention convolution. Default: 7
        use_bias (bool): Whether to use bias in dense layers. Default: True
        **kwargs: Additional keyword arguments for Layer
    """

    def __init__(self, reduction_ratio=16, kernel_size=7, use_bias=True, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        # Validate inputs
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd for proper padding")

    def build(self, input_shape):
        """Build the CBAM layers based on input shape"""
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, height, width, channels), got {len(input_shape)}D"
            )

        self.channels = input_shape[-1]

        if self.channels < self.reduction_ratio:
            raise ValueError(
                f"Number of channels ({self.channels}) must be >= reduction_ratio ({self.reduction_ratio})"
            )

        # Channel Attention Components
        self.mlp_dense1 = Dense(
            units=self.channels // self.reduction_ratio,
            activation="relu",
            use_bias=self.use_bias,
            name="channel_mlp_1",
        )
        self.mlp_dense2 = Dense(
            units=self.channels, use_bias=self.use_bias, name="channel_mlp_2"
        )

        # Spatial Attention Components
        self.spatial_conv = Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            use_bias=False,  # Standard CBAM doesn't use bias here
            name="spatial_conv",
        )

        super(CBAMBlock, self).build(input_shape)

    def channel_attention(self, input_feature):
        """
        Channel Attention Module

        Args:
            input_feature: Input tensor (B, H, W, C)

        Returns:
            channel_attention_map: Channel attention weights (B, 1, 1, C)
        """
        # Global Average Pooling: (B, H, W, C) -> (B, C)
        avg_pool = GlobalAveragePooling2D()(input_feature)

        # Global Max Pooling: (B, H, W, C) -> (B, C)
        max_pool = GlobalMaxPooling2D()(input_feature)

        # Shared MLP processing
        avg_out = self.mlp_dense2(self.mlp_dense1(avg_pool))  # (B, C)
        max_out = self.mlp_dense2(self.mlp_dense1(max_pool))  # (B, C)

        # Element-wise addition
        channel_attention = Add()([avg_out, max_out])  # (B, C)

        # Sigmoid activation
        channel_attention = Activation("sigmoid")(channel_attention)  # (B, C)

        # Reshape to broadcast with input: (B, C) -> (B, 1, 1, C)
        channel_attention = Reshape((1, 1, self.channels))(channel_attention)

        return channel_attention

    def spatial_attention(self, input_feature):
        """
        Spatial Attention Module

        Args:
            input_feature: Input tensor (B, H, W, C)

        Returns:
            spatial_attention_map: Spatial attention weights (B, H, W, 1)
        """
        # Channel-wise average pooling: (B, H, W, C) -> (B, H, W, 1)
        avg_pool = ops.mean(input_feature, axis=-1, keepdims=True)

        # Channel-wise max pooling: (B, H, W, C) -> (B, H, W, 1)
        max_pool = ops.max(input_feature, axis=-1, keepdims=True)

        # Concatenate along channel dimension: (B, H, W, 2)
        concat_feature = Concatenate(axis=-1)([avg_pool, max_pool])

        # 7x7 convolution followed by sigmoid
        spatial_attention = self.spatial_conv(concat_feature)  # (B, H, W, 1)
        spatial_attention = Activation("sigmoid")(spatial_attention)  # (B, H, W, 1)

        return spatial_attention

    def call(self, inputs, training=None):
        """
        Forward pass of CBAM

        Args:
            inputs: Input tensor (B, H, W, C)
            training: Training mode flag

        Returns:
            refined_feature: CBAM enhanced feature map (B, H, W, C)
        """
        # Step 1: Channel Attention
        channel_attention_map = self.channel_attention(inputs)

        # Apply channel attention
        channel_refined_feature = Multiply()([inputs, channel_attention_map])

        # Step 2: Spatial Attention
        spatial_attention_map = self.spatial_attention(channel_refined_feature)

        # Apply spatial attention
        final_refined_feature = Multiply()(
            [channel_refined_feature, spatial_attention_map]
        )

        return final_refined_feature

    def get_config(self):
        """Return layer configuration for serialization"""
        config = super(CBAMBlock, self).get_config()
        config.update(
            {
                "reduction_ratio": self.reduction_ratio,
                "kernel_size": self.kernel_size,
                "use_bias": self.use_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration"""
        return cls(**config)
