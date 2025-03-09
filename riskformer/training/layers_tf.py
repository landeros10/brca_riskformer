
# tensorflow version
'''
Created June 2023
author: landeros10

Lee Laboratory
Center for Systems Biology
Massachusetts General Hospital
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np
from einops import rearrange, reduce
from math import ceil
import random
from os.path import join

# BACKGROUND_TILE_FILE = join(data_dir, "background_tile.npy")
# BACKGROUND_TILE = tf.constant(np.load(BACKGROUND_TILE_FILE))

# Helper function to initialize weights with truncated normal distribution
def trunc_normal_(shape, mean=0., std=1., a=-2., b=2.):
    truncated = tf.random.truncated_normal(shape, mean, std)
    return tf.clip_by_value(truncated, a, b)


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm_layer=None):
    if pool is None:
        return tensor, hw_shape

    revert_shape = False
    if tf.rank(tensor) == 3:
        revert_shape = True
        tensor = tensor[:, tf.newaxis, ...]

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    tensor_shape = tf.shape(tensor)
    B, N, L, C = tensor_shape[0], tensor_shape[1] , tensor_shape[2] , tensor_shape[3]
    H, W = hw_shape
    tensor = tf.reshape(tensor, [B, L, N*C])
    tensor = tf.reshape(tensor, [B, H, W, N*C])
    tensor = pool(tensor)

    hw_shape = (tf.shape(tensor)[1], tf.shape(tensor)[2])
    L_pooled = tf.shape(tensor)[1] * tf.shape(tensor)[2]

    tensor = tf.reshape(tensor, [B, L_pooled, N*C])
    tensor = tf.reshape(tensor, [B, N, L_pooled, C])
    if has_cls_embed:
        tensor = tf.concat([cls_tok, tensor], axis=2)
    if norm_layer is not None:
        tensor = norm_layer(tensor)

    if revert_shape:
        tensor = tensor[:, 0, ...]
    return tensor, hw_shape


def cal_rel_pos_spatial(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = tf.cast(max(k_h / q_h, 1.0), tf.float32)
    k_h_ratio = tf.cast(max(q_h / k_h, 1.0), tf.float32)
    dist_h = (
        tf.cast(tf.range(q_h), tf.float32)[:, None] * q_h_ratio -
        tf.cast(tf.range(k_h), tf.float32)[None, :] * k_h_ratio
    )
    dist_h += tf.cast(k_h - 1, tf.float32) * k_h_ratio

    q_w_ratio = tf.cast(max(k_w / q_w, 1.0), tf.float32)
    k_w_ratio = tf.cast(max(q_w / k_w, 1.0), tf.float32)
    dist_w = (
        tf.cast(tf.range(q_w), tf.float32)[:, None] * q_w_ratio -
        tf.cast(tf.range(k_w), tf.float32)[None, :] * k_w_ratio
    )
    dist_w += tf.cast(k_w - 1, tf.float32) * k_w_ratio

    Rh = tf.gather(rel_pos_h, tf.cast(dist_h, tf.int32))
    Rw = tf.gather(rel_pos_w, tf.cast(dist_w, tf.int32))

    B, n_head, q_N, dim = q.shape

    r_q = tf.reshape(q[:, :, sp_idx:], [B, n_head, q_h, q_w, dim])
    rel_h = tf.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = tf.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn_slice = attn[:, :, sp_idx:, sp_idx:]
    attn_shape = tf.shape(attn_slice)
    attn_slice = tf.reshape(attn_slice, [attn_shape[0], attn_shape[1], q_h, q_w, k_h, k_w])
    attn_slice += rel_h[:, :, :, :, :, None] + rel_w[:, :, :, :, None, :]
    attn_slice = tf.reshape(attn_slice, [attn_shape[0], attn_shape[1], q_h * q_w, k_h * k_w])
    attn_slice = tf.concat([attn[:, :, :sp_idx, sp_idx:], attn_slice], axis=2)
    attn_slice = tf.concat([attn[:, :, :, :sp_idx], attn_slice], axis=3)
    return attn_slice


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    dynamic_shape = tf.shape(emb)
    new_shape = tf.concat([dynamic_shape[:-2], [-1]], axis=0)
    emb = tf.reshape(emb, new_shape)

    # emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb


def masked_gap(x, mask):
    """
    Compute the average of tensor values in x, using mask to
    filter out unwanted values.

    Parameters:
    x (tf.Tensor): A tensor of shape [batch_size, sequence_length, feature_dim].
    mask (tf.Tensor): A boolean tensor of shape [batch_size, sequence_length]
                      indicating  which values in x should be considered.

    Returns:
    tf.Tensor: A tensor containing the average values.
    """
    if mask is None:
        return tf.reduce_mean(x, axis=1, keepdims=True)

    float_mask = tf.cast(mask, tf.float32)[..., tf.newaxis]
    filtered_x = x * float_mask
    sum_filtered = tf.reduce_sum(filtered_x, axis=1, keepdims=True) # [bs, 1, D]
    average = sum_filtered / (tf.reduce_sum(float_mask, axis=1, keepdims=True) + 1e-7)

    return average


def split_sequences(padded_tensor, bboxes, max_dim):
    def process_tensor(tensor, bbox):
        off_h, off_w, target_h, target_w = bbox[0], bbox[1], bbox[2], bbox[3]
        cropped_tensor = tf.image.crop_to_bounding_box(tensor, off_h, off_w, target_h, target_w)

        def crop_and_append(row_start, col_start):
            cropped = cropped_tensor[row_start:row_start + max_dim, col_start:col_start + max_dim, :]
            pad_bottom = max_dim - tf.shape(cropped)[0]
            pad_right = max_dim - tf.shape(cropped)[1]
            padding = [[0, pad_bottom], [0, pad_right], [0, 0]]
            return tf.pad(cropped, padding, "CONSTANT")

        tensor_shape = tf.shape(cropped_tensor)
        num_splits_row = tf.maximum(1, tf.cast(tf.math.ceil(tf.cast(tensor_shape[0], tf.float32) / max_dim), tf.int32))
        num_splits_col = tf.maximum(1, tf.cast(tf.math.ceil(tf.cast(tensor_shape[1], tf.float32) / max_dim), tf.int32))

        splits_row = tf.linspace(0.0, tf.cast(tensor_shape[0], tf.float32) - max_dim, num_splits_row)
        splits_col = tf.linspace(0.0, tf.cast(tensor_shape[1], tf.float32) - max_dim, num_splits_col)
        splits_row, splits_col = tf.cast(splits_row, tf.int32), tf.cast(splits_col, tf.int32)
        row_indices, col_indices = tf.meshgrid(splits_row, splits_col, indexing='ij')
        indices = tf.reshape(tf.stack([row_indices, col_indices], axis=-1), [-1, 2])

        crops = tf.map_fn(lambda idx: crop_and_append(idx[0], idx[1]), indices, dtype=cropped_tensor.dtype)
        return crops
    N = tf.shape(padded_tensor)[0]
    # all_crops = tf.TensorArray(dtype=tf.float32, size=N)
    all_crops = tf.TensorArray(dtype=padded_tensor.dtype, size=0, dynamic_size=True, infer_shape=False)

    for i in tf.range(N):
        crops = process_tensor(padded_tensor[i], tf.cast(bboxes[i], tf.int32))
        all_crops = all_crops.write(i, crops)
    return all_crops.concat()


    # reverted_tensors = []
    # for i in tf.range(N):
    #     tensor = padded_tensor[i]
    #     bbox = tf.cast(bboxes[i], tf.int32)
    #
    #     off_h, off_w, target_h, target_w = bbox[0], bbox[1], bbox[2], bbox[3]
    #     cropped_tensor = tf.image.crop_to_bounding_box(tensor, off_h, off_w, target_h, target_w)
    #
    #     if max_dim is not None:
    #         tensor_shape = tf.shape(cropped_tensor)
    #         splits = []
    #         for axis in tf.range(2):  # Only the first two dimensions
    #             if tensor_shape[axis] > max_dim:
    #                 axis_length = tf.cast(tensor_shape[axis], tf.float32)
    #                 max_dim_float = tf.cast(max_dim, tf.float32)
    #
    #                 num_splits = tf.maximum(1, tf.cast(tf.math.ceil(axis_length / max_dim_float), tf.int32))
    #
    #                 # Calculate start indices
    #                 start_indices = tf.linspace(0.0, axis_length - max_dim_float, num_splits)
    #                 start_indices = tf.cast(start_indices, tf.int32)  # Cast back to int for indexing
    #                 splits.append(start_indices)
    #             else:
    #                 splits.append(tf.constant([0], dtype=tf.int32))
    #         for row_start in splits[0]:
    #             for col_start in splits[1]:
    #                 split_tensor = tensor_cols_masked[row_start:row_start+max_dim, col_start:col_start+max_dim, :]
    #                 reverted_tensors.append(split_tensor)
    #     else:
    #         reverted_tensors.append(tensor_cols_masked)
    # return reverted_tensors

    # # for t in reverted_tensors:
    # #     print(t.shape)
    # # print()
    #
    # # Determine the maximum height and width
    # max_height = max(tensor.shape[0] for tensor in reverted_tensors)
    # max_width = max(tensor.shape[1] for tensor in reverted_tensors)
    # max_width = max_height = max(max_width, max_height, max_dim)
    #
    # # Randomly pad each tensor to match the largest dimensions
    # padded_tensors = []
    # for tensor in reverted_tensors:
    #     # Calculate padding sizes
    #     pad_height = max_height - tensor.shape[0]
    #     pad_width = max_width - tensor.shape[1]
    #
    #     if pad_height or pad_width > 0:
    #         if training:
    #             top_pad = random.randint(0, pad_height)
    #             left_pad = random.randint(0, pad_width)
    #         else:
    #             top_pad, left_pad = pad_height//2, pad_width//2
    #         padded_tensor = tf.pad(tensor, [[top_pad, pad_height - top_pad], [left_pad, pad_width - left_pad], [0, 0]], mode='CONSTANT', constant_values=0)
    #     else:
    #         padded_tensor = tensor
    #     padded_tensors.append(padded_tensor)
    #
    # # for t in padded_tensors:
    # #     print(t.shape)
    # # print()
    # return padded_tensors


# DropPath is a regularization technique to improve model's generalization
class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):  # adding the 'training' argument
        if self.drop_prob is None or not training:  # use 'training' here
            return x
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(shape=tf.shape(x), dtype=x.dtype)
        return tf.math.divide(x, keep_prob) * tf.floor(random_tensor)


# MLP as used in the Transformer blocks
class Mlp(layers.Layer):
    def __init__(self, in_features,
                 hidden_features=None, out_features=None,
                 act_layer=None, drop=0., name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.fc1 = layers.Dense(hidden_features,
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                bias_initializer='zeros',
                                name=f'{name}_fc1')
        self.fc2 = layers.Dense(out_features,
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                bias_initializer='zeros',
                                name=f'{name}_fc2')
        self.drop = layers.Dropout(drop)
        self.act_layer = act_layer if act_layer is not None else tf.nn.gelu

    def call(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention mechanism for the Transformer blocks
class Attention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 residual=False, residual_conv_kernel=3, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim = dim

        self.qkv_bias = qkv_bias
        self.qkv = layers.Dense(dim * 3,
                                use_bias=qkv_bias,
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                bias_initializer='zeros' if qkv_bias else None,
                                name=f'{name}_qkv')
        self.proj = layers.Dense(dim,
                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                 bias_initializer='zeros',
                                 name=f'{name}_proj')
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)

        self.residual = residual
        if self.residual:
            self.residual_conv_kernel = residual_conv_kernel
            kernel_size = residual_conv_kernel

            self.res_conv_layer = layers.Conv2D(
                    use_bias=False,
                    groups=num_heads,
                    kernel_size=(kernel_size, kernel_size),
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    filters=self.dim,
                    padding="same",)

            # self.res_conv_layer = DepthwisePointwiseConv(
            #             filters=self.dim,
            #             kernel_size=9,
            #             strides=1,
            #             multiplier=1,
            #             use_depthwise=True,
            #             activation=None,
            #             name=f"{self.name}_res_conv")

    def res_conv(self, v, height=None, width=None):
        if height is not None:
            B = tf.shape(v)[0]
            if tf.shape(v)[2] > (height * width) :
                v_cls = v[:, :, 0, :]
                v = v[:, :, 1:, :]

            v = tf.transpose(v, perm=[0, 2, 1, 3])
            v = tf.reshape(v, [B, height, width, self.dim])

            v = self.res_conv_layer(v)
            v = tf.reshape(v, [B, height * width, self.num_heads, -1])
            v = tf.transpose(v, [0, 2, 1, 3])
            if tf.shape(v)[2] < (height * width + 1) :
                v_cls = tf.expand_dims(v_cls, 2)
                v = tf.concat([v_cls, v], axis=2)
        return v

    def call(self, x, attention_mask=None, height=None, width=None):
        # B, N, C = tf.shape(x)
        shape = tf.shape(x)
        B = shape[0]
        N = shape[1]
        C = self.dim  # out_dim
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=0)  # BS, head, h*w + 1, embed_dim /head

        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, attn.dtype)
            if tf.rank(attention_mask) == 1:
                attention_mask = attention_mask[tf.newaxis, ...]

            attention_mask_2d = tf.einsum('bi,bj->bij', attention_mask, attention_mask)
            attention_mask_2d = attention_mask_2d[:, tf.newaxis, ...]
            attn = attn - 1e9 * (1 - attention_mask_2d)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        if self.residual:
            x += self.res_conv(v, height=height, width=width)

        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MultiScaleAttention(Attention):
    def __init__(self, dim, dim_out, input_size,
                 kernel_q=(1, 1), kernel_kv=(1, 1), stride_q=(1, 1), stride_kv=(1, 1),
                 mode="conv", pool_first=False,
                 norm_layer=layers.LayerNormalization, has_cls_embed=True,
                 rel_pos_spatial=False, rel_pos_zero_init=False,
                 residual_pooling=True, **kwargs):
        super().__init__(dim=dim, **kwargs)

        self.pool_first = pool_first
        self.dim_out = dim_out
        self.input_size = input_size
        self.has_cls_embed = has_cls_embed

        head_dim = dim_out // self.num_heads
        self.scale = head_dim ** -0.5

        if self.pool_first:
            self.q = layers.Dense(dim_out,
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  use_bias=self.qkv_bias,
                                  name=f"{self.name}_q")
            self.k = layers.Dense(dim_out,
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  use_bias=self.qkv_bias,
                                  name=f"{self.name}_k")
            self.v = layers.Dense(dim_out,
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  use_bias=self.qkv_bias,
                                  name=f"{self.name}_v")
            self.qkv = None
        else:
            self.qkv = layers.Dense(dim_out * 3,
                                    use_bias=self.qkv_bias,
                                    kernel_initializer=tf.keras.initializers.HeNormal(),
                                    name=f'{self.name}_qkv')
        self.proj = layers.Dense(self.dim_out,
                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                 name=f'{self.name}_proj')

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if self.mode in ("avg", "max"):
            pool_op = layers.MaxPooling2D if mode == "max" else layers.AveragePooling2D
            self.pool_q = pool_op(
                pool_size=kernel_q, strides=stride_q, padding='same') if len(kernel_q) > 0 else None
            self.pool_k = pool_op(
                pool_size=kernel_kv, strides=stride_kv, padding='same') if len(kernel_kv) > 0 else None
            self.pool_v = pool_op(
                pool_size=kernel_kv, strides=stride_kv, padding='same') if len(kernel_kv) > 0 else None
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim_out // self.num_heads

            self.pool_q = layers.DepthwiseConv2D(
                kernel_size=kernel_q,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                strides=stride_q,
                padding='same',
                use_bias=False,
                name=f'{self.name}_convq'
            ) if len(kernel_q) > 0 else None

            self.norm_q = norm_layer(axis=-1,
                                    gamma_initializer='ones',
                                    beta_initializer='zeros',
                                    name=f'{self.name}_norm_q') if len(kernel_q) > 0 else None

            self.pool_k = layers.DepthwiseConv2D(
                kernel_size=kernel_kv,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                strides=stride_kv,
                padding='same',
                use_bias=False,
                name=f'{self.name}_convk'
            ) if len(kernel_kv) > 0 else None

            self.norm_k = norm_layer(axis=-1,
                                    gamma_initializer='ones',
                                    beta_initializer='zeros',
                                    name=f'{self.name}_norm_k') if len(kernel_kv) > 0 else None

            self.pool_v = layers.DepthwiseConv2D(
                kernel_size=kernel_kv,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                strides=stride_kv,
                padding='same',
                use_bias=False,
                name=f'{self.name}_convv'
            ) if len(kernel_kv) > 0 else None

            self.norm_v = norm_layer(axis=-1,
                                    gamma_initializer='ones',
                                    beta_initializer='zeros',
                                    name=f'{self.name}_norm_q') if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            assert input_size[0] == input_size[1], "Expected square input size for relative position embedding."

            size = input_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            initializer = tf.keras.initializers.TruncatedNormal if not rel_pos_zero_init else tf.zeros_initializer
            self.rel_pos_h = self.add_weight(name="rel_pos_h",
                                            shape=(rel_sp_dim, head_dim),
                                            initializer=initializer(stddev=0.02) if not rel_pos_zero_init else initializer(),
                                            trainable=True)
            self.rel_pos_w = self.add_weight(name="rel_pos_w",
                                            shape=(rel_sp_dim, head_dim),
                                            initializer=initializer,
                                            trainable=True)
        self.residual_pooling = residual_pooling

    def call(self, x, hw, attention_mask=None):
        shape = tf.shape(x)
        B = shape[0]
        N = shape[1]

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = tf.reshape(x, [B, N, fold_dim, -1])
            x = tf.transpose(x, [0, 2, 1, 3])
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"

            qkv = self.qkv(x)
            qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, -1])
            qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
            q, k, v = tf.unstack(qkv, axis=0) # [bs, num_head, L, D//num_head]

        q, q_shape = attention_pool(q, self.pool_q, hw,
                                    has_cls_embed=self.has_cls_embed,
                                    norm_layer=getattr(self, "norm_q", None))
        k, k_shape = attention_pool(k, self.pool_k, hw,
                                    has_cls_embed=self.has_cls_embed,
                                    norm_layer=getattr(self, "norm_k", None))
        v, v_shape = attention_pool(v, self.pool_v, hw,
                                    has_cls_embed=self.has_cls_embed,
                                    norm_layer=getattr(self, "norm_v", None))

        if self.pool_first:
            q_N = np.prod(q_shape) + 1 if self.has_cls_embed else np._prod(q_shape)
            k_N = np.prod(k_shape) + 1 if self.has_cls_embed else np._prod(k_shape)
            v_N = np.prod(v_shape) + 1 if self.has_cls_embed else np._prod(v_shape)

            q = tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [B, q_N, -1])
            q = tf.transpose(tf.reshape(self.q(q), [B, q_N, self.num_heads, -1]), [0, 2, 1, 3])

            v = tf.reshape(tf.transpose(v, [0, 2, 1, 3]), [B, v_N, -1])
            v = tf.transpose(tf.reshape(self.v(v), [B, v_N, self.num_heads, -1]), [0, 2, 1, 3])

            k = tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [B, k_N, -1])
            k = tf.transpose(tf.reshape(self.k(k), [B, k_N, self.num_heads, -1]), [0, 2, 1, 3])

        N = q.shape[2]
        attn = (q * self.scale) @ tf.transpose(k, [0, 1, 3, 2])
        # if attention_mask is not None:
        #     attention_mask = tf.cast(attention_mask, attn.dtype)
        #     attn -= 1e9 * (1 - attention_mask)

        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(attn, q, self.has_cls_embed, q_shape, k_shape, self.rel_pos_h, self.rel_pos_w)

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, attn.dtype)
            # attention_mask_2d = attention_mask[:, tf.newaxis, tf.newaxis, :]
            attention_mask_2d = tf.einsum('bi,bj->bij', attention_mask, attention_mask)[:, tf.newaxis, ...]
            attn = attn - 1e9 * (1 - attention_mask_2d)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                residual = q[:, :, 1:, :]
                x = tf.concat([x[:, :, :1, :], x[:, :, 1:, :] + residual], axis=2)

            else:
                x += q

        x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B, -1, self.dim_out])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn, q_shape

# The VisionTransformer4K model architecture
class Block(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.layers.Activation('gelu'),
                 has_cls_embed=True,
                 use_nystrom=False, num_landmarks=32,
                 residual=False,
                 norm_layer=tf.keras.layers.LayerNormalization,
                 name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.has_cls_embed = has_cls_embed
        self.dim = dim
        self.num_heads=num_heads

        self.qkv_bias = qkv_bias

        self.norm_layer = norm_layer
        self.norm1 = norm_layer(axis=-1,
                                gamma_initializer='ones',
                                beta_initializer='zeros',
                                name=f'{name}_norm1')
        self.norm2 = norm_layer(axis=-1,
                                gamma_initializer='ones',
                                beta_initializer='zeros',
                                name=f'{name}_norm2')
        if use_nystrom:
            self.attn = NystromAttention(
                dim, num_heads=num_heads, num_landmarks=num_landmarks,
                residual=residual,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop,
                name=f'{name}_attn_nystrom')
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, residual=residual,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop,
                name=f'{name}_attn')

        def identity_with_training(x, training=None):
            return tf.identity(x)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else identity_with_training
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act_layer = act_layer
        self.drop = drop

        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=self.act_layer, drop=self.drop)

    def call(self, x, training=False, attention_mask=None,
             h=None, w=None):
        y, attn = self.attn(self.norm1(x), attention_mask=attention_mask,
                            height=h, width=w)
        x = x + self.drop_path(y, training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x, attn, (h, w), attention_mask


class MultiScaleBlock(Block):
    def __init__(self, dim_out, input_size, *args,
                 kernel_q=(1, 1),
                 kernel_kv=(1, 1),
                 stride_q=(1, 1),
                 stride_kv=(1, 1),
                 mode="conv",
                 pool_first=False,
                 rel_pos_spatial=False,
                 rel_pos_zero_init=False,
                 residual_pooling=True,
                 dim_mul_in_att=False,
                 use_mlp=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_out = dim_out
        self.dim_mul_in_att = dim_mul_in_att

        attn_dim = dim_out if dim_mul_in_att else self.dim

        self.attn = MultiScaleAttention(
            self.dim,
            attn_dim,
            num_heads=self.num_heads,
            input_size=input_size,
            qkv_bias=self.qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=self.norm_layer,
            has_cls_embed=self.has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
        )

        mlp_hidden_dim = int(attn_dim * self.mlp_ratio)
        if use_mlp:
            self.mlp = Mlp(in_features=attn_dim, hidden_features=mlp_hidden_dim,
                           out_features=dim_out,
                           act_layer=self.act_layer, drop=self.drop)
        else:
            self.mlp = lambda x: tf.identity(x)

        if self.dim != dim_out:
            self.proj = tf.keras.layers.Dense(dim_out)

        self.pool_skip = None
        if len(stride_q) > 0 and np.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            self.pool_skip = tf.keras.layers.MaxPooling2D(
                pool_size=kernel_skip,
                strides=stride_skip,
                padding='same',
            )

    def call(self, x, training=False, attention_mask=None,
             h=None, w=None):
        hw = (h, w)

        if attention_mask is not None:
            old_dtype = attention_mask.dtype
            if tf.rank(attention_mask) == 1:
                attention_mask = attention_mask[tf.newaxis, ...]
            attention_mask = tf.cast(attention_mask[..., tf.newaxis], x.dtype)
            attention_mask, _ = attention_pool(attention_mask, self.pool_skip, hw,
                                            has_cls_embed=self.has_cls_embed)
            attention_mask = tf.cast(attention_mask[:, :, 0], old_dtype)

        x_norm = self.norm1(x)
        y, attn, hw_new = self.attn(x_norm, hw,
                                    attention_mask=attention_mask)

        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(x, self.pool_skip, hw,
                                  has_cls_embed=self.has_cls_embed)
        x = x_res + self.drop_path(y, training=training)

        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp, training=training)
        return x, attn, hw_new, attention_mask


class MoorePenrosePseudoinverse(tf.keras.layers.Layer):
    def __init__(self, iteration=6, **kwargs):
        super(MoorePenrosePseudoinverse, self).__init__(**kwargs)

        self.iteration = iteration

    def call(self, inputs, **kwargs):
        abs_inputs = tf.abs(inputs)
        cols = tf.math.reduce_sum(abs_inputs, axis=-1)
        rows = tf.math.reduce_sum(abs_inputs, axis=-2)
        z = rearrange(inputs, "... i j -> ... j i") / (
            tf.math.reduce_max(cols) * tf.math.reduce_max(rows)
        )

        identity = tf.eye(z.shape[-1])
        identity = rearrange(identity, "i j -> () i j")

        for _ in range(self.iteration):
            inputs_bbm_z = inputs @ z
            z = (
                0.25
                * z
                @ (
                    13 * identity
                    - (
                        inputs_bbm_z
                        @ (
                            15 * identity
                            - (inputs_bbm_z @ (7 * identity - inputs_bbm_z))
                        )
                    )
                )
            )

        return z


class NystromAttention(layers.Layer):
    def __init__(self, dim, num_heads=8, num_landmarks=32, qkv_bias=False, qk_scale=None,
                 num_iters=6, attn_drop=0., proj_drop=0.,
                 residual=False, residual_conv_kernel=3, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim = dim

        self.qkv = layers.Dense(dim * 3,
                                use_bias=qkv_bias,
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                bias_initializer='zeros' if qkv_bias else None,
                                name=f'{name}_qkv')
        self.proj = layers.Dense(dim,
                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                 bias_initializer='zeros',
                                 name=f'{name}_proj')
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)

        self.residual = residual
        if self.residual:
            self.residual_conv_kernel = residual_conv_kernel

            # self.res_conv_layer = tf.keras.Sequential(
            #     # [tf.keras.layers.Conv2D(
            #     #         use_bias=False,
            #     #         groups=num_heads,
            #     #         kernel_size=(kernel_size, kernel_size),
            #     #         kernel_initializer=tf.keras.initializers.HeNormal(),
            #     #         filters=self.dim,
            #     #         padding="same",),]
            #     [
            #     DepthwisePointwiseConv(
            #                     filters=self.dim,
            #                     kernel_size=self.residual_conv_kernel,
            #                     strides=1,
            #                     use_depthwise=True,
            #                     activation='gelu',
            #                     multiplier=1,
            #                     name=f"{self.name}_3")
            #     ]
            # )
            self.res_conv_layer = DepthwisePointwiseConv(
                        filters=self.dim,
                        kernel_size=self.residual_conv_kernel,
                        strides=1,
                        activation=None,
                        multiplier=1,
                        name=f"{self.name}_res_conv")

    def res_conv(self, v, height=None, width=None, N=None):
        if height is not None:
            pad = v[:, :, :-N, :] if v.shape[2] != N else None
            v = v[:, :, -N:, :]

            B = tf.shape(v)[0]
            if tf.shape(v)[2] > (height * width) :
                v_cls = v[:, :, 0, :]
                v = v[:, :, 1:, :]

            v = tf.transpose(v, perm=[0, 2, 1, 3])
            v = tf.reshape(v, [B, height, width, self.dim])

            v = self.res_conv_layer(v)
            v = tf.reshape(v, [B, height * width, self.num_heads, -1])
            v = tf.transpose(v, [0, 2, 1, 3])
            if tf.shape(v)[2] < (height * width + 1) :
                v_cls = tf.expand_dims(v_cls, 2)
                v = tf.concat([v_cls, v], axis=2)

            if pad is not None:
                v = tf.concat([pad, v], axis=2)
        return v

    def call(self, x, attention_mask=None, height=None, width=None):
        shape = tf.shape(x)
        B, N, C = shape[0], shape[1], self.dim
        h, m, iters, eps = self.num_heads, self.num_landmarks, self.num_iters, 1e-8

        if N % m > 0:
            pad = m - (N % m)
            x = tf.pad(x, [[0, 0], [pad, 0], [0, 0]], constant_values=0.0)
            if attention_mask is not None:
                attention_mask = tf.pad(attention_mask, [[pad, 0]], constant_values=False)

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, -1, 3, h, C // h])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=0)

        if attention_mask is not None:
            attention_mask = tf.reshape(attention_mask, [B, 1, -1])
            q, k, v = map(
                lambda t: t * tf.cast(attention_mask[..., None], q.dtype),
                (q, k, v)
            )
        q = q * self.scale

        l = ceil(N / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        divisor = l
        if attention_mask is not None:
            mask_landmarks_sum = reduce(
                tf.cast(attention_mask, q.dtype),
                "b () (n l) -> b () n",
                "sum",
                l=l
            )
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks /= divisor
        k_landmarks /= divisor

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = tf.einsum(einops_eq, q, k_landmarks)
        sim2 = tf.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = tf.einsum(einops_eq, q_landmarks, k)

        if attention_mask is not None:
            mask_value = -tf.experimental.numpy.finfo(q.dtype).max
            sim1 = tf.where(
                tf.cast(
                    tf.cast(~(attention_mask[..., None]), dtype=q.dtype)
                    * tf.cast(mask_landmarks[..., None, :], dtype=q.dtype),
                    dtype=tf.bool,
                ),
                mask_value, sim1,
            )
            sim2 = tf.where(
                tf.cast(
                    tf.cast(~(mask_landmarks[..., None]), dtype=q.dtype)
                    * tf.cast(mask_landmarks[..., None, :], dtype=q.dtype),
                    dtype=tf.bool,
                ),
                mask_value, sim2,
            )
            sim3 = tf.where(
                tf.cast(
                    tf.cast(~(mask_landmarks[..., None]), dtype=q.dtype)
                    * tf.cast(attention_mask[..., None, :], dtype=q.dtype),
                    dtype=tf.bool,
                ),
                mask_value, sim3,
            )

        attn1, attn2, attn3 = map(
            lambda t: tf.nn.softmax(t, axis=-1), (sim1, sim2, sim3)
        )
        attn2_inv = MoorePenrosePseudoinverse(iteration=iters)(attn2)
        x = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            x += self.res_conv(v, height=height, width=width, N=N)

        x = rearrange(x, "b h n d -> b n (h d)", h=h)
        x = self.proj(x)
        x = x[:, -N:]

        # attn = attn1 @ attn2_inv @ attn3
        return x, None


class PyramidPositionEncoding(tf.keras.layers.Layer):
    def __init__(self, filters, kernels, depth_multiplier=1,
                 activation='gelu', use_DepthwisePointwise=False,
                 nested_convs=False, **kwargs):
        self.filters = filters
        self.nested_convs = nested_convs
        super(PyramidPositionEncoding, self).__init__(**kwargs)

        self.kernels = kernels
        self.convs = []

        for i, kernel in enumerate(self.kernels):
            if use_DepthwisePointwise:
                conv_layer = DepthwisePointwiseConv(
                                filters=self.filters,
                                kernel_size=kernel,
                                strides=1,
                                use_depthwise=True,
                                activation=activation,
                                multiplier=depth_multiplier,
                                name=f"{self.name}_{i}_k{kernel}")
            else:
                conv_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel,
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                padding='same',
                                depth_multiplier=depth_multiplier,
                                name=f"{self.name}_{i}_k{kernel}")
            self.convs.append(conv_layer)

        if self.nested_convs:
            self.convs = tf.keras.Sequential(self.convs)

    def call(self, x):
        if self.nested_convs:
            return self.convs(x)
        outputs = [conv(x) for conv in self.convs]
        return tf.math.add_n(outputs)


class DifferentiableKMeans(layers.Layer):
    def __init__(self, num_clusters, temperature=0.1, **kwargs):
        super(DifferentiableKMeans, self).__init__(**kwargs)
        self.num_clusters = num_clusters
        self.temperature = temperature

    def build(self, input_shape):
        self.cluster_centers = self.add_weight(
            shape=(self.num_clusters, input_shape[-1]),
            initializer='uniform',
            trainable=True,
            name='cluster_centers')

    def call(self, x):
        # Calculate distances between x and cluster_centers
        distances = tf.norm(tf.expand_dims(x, -2) - self.cluster_centers, axis=-1)
        # distances = tf.transpose(distances)
        # print(distances.shape)
        _, indices = tf.nn.top_k(-distances, k=10)
        closest_points = tf.gather(x, indices)
        return tf.reshape(closest_points, [1, -1, tf.shape(x)[-1]])
        #
        #
        #
        # weights = tf.nn.softmax(-distances / self.temperature, axis=1)
        # mean_tokens = tf.matmul(weights, self.cluster_centers)
        # return mean_tokens


class DepthwisePointwiseConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_depthwise=True,
                 activation="gelu", multiplier=2, name=None, **kwargs):
        super(DepthwisePointwiseConv, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.multiplier = multiplier
        self.use_depthwise = use_depthwise
        if use_depthwise:
            self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                padding='same',
                strides=self.strides,
                depth_multiplier=self.multiplier,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                name=f'{self.name}_depthwise_conv')

            self.pointwise_conv = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,  # 1x1 convolution
                padding='same',
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer='zeros',
                name=f'{self.name}_pointwise_conv')

        else:
            self.conv_layer = tf.keras.layers.Conv2D(
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    filters=self.filters,
                    padding="same",)

        self.batch_norm = tf.keras.layers.BatchNormalization(name=f"{self.name}_batch_norm")

        if self.strides != 1:
            self.skip_conv = tf.keras.layers.AveragePooling2D(
                pool_size=self.strides,
                strides=self.strides,
                padding='same',
                name=f'{self.name}_skip_conv')

        self.activation = None
        if activation is not None:
            self.activation = tf.keras.layers.Activation(activation, name=f'{self.name}_pointwise_{activation}')

    def call(self, inputs):
        shortcut = inputs
        if self.strides != 1:
            shortcut = self.skip_conv(inputs)

        if self.use_depthwise:
            x = self.depthwise_conv(inputs)
            x = self.batch_norm(x)
            x = self.pointwise_conv(x)
        else:
            x = self.conv_layer(inputs)
            x = self.batch_norm(x)

        x = x + shortcut
        if self.activation is not None:
            x = self.activation(x)
        return x


class TFPositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.

        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".

        """
        super(TFPositionalEncoding2D, self).__init__()

        self.channels = int(2 * np.ceil(channels / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )

    # @tf.function
    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if tf.rank(inputs) != 4:
            inputs = inputs[tf.newaxis, ...]

        shape = tf.shape(inputs)
        x = shape[1]
        y = shape[2]
        org_channels = shape[3]

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = tf.expand_dims(get_emb(sin_inp_x), 1)
        emb_y = tf.expand_dims(get_emb(sin_inp_y), 0)

        emb_x = tf.tile(emb_x, (1, y, 1))
        emb_y = tf.tile(emb_y, (x, 1, 1))
        emb = tf.concat((emb_x, emb_y), -1)
        penc = tf.repeat(
            emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        return penc


class GlobalMaxPoolLayer(tf.keras.layers.Layer):
    def __init__(self, use_class_token, **kwargs):
        super().__init__(**kwargs)
        self.use_class_token = use_class_token

    def call(self, x, attention_mask, training=False, h=0, w=0):
        if self.use_class_token:
            class_token = x[:, 0:1, :]
            mask_token = attention_mask[:, 0:1]
            x_pooled = x[:, 1:, :]
            masks_pooled = attention_mask[:, 1:]
        else:
            x_pooled = x
            masks_pooled = attention_mask

        # Max pooling for x
        # x_pooled = tf.reduce_max(x_pooled, axis=1, keepdims=True)
        avg_pooled = masked_gap(x_pooled, masks_pooled)
        max_pooled = tf.reduce_max(x_pooled, axis=1, keepdims=True)
        x_pooled = tf.concat([avg_pooled, max_pooled], axis=-1)

        masks_pooled = attention_mask
        if attention_mask is not None:
            masks_pooled = tf.reduce_any(masks_pooled, axis=1, keepdims=True)

        if self.use_class_token:
            x_pooled = tf.concat([class_token, x_pooled], axis=1)
            if masks_pooled is not None:
                masks_pooled = tf.concat([mask_token, masks_pooled], axis=1)

        # Return the pooled x, None for attni, (1, 1) for (h, w), and the updated masks
        return x_pooled, None, (1, 1), masks_pooled


class VisionTransformerWSI(tf.keras.Model):
    def __init__(self, num_classes=0, num_patches=1024,
                 input_embed_dim=384,
                 phi_dim=None,
                 use_phi=True,
                 max_dim=None,
                 output_embed_dim=192,
                 depth=12, global_depth=1,
                 num_heads=12,
                 use_attn_mask=True,
                 mlp_ratio=1.0, qkv_bias=False, qk_scale=None,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 norm_layer=tf.keras.layers.LayerNormalization,
                 num_prototypes=64, regular_training=False,
                 encoding_method="standard",
                 mask_num=2,
                 mask_preglobal=False,
                 use_class_token=False,
                 use_nystrom=False, num_landmarks=32,
                 global_k=-1,
                 name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        config_params = locals().copy()  # Grab the arguments as a dictionary
        del config_params['self']  # Remove 'self'
        del config_params['kwargs']  # Remove 'kwargs' if you don't want to include it
        self.config_params = config_params

        # Store output dimension size
        self.input_embed_dim = input_embed_dim
        self.embed_dim = output_embed_dim
        self.max_dim = max_dim
        self.num_heads = num_heads
        self.use_attn_mask = use_attn_mask

        # Random masking splits up sequences into m sub-sequences
        self.mask_num = mask_num
        self.mask_preglobal = mask_preglobal

        # Store our num_patches grid shape
        self.num_patch = num_patches
        self.num_patch_sq = int(np.sqrt(num_patches))

        # used to adjust the dimensionality of the extracted image patch
        # embeddings before they're fed into the transformer blocks
        self.use_phi = use_phi
        if self.use_phi:
            self.phi_dim = self.embed_dim if phi_dim is None else phi_dim
            self.phi = tf.keras.Sequential(
                [
                tf.keras.layers.Dense(self.phi_dim,
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      kernel_regularizer=tf.keras.regularizers.l1(l=0.5e-4),
                                      use_bias=False,
                                      name=f'{name}_phi_dense_1'),
                tf.keras.layers.Activation('gelu'),
                # tf.keras.layers.Dense(self.phi_dim,
                #                       kernel_initializer=tf.keras.initializers.HeNormal(),
                #                       bias_initializer='zeros',
                #                       use_bias=False,
                #                       name=f'{name}_phi_dense_2'),
                # tf.keras.layers.Activation('tanh'),
                # tf.keras.layers.Dropout(drop_rate, name=f'{name}_phi_dropout'),
                ], name=f'{self.name}_phi')

        self.use_class_token = use_class_token
        if self.use_class_token:
            self.generate_class_tokens(self.embed_dim if self.use_phi else self.input_embed_dim)

        self.encoding_method = encoding_method
        self._initialize_position_encodings()

        # Set up StochasticDepth dropout and transformer blocks
        self.drop_path_rate = drop_path_rate
        self.dpr = np.linspace(0, self.drop_path_rate, depth).tolist()
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.norm_layer = norm_layer
        self.use_nystrom = use_nystrom
        self.num_landmarks = num_landmarks

        self.depth = depth
        self.global_depth = global_depth
        self.blocks = [
            Block(
                dim=self.embed_dim, num_heads=num_heads,
                mlp_ratio=self.mlp_ratio,
                residual=False,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[i],
                norm_layer=self.norm_layer,
                has_cls_embed=self.use_class_token,
                use_nystrom=self.use_nystrom, num_landmarks=self.num_landmarks,
                name=f'{name}_local_block_{i}'
                )
            for i in range(self.depth - self.global_depth)
        ]
        self.global_depth = global_depth
        self.global_blocks = [
            Block(
                dim=self.embed_dim, num_heads=num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=self.dpr[i],
                norm_layer=norm_layer,
                use_nystrom=use_nystrom, num_landmarks=num_landmarks,
                name=f'{name}_global_block_{i}'
                )
            for i in range(self.global_depth)
        ]

        self.norm_local = norm_layer(axis=-1,
                               gamma_initializer=tf.keras.initializers.Constant(1.0),
                               beta_initializer=tf.keras.initializers.Zeros(),
                               name=f'{name}_norm_local_layer')
        self.norm_global = norm_layer(axis=-1,
                               gamma_initializer=tf.keras.initializers.Constant(1.0),
                               beta_initializer=tf.keras.initializers.Zeros(),
                               name=f'{name}_norm_global_layer')

        self.global_k = global_k
        if self.global_k > 0:
            self.cluster_layer = DifferentiableKMeans(num_clusters=self.global_k)

        self.regular_training = regular_training
        if num_classes > 0:
            self.head_local = tf.keras.layers.Dense(num_classes,
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              bias_initializer='zeros',
                                              activation='softmax',
                                              name=f'{name}_head_dense_local')
            self.head_global = tf.keras.layers.Dense(num_classes,
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              bias_initializer='zeros',
                                              activation='softmax',
                                              name=f'{name}_head_dense_global')
        else:
            self.head = tf.identity

    def get_config(self):
        config = super().get_config().copy()
        config.update(self.config_params)
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)

    def generate_class_tokens(self, dim):
        self.cls_token_local = self.add_weight(
            shape=(1, 1, dim),
            initializer='zeros', name=f'{self.name}_cls_local')
        self.cls_token_global = self.add_weight(
            shape=(1, 1, dim),
            initializer='zeros', name=f'{self.name}_cls_global')

    def _initialize_position_encodings(self):
        """Initializes position encodings based on the encoding method."""
        if self.encoding_method == "standard" or self.encoding_method == "":
            self.pos_embed = self.add_weight(
                shape=(1, self.num_patch + 1, self.embed_dim),
                initializer='zeros', name=f'{self.name}_pos')
        elif self.encoding_method == "conditional":
            self.peg = tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', depth_multiplier=1, name='peg')
        elif self.encoding_method == "ppeg":
            self.peg = PyramidPositionEncoding(self.embed_dim,
                                               kernels=[5, 5, 5],
                                               use_DepthwisePointwise=True,
                                               nested_convs=True,
                                               name="ppeg")
        elif self.encoding_method == "sinusoidal":
            self.sin_embed = TFPositionalEncoding2D(self.embed_dim if self.use_phi else self.input_embed_dim)
        elif self.encoding_method is None:
            return
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")

    def set_mask_num(self, mask_num):
        self.mask_num = mask_num

    def prepare_tokens(self, x, training):
        """
        The x input will be of shape [N, H, W, D] where N is the number of
        individual tissue sections per patient and ranges from 1 to 8 for
        our dataset.

        H, W are the height and width of the largest rectangular array of
        features extracted from the WSI tissue sections.  All other N-1 arrays
        along axis 0 are padded to this H and W.

        D is the embedding dimension of the input features.
        """
        # Separate Sequences along axis 0 and remove padding
        x = self.split_sequences(x, max_dim=self.max_dim)
        masks = self.generate_masks(x)

        for i, xi in enumerate(x):
            shape = tf.shape(xi)
            h = shape[0]
            w = shape[1]
            D = shape[2]
            # h, w, D = shape
            xi_phi = self.phi(tf.reshape(xi, [-1, D]))

            if self.use_attn_mask:
                mask = masks[i][:, tf.newaxis]
                xi_phi = xi_phi * tf.cast(mask, xi_phi.dtype)
            x[i] = tf.reshape(xi_phi, [h, w, -1])

        x = self.augment(x, training=training)
        if training:
            if self.mask_num > 1 and not self.mask_preglobal:
                x = self.random_mask(x, self.mask_num)
            masks = self.generate_masks(x)
        return x, masks

    def call(self, x, training=False, return_attns=False, return_gradcam=False):
        x, masks = self.prepare_tokens(x, training) # [h, w, D] x N

        outs = []
        bag_preds = []
        original_shapes = []
        attnis = []

        for j, xi in enumerate(x):
            xi = self.process_preblock_local(xi)
            xi, mask, attni = self.process_blocks_local(xi, masks[j], training)
            masks[j] = mask

            if self.use_class_token:
                outs.append(xi[:, 1:, :])
                xi = self.norm_local(xi)
                bag_preds.append(self.head_local(xi[:, 0, :]))
            else:
                outs.append(xi)
                xi = self.norm_local(xi)
                bag_preds.append(self.head_local(masked_gap(xi, mask)))
            attnis.append(attni)

        x = tf.concat(outs, axis=1)
        global_mask = self.define_global_mask(masks)
        x = self.select_and_shuffle_unmasked(x, global_mask, training=training)
        all_preds = self.define_global_preds(x, bag_preds, training, return_gradcam)
        all_preds = tf.squeeze(tf.concat([all_preds, *bag_preds], axis=0))

        if return_attns:
            return all_preds, original_shapes, [attn, *attnis]
        return all_preds

    # def split_sequences(self, padded_tensor, max_dim=None, training=False):
    #     if max_dim is None:
    #         raise ValueError("max_dim must be specified")
    #
    #     max_files, max_h, max_w, D = padded_tensor.shape
    #
    #     # Calculate random padding for training
    #     if training:
    #         top_pad = random.randint(0, max_dim // 4)
    #         left_pad = random.randint(0, max_dim // 4)
    #     else:
    #         top_pad, left_pad = 0, 0
    #
    #     # Calculate right and bottom padding
    #     bottom_pad = max_dim - top_pad if top_pad > 0 else 0
    #     right_pad = max_dim - left_pad if left_pad > 0 else 0
    #
    #     # Apply padding
    #     padded_tensor = tf.pad(padded_tensor, [[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]], mode='CONSTANT')
    #
    #     # Reshape and transpose to get tiles
    #     total_h = max_h + top_pad + bottom_pad
    #     total_w = max_w + left_pad + right_pad
    #     reshaped = tf.reshape(padded_tensor, [max_files, total_h // max_dim, max_dim,
    #                                           total_w // max_dim, max_dim, D])
    #     transposed = tf.transpose(reshaped, [0, 1, 3, 2, 4, 5])
    #     tiles = tf.reshape(transposed, [-1, max_dim, max_dim, D])
    #
    #     # Exclude empty tiles
    #     non_empty_tiles = tf.reduce_any(tf.reduce_any(tiles != 0, axis=-1), axis=[1, 2])
    #     split_tensors = tf.boolean_mask(tiles, non_empty_tiles)
    #
    #     return split_tensors

    def generate_masks(self, tensors_list):
        if not self.use_attn_mask:
            return [None] * len(tensors_list)

        masks = []
        for tensor in tensors_list:
            mask = tf.math.reduce_any(tensor != 0, axis=-1)
            mask = tf.reshape(mask, [-1])
            if self.use_class_token:
                cls_mask = tf.constant([True], dtype=tf.bool)
                mask = tf.concat([cls_mask, mask], axis=0)
            masks.append(mask)
        return masks

    def random_mask(self, x_list, m):
        """Randomly masks each tensor in x_list into m sub-tensors."""
        if m == 1:
            return x_list
        all_masked_tensors = []

        for x in x_list:
            # Define the foreground regions
            foreground_mask = tf.math.reduce_any(x != 0, axis=-1)

            # Get the indices of the true regions in our foreground mask
            true_indices = tf.where(foreground_mask)

            # Shuffle the indices randomly
            num_indices = tf.shape(true_indices)[0]
            shuffled_indices = tf.random.shuffle(true_indices)

            # Calculate how many indices should be in each group
            indices_per_group = num_indices // m

            for i in range(m):
                start_idx = i * indices_per_group
                end_idx = (i + 1) * indices_per_group if i != m - 1 else num_indices
                group_indices = shuffled_indices[start_idx:end_idx]

                group_mask = tf.scatter_nd(group_indices,
                                           tf.ones((end_idx - start_idx,), dtype=tf.bool),
                                           tf.cast(tf.shape(foreground_mask), group_indices.dtype))

                group_tensor = tf.where(group_mask[:, :, tf.newaxis],
                                        x,
                                        tf.zeros_like(x))
                all_masked_tensors.append(group_tensor)
        return all_masked_tensors

    def select_and_shuffle_unmasked(self, x, global_mask, training=False):
        """
        Select unmasked tokens in x and shuffle them if in training mode.

        Arguments:
        - x: The input tokens.
        - global_mask: The mask associated with the input tokens.
        - training: A boolean to indicate whether or not to shuffle.

        Returns:
        - selected and optionally shuffled x.
        """
        # Extract unmasked tokens
        # if global_mask is not None:
        #     unmasked_tokens = tf.boolean_mask(x, global_mask, axis=1)
        # else:
        unmasked_tokens = x
        if training and self.mask_num > 1:
            if self.mask_preglobal:
                N = tf.shape(unmasked_tokens)[1]
                desired_split_size = N // self.mask_num
                remainder = N % self.mask_num

                split_sizes = [desired_split_size] * (self.mask_num - 1) + [desired_split_size + remainder]
                split_tokens = tf.split(unmasked_tokens, split_sizes, axis=1)

                if global_mask is not None:
                    split_masks = tf.split(global_mask, split_sizes, axis=0)
                else:
                    split_masks = [None] * len(split_tokens)

                return split_tokens, split_masks
        return [unmasked_tokens], [global_mask]

    def split_mask(self, unmasked_tokens):
        N = tf.shape(unmasked_tokens)[1]
        desired_split_size = N // self.mask_num
        remainder = N % self.mask_num

        unmasked_tokens = tf.transpose(unmasked_tokens, [1, 0, 2])
        unmasked_tokens = tf.random.shuffle(unmasked_tokens)
        unmasked_tokens = tf.transpose(unmasked_tokens, [1, 0, 2])

        split_sizes = [desired_split_size] * (self.mask_num - 1) + [desired_split_size + remainder]
        split_tokens = tf.split(unmasked_tokens, split_sizes, axis=1)
        return split_tokens

    def interpolate_pos_encoding(self, x, w, h):
        # This function implements bicubic interpolation of position encodings
        # It returns interpolated position encoding
        npatch = tf.shape(x)[1] - 1
        N = self.num_patch

        def interpolate():
            tokens_pos_embed = self.pos_embed[:, 0:1]
            patch_pos_embed = self.pos_embed[:, 1:]
            n = self.num_patch_sq
            dim = tf.shape(x)[-1]

            # Convert to grid and up-scale
            patch_pos_embed = tf.reshape(patch_pos_embed, [1, n, n, dim])
            patch_pos_embed = tf.image.resize(patch_pos_embed, [w, h],
                                              method=tf.image.ResizeMethod.BICUBIC)
            patch_pos_embed = tf.reshape(patch_pos_embed, [1, -1, dim])

            # Add back the space for the tokens
            patch_pos_embed = tf.concat([tokens_pos_embed, patch_pos_embed], axis=1)
            return patch_pos_embed

        interpolate_cond = tf.logical_and(
            tf.reduce_all(tf.equal(npatch, N)),
            tf.reduce_all(tf.equal(w, h))
        )

        return tf.cond(interpolate_cond,
                       lambda: self.pos_embed,
                       interpolate)

    def augment(self, x, training=True):
        if not training:
            return x

        # Check if x is a list
        if isinstance(x, list):
            # If it's a list, apply the augment function to each tensor in the list
            return [self.augment(xi[tf.newaxis, ...]) for xi in x]

        # Rotate
        x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Flip
        x = tf.cond(tf.random.uniform([]) < 0.5,lambda: tf.image.flip_up_down(x), lambda: x)
        x = tf.cond(tf.random.uniform([]) < 0.5,lambda: tf.image.flip_left_right(x), lambda: x)

        return x[0, ...]

    def extract_hw(self, tensor):
        original_shape = tf.shape(tensor)
        h = original_shape[0]
        w = original_shape[1]
        D = original_shape[-1]
        return h, w, D, original_shape

    def process_preblock_local(self, xi):
        D = tf.shape(xi)[-1]
        if self.encoding_method == "sinusoidal":
            xi = xi + self.sin_embed(xi)

        if tf.rank(xi) == 3:
            xi = xi[tf.newaxis, ...]  # Add a batch dimension
        xi = tf.reshape(xi, [tf.shape(xi)[0], -1, D])
        if self.use_class_token:
            cls_tokens = tf.broadcast_to(self.cls_token_local, [tf.shape(xi)[0], 1, D])
            xi = tf.concat([cls_tokens, xi], axis=1)

        if self.encoding_method == "standard":
            xi = xi + self.pos_embed
        return xi

    def process_blocks_local(self, xi, mask, training, h, w):
        for i, blk in enumerate(self.blocks):
            xi, attni, (h, w), mask = blk(xi, attention_mask=mask, training=training, h=h, w=w)

            if i == 0 and (self.encoding_method == "conditional" or self.encoding_method == "ppeg"):
                if self.use_class_token:
                    peg_input = tf.reshape(xi[:, 1:, :], [1, h, w, D])
                else:
                    peg_input = tf.reshape(xi, [1, h, w, D])
                peg_encodings = self.peg(peg_input)  # [h, w, D]
                peg_encodings = tf.reshape(peg_encodings, [1, -1, D])
                if self.use_class_token:
                    zeros = tf.zeros([1, 1, D])
                    peg_encodings = tf.concat([zeros, peg_encodings], axis=1)
                xi = xi + peg_encodings
        return xi, mask, attni

    def define_global_mask(self, masks):
        global_mask = None
        if self.use_attn_mask:
            if self.use_class_token:
                masks = masks[:, 1:]
            global_mask = tf.reshape(masks, [-1])
        return global_mask

    def process_blocks_global(self, x, training=False, return_gradcam=False):
        # if self.global_k > 0:
        #     x = self.cluster_layer(x)
        # if self.use_class_token:
        #     x = tf.concat([self.cls_token_global, x], axis=1)

        for blk in self.global_blocks:
            x, attn, _, _ = blk(x, training=training)

        # if return_gradcam:
        #     with tf.GradientTape(persistent=True) as tape:
        #         tape.watch(x)
        #         if self.use_class_token:
        #             x_gap = self.norm_global(x)[:, 0, :]
        #         else:
        #             x_gap = tf.reduce_mean(self.norm_global(x), axis=1)
        #         global_pred = self.head_global(x_gap)
        #         grads = tape.gradient(global_pred, x)
        #     pooled_grads = tf.reduce_mean(grads, axis=-1)[..., tf.newaxis]
        #     gradcam = tf.reduce_sum(pooled_grads * x, axis=-1)
        #     # gradcam = tf.nn.relu(gradcam)
        #     # gradcam = gradcam / (tf.reduce_max(gradcam) + 1e-8)
        #     return gradcam
        # else:
        #     # if self.use_class_token:
        #     #     x_gap = self.norm_global(x)[:, 0, :]
        #     # else:
        x_gap = tf.reduce_max(self.norm_global(x), axis=1)
        global_pred = self.head_global(x_gap)
        return global_pred

    def define_global_preds(self, x, training, return_gradcam):
        all_preds = [
            self.process_blocks_global(x_split, training=training, return_gradcam=return_gradcam)
            for x_split in x
            ]
        all_preds = tf.reduce_mean(all_preds, axis=0)
        return all_preds



class VisionTransformerWSI_256(VisionTransformerWSI):

    def __init__(self, *args,
                 noise_aug = 0.05,
                 downscale_depth=1,
                 downscale_multiplier=1.5,
                 attnpool_mode="conv",
                 downscale_first=True,
                 downscale_stride_q=2,
                 downscale_stride_k=2,
                 local_kernel=1,
                 global_stride_q=4,
                 global_stride_k=4,
                 data_dir,
                 **kwargs):
        super().__init__(*args, **kwargs)
        config_params = locals().copy()  # Grab the arguments as a dictionary
        del config_params['self']  # Remove 'self'
        del config_params['args']  # Remove 'args'
        del config_params['kwargs']  # Remove 'kwargs' if you don't want to include it
        self.config_params.update(config_params)
        self.background_tile  = join(data_dir, "background_tile_norm.npy")
        print(f"Using Background tile located: {self.background_tile}")
        self.background_tile = tf.constant(np.load(self.background_tile))

        self.downscale_depth = downscale_depth
        self.downscale_multiplier = downscale_multiplier
        self.downscale_first = downscale_first
        output_dims = [int(self.embed_dim * ((self.downscale_multiplier)**(i + 1))) for i in range(downscale_depth)]
        output_dims = [(dim + self.num_heads - 1) // self.num_heads * self.num_heads for dim in output_dims]
        print(output_dims)
        self.input_size = (self.max_dim, self.max_dim) if self.max_dim is not None else (150, 150)
        self.attnpool_mode = attnpool_mode

        s_q = downscale_stride_q
        s_k = downscale_stride_k

        total_depth =  self.depth + self.downscale_depth

        self.dpr = np.linspace(0, self.drop_path_rate, total_depth + 1).tolist()
        self.input_sizes = [int(self.max_dim / (s_q**i)) for i in range(self.downscale_depth + 1)]
        self.input_sizes = [(s, s) for s in self.input_sizes]
        print("input sizes", self.input_sizes)

        self.downscale_blocks = [
            MultiScaleBlock(
                dim=(self.embed_dim if self.use_phi else self.input_embed_dim) if i == 0 else output_dims[i-1],
                dim_out=output_dims[i],
                num_heads=1,
                mlp_ratio=self.mlp_ratio,
                input_size=self.input_sizes[i],
                rel_pos_spatial=True,
                has_cls_embed=self.use_class_token,
                kernel_q=(s_q + 1, s_q + 1),
                kernel_kv=(s_k + 1, s_k + 1),
                stride_q=(s_q, s_q),
                stride_kv=(s_k, s_k),
                mode=self.attnpool_mode,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                name=f'{self.name}_downscale_block_{i}'
            )
            for i in range(self.downscale_depth)
        ]

        self.blocks = [
            MultiScaleBlock(
                dim=output_dims[-1] if (self.downscale_first and len(output_dims) > 0) else self.embed_dim,
                dim_out=output_dims[-1] if (self.downscale_first and len(output_dims) > 0) else self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                input_size=self.input_sizes[-1] if self.downscale_first else (self.max_dim, self.max_dim),
                rel_pos_spatial=True,
                has_cls_embed=self.use_class_token,
                # kernel_q=(1, 1), #if i >= self.depth - 2 else (1, 1),
                # kernel_kv=(1, 1), #if i >= self.depth - 2 else (1, 1),
                # stride_q=(1, 1),
                # stride_kv=(1, 1),
                mode=self.attnpool_mode,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[i + self.downscale_depth if self.downscale_first else i],
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                norm_layer=self.norm_layer,
                name=f'{self.name}_local_block_{i}'
            )
            for i in range(self.depth)
        ]

        s_q = global_stride_q
        s_k = global_stride_k

        # self.global_blocks = []
        # self.global_blocks = [
        #     MultiScaleBlock(
        #         dim=output_dims[-1],
        #         dim_out=output_dims[-1],
        #         num_heads=1,
        #         mlp_ratio=self.mlp_ratio,
        #         input_size=self.input_sizes[-1],
        #         rel_pos_spatial=True,
        #         has_cls_embed=self.use_class_token,
        #         kernel_q=(s_q + 1, s_q + 1),
        #         kernel_kv=(s_k + 1, s_k + 1),
        #         stride_q=(s_q, s_q),
        #         stride_kv=(s_k, s_k),
        #         drop_path=0.0,
        #         mode="conv",
        #         name=f'{self.name}_global_block'
        #         )
        # ]
        self.global_blocks = [
            GlobalMaxPoolLayer(self.use_class_token,
                               name=f'{self.name}_global_maxpool')
        ]

        self.global_attn = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dims[-1] if len(output_dims) > 0 else self.embed_dim,
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  name=f'{self.name}_global_attn_hidden'),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(self.drop_rate, name=f'{self.name}_global_attn_dropout'),
            tf.keras.layers.Dense(1,
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  name=f'{self.name}_global_attn_final'),
            tf.keras.layers.Activation('relu'),
        ], name=f'{self.name}_global_attn')

        self.head_local = None

        self.noise_aug = noise_aug
        self.dampen_noise = 0.2
        self.dampen_fraction = 0.5

    def mask_pool(self, original_mask):
        original_dtype = original_mask.dtype
        original_mask = tf.cast(original_mask, tf.float32)

        pooled_mask = original_mask
        for s in self.consolidate_strides:
            pooled_mask = tf.nn.max_pool2d(
                pooled_mask,
                ksize=[1, self.consolidate_kernel, self.consolidate_kernel, 1],
                strides=[1, s, s, 1],
                padding='SAME'
            )
        if self.use_maxpool:
            pooled_mask = tf.nn.max_pool2d(pooled_mask, ksize=(2, 2), strides=(2, 2), padding='SAME')
        pooled_mask = tf.squeeze(pooled_mask, axis=0)

        return tf.cast(pooled_mask, original_dtype)

    # @tf.function
    def random_noise(self, x, masks, training=True, block_exp=3, background_tile=None):
        # Assume x is either a list of xi of shape [h, w, D] or a tensor of
        # shape [h, w, D]
        # noise_level=0.10
        # fraction=0.25
        if not training:
            return x

        if background_tile is None:
            background_tile = self.background_tile
        batch_size, h, w, D = x.shape
        block_size = 2 ** (random.randint(2, block_exp))
        block_size = min(block_size, h, w)

        n_blocks_h = (h // block_size)
        n_blocks_w = (w // block_size)

        def dampen_image(image):
            use_noise = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)  # 0 for zeroing, 1 for noise

            num_blocks_to_dampen = int(n_blocks_h * n_blocks_w * self.dampen_fraction)
            num_blocks_to_dampen = tf.random.uniform(shape=(), minval=0, maxval=num_blocks_to_dampen + 1, dtype=tf.int32)
            if num_blocks_to_dampen == 0:
                return image
            indices = tf.random.shuffle(tf.range(n_blocks_h * n_blocks_w))[:num_blocks_to_dampen]
            row_indices = indices // n_blocks_w
            col_indices = indices % n_blocks_w

            for i in range(tf.size(row_indices)):
                row = row_indices[i] * block_size
                col = col_indices[i] * block_size

                # Create indices for each pixel in the block
                rows = tf.range(row, row + block_size)
                cols = tf.range(col, col + block_size)
                depth = tf.range(D)

                rows, cols, depth = tf.meshgrid(rows, cols, depth, indexing='ij')
                indices_to_update = tf.stack([tf.reshape(rows, [-1]), tf.reshape(cols, [-1]), tf.reshape(depth, [-1])], axis=-1)

                if use_noise == 0:
                    # block_value = tf.zeros([block_size * block_size * D], dtype=image.dtype)
                    block_value = tf.tile(
                        tf.cast(background_tile[tf.newaxis, tf.newaxis, :], image.dtype),
                        [block_size, block_size, 1])
                    block_value = tf.reshape(block_value, [-1])
                elif use_noise == 1:
                    image_block = image[row:row + block_size, col:col + block_size, :]
                    average_tile = tf.reduce_mean(image_block, axis=(0, 1), keepdims=True)
                    block_value = tf.tile(average_tile, [block_size, block_size, 1])
                    block_value = tf.reshape(block_value, [-1])
                else:
                    image_block = image[row:row + block_size, col:col + block_size, :]
                    block_value = tf.random.normal((block_size, block_size, D), mean=0.0, stddev=self.dampen_noise, dtype=image.dtype)
                    block_value = tf.reshape(block_value + image_block, [-1])

                image = tf.tensor_scatter_nd_update(
                    image,
                    indices_to_update,
                    block_value
                )

            return image

        x = tf.map_fn(dampen_image, x, fn_output_signature=tf.float32)
        x += tf.random.normal(x.shape, stddev=self.noise_aug, dtype=x.dtype)

        if self.use_attn_mask:
            masks = masks[:, 1:] if self.use_class_token else masks
            x *= tf.cast(tf.reshape(masks, [batch_size, h, w, 1]), x.dtype)
        return x

    # @tf.function
    def flip_rotate(self, x, training=True):
        """
        Augment a batch of tensors. The input x is expected to be a batch of
        tensors with shape [bs, hi, wi, D].
        """
        if not training:
            return x

        # Rotate (randomly rotate each image in the batch)
        def random_augment(image):
            image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.image.flip_up_down(image), lambda: image)
            image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
            return image

        x = tf.map_fn(random_augment, x, fn_output_signature=tf.float32)
        return x

    def generate_masks(self, x):
        """
        Generate attention masks for a batch of tensors.
        The input x is expected to be a batch of tensors with shape [bs, h, w, D].
        """
        if not self.use_attn_mask:
            return None  # No need to generate masks

        mask = tf.math.reduce_any(x != 0, axis=-1)  # Shape: [bs, h, w]
        mask = tf.reshape(mask, [tf.shape(x)[0], -1])  # Shape: [bs, h*w]

        if self.use_class_token:
            # Add a True mask for the class token at the start of each mask in the batch
            cls_mask = tf.ones([tf.shape(x)[0], 1], dtype=mask.dtype)
            mask = tf.concat([cls_mask, mask], axis=1)
        return mask

    def fill_blanks(self, x):
        """
        Fill the background regions in x with self.background_tile.

        Args:
            x (Tensor): Input tensor of shape (batch_size, h, w, D).

        Returns:
            Tensor: The tensor with background regions filled.
        """
        background_mask = tf.math.reduce_all(x == 0, axis=-1, keepdims=True) # Shape: (batch_size, h, w, 1)

        background_feat = tf.reshape(self.background_tile, [1, 1, 1, -1])
        background_feat = tf.cast(background_feat, x.dtype)
        background_tiled = tf.tile(background_feat, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])

        return tf.where(background_mask, background_tiled, x)

    @tf.function
    def prepare_tokens(self, x, training):
        """
        The x input will be of shape [N, H, W, D] where N is the number of
        individual tissue sections per patient and ranges from 1 to 8 for
        our dataset.

        H, W are the height and width of the largest rectangular array of
        features extracted from the WSI tissue sections.  All other N-1 arrays
        along axis 0 are padded to this H and W.

        D is the embedding dimension of the input features.
        """
        batch_size, h, w, D = x.shape
        if not self.use_attn_mask:
            x = self.fill_blanks(x)

        x = self.flip_rotate(x, training=training)
        masks = self.generate_masks(x)
        x = self.random_noise(x, masks, block_exp=3, training=training)
        masks = self.generate_masks(x)
        if self.use_phi:
            x = self.phi(tf.reshape(x, [-1, D]))  # Shape: [N*H*W, D]
            if self.use_attn_mask:
                # Adapt the mask application for the entire batch
                mask = masks[:, 1:] if self.use_class_token else masks
                x = x * tf.cast(tf.reshape(mask, [-1, 1]), x.dtype)
            x = tf.reshape(x, [batch_size, h, w, -1])  # Reshape back to original batch shape
        return x, masks

        # for i, xi in enumerate(x):
        #     shape = tf.shape(xi)
        #     h, w, D = shape[0], shape[1], shape[2]
        #
        #     if self.use_phi:
        #         xi = self.phi(tf.reshape(xi, [-1, D]))
        #         if self.use_attn_mask:
        #             mask = masks[i][1:] if self.use_class_token else masks[i]  # (h*w, D)
        #             xi = xi * tf.cast(mask[..., tf.newaxis], xi.dtype) # (h*w, D)
        #         xi = tf.reshape(xi, [h, w, -1])  # (h, w, D)
        #
        #     if self.consolidate_tokens:
        #         xi = self.token_consolidation_layer(xi[tf.newaxis, ...])
        #         if self.use_attn_mask:
        #             mask = self.mask_pool(tf.reshape(mask, [1, h, w, 1]))
        #             xi = xi * tf.cast(mask, xi.dtype)
        #             mask = tf.reshape(mask, [-1])
        #         xi = tf.squeeze(xi, axis=0)
        #
        #     if self.use_class_token and self.use_attn_mask:
        #         cls_mask = tf.constant([True], dtype=tf.bool)
        #         mask = tf.concat([cls_mask, mask], axis=0)
        #         masks[i] = mask
        #
        #     x[i] = xi
        #
        # x = self.augment(x, training=training)
        # x = self.random_dampen(x, training)
        # if training:
        #     if self.mask_num > 1 and not self.mask_preglobal:
        #         x = self.random_mask(x, self.mask_num)
        #     masks = self.generate_masks(x)
        # return [t[0, ...] for t in tf.split(x, batch_size, axis=0)], [t[0, ...] for t in tf.split(masks, batch_size, axis=0)]

    def call(self, x, training=False, return_attns=False, return_gradcam=False):
        x, masks = self.prepare_tokens(x, training) # [bs, h, w, D], [bs, 1 + h*w]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]

        x = self.process_preblock_local(x) # [bs, 1 + hw, D]
        if self.downscale_first:
            x, masks, _, (h, w) = self.process_blocks_downscale(x, masks, training, h, w)
            x, masks, attni, (h, w) = self.process_blocks_local(x, masks, training, h, w)
        else:
            x, masks, attni, (h, w) = self.process_blocks_local(x, masks, training, h, w)
            x, masks, _, (h, w) = self.process_blocks_downscale(x, masks, training, h, w)
        D = tf.shape(x)[-1]

        if self.use_class_token:
            bag_preds = self.head_global(self.norm_local(x)[:, 0, :]) # [bs, n_class]
            x = tf.reshape(x[:, 1:, :], [1, -1, D]) # [1, bs * h*w, D]
        else:
            bag_preds = self.head_global(self.norm_local(x)[:, 0, :]) # [bs, n_class]
            x = tf.reshape(x, [1, -1, D])

        global_mask = self.define_global_mask(masks)
        x, global_mask = self.select_and_shuffle_unmasked(x, global_mask, training=training)
        global_pred = self.define_global_preds(x, global_mask, training=training, return_weights=return_attns)
        if return_attns:
            global_pred, global_attns = global_pred
            global_attns = tf.reshape(global_attns, [-1, h, w])
        all_preds = tf.concat([global_pred, bag_preds], axis=0)

        if return_attns:
            return all_preds, attni, global_attns
        return all_preds

    def process_blocks_downscale(self, x, masks, training, h, w):
        attni = None
        for i, blk in enumerate(self.downscale_blocks):
            x, attni, (h, w), masks = blk(x, attention_mask=masks, training=training, h=h, w=w)
            # print(h, w)
            # if prev_h != tf.shape(xi)[0] and self.encoding_method == "sinusoidal":
            #     xi = xi + self.sin_embed(xi)
        return x, masks, attni, (h, w)

    def process_blocks_local(self, x, masks, training, h, w):
        attnis = []
        for i, blk in enumerate(self.blocks):
            x, attni, (h, w), mask = blk(x, attention_mask=masks, training=training, h=h, w=w)
            attnis.append(attni)

        for i, blk in enumerate(self.global_blocks):
            x, _, (h, w), masks = blk(x, attention_mask=masks, training=training, h=h, w=w)
        # print(x.shape)
        return x, masks, tf.stack(attnis), (h, w)

    def process_blocks_global(self, x, mask, training=False, return_weights=False):
        x = self.norm_global(x)
        if mask is not None:
            mask = mask[tf.newaxis, :, tf.newaxis]

        weights = self.global_attn(x) # x [1, N, D] -> weights[1, N, 1]
        if mask is not None:
            weights -= (1 - tf.cast(mask, weights.dtype)) * 1e9
        weights = tf.nn.softmax(weights, axis=1)

        x_avg = tf.reduce_sum(x * weights, axis=1)
        global_pred = self.head_global(x_avg)
        if return_weights:
            return global_pred, weights
        return global_pred

    def define_global_preds(self, x, global_mask, training=False, return_weights=False):
        results = [self.process_blocks_global(x_split, mask,
                                               training=training,
                                               return_weights=return_weights)
                    for (x_split, mask) in zip(x, global_mask)]

        if return_weights:
            all_preds, all_weights = zip(*results)
        else:
            all_preds = results
        global_pred = tf.reduce_mean(all_preds, axis=0)

        if return_weights:
            return global_pred, tf.concat(all_weights, axis=1)
        return global_pred
