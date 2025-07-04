import jax
import jax.numpy as jnp

import equinox as eqx
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float
from typing import Tuple

from rope import calculate_rope


class RMSNorm(eqx.Module):
    eps: float
    weight: Float[Array, "*shape"]  # noqa: F821

    def __init__(self, dim, eps, dtype=jnp.bfloat16):
        dtype = default_floating_dtype() if dtype is None else dtype
        self.eps = eps
        self.weight = jnp.ones(shape=dim, dtype=dtype)

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.mean(x**2, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        return output * self.weight


class FeedForward(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, args, key, dtype=jnp.bfloat16):
        dtype = default_floating_dtype() if dtype is None else dtype
        key1, key2, key3 = jax.random.split(key, 3)

        self.w1 = eqx.nn.Linear(
            args.dim, args.hidden_dim, use_bias=False, key=key1, dtype=dtype
        )
        self.w2 = eqx.nn.Linear(
            args.hidden_dim, args.dim, use_bias=False, key=key2, dtype=dtype
        )
        self.w3 = eqx.nn.Linear(
            args.dim, args.hidden_dim, use_bias=False, key=key3, dtype=dtype
        )

    def __call__(self, x):
        h = jax.nn.silu(self.w1(x).astype(jnp.float32)).astype(x.dtype)
        return self.w2(h * self.w3(x))


class Attention(eqx.Module):
    dim: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    kv_repeats: int
    sliding_window: int
    scale: float
    split_sizes: Tuple
    wqkv: eqx.nn.Linear
    wo: eqx.nn.Linear

    def __init__(self, args, key, dtype=jnp.bfloat16):
        dtype = default_floating_dtype() if dtype is None else dtype
        key1, key2 = jax.random.split(key, 2)

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.kv_repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window
        self.scale = args.head_dim**-0.5
        total_head_dim = (args.n_heads + 2 * args.n_kv_heads) * args.head_dim
        self.split_sizes = (
            args.n_heads * args.head_dim,
            (args.n_heads * args.head_dim) + (args.n_kv_heads * args.head_dim),
        )

        self.wqkv = eqx.nn.Linear(
            args.dim, total_head_dim, use_bias=False, key=key1, dtype=dtype
        )
        self.wo = eqx.nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            use_bias=False,
            key=key2,
            dtype=dtype,
        )

    def compute_scores_and_output(self, xq, key, value, mask, seqlen, pos_mask):
        query = jnp.transpose(xq, (1, 0, 2))
        key = jnp.transpose(key, (1, 0, 2))
        value = jnp.transpose(value, (1, 0, 2))

        # # # scores : [n_heads, seqlen | 1, seqlen]
        scores = jnp.matmul(query, jnp.transpose(key, (0, 2, 1))) * self.scale
        if pos_mask is not None:
            scores = jnp.where(pos_mask, -jnp.inf, scores)

        if mask is not None:
            # Mask will of shape [seqlen, seqlen] but our scores
            # have shape [num_heads, seqlen, seqlen], hence we need
            # to introduce another dimension in the mask
            mask = mask[jnp.newaxis, ...]
            scores = scores + mask

        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        output = jnp.matmul(scores, value)
        output = jnp.reshape(jnp.transpose(output, (1, 0, 2)), (seqlen, -1))
        output = jax.vmap(self.wo)(output)
        return output

    def __call__(
        self, x, cos_freq, sin_freq, positions, mask=None, cache_k=None, cache_v=None
    ):
        # x shape: [seqlen, embed_dim]
        seqlen = x.shape[0]
        #print(x.shape)
        xqkv = jax.vmap(self.wqkv)(x)
        xq, xk, xv = jnp.split(xqkv, self.split_sizes, axis=-1)

        xq = jnp.reshape(xq, (seqlen, self.n_heads, self.head_dim))
        xk = jnp.reshape(xk, (seqlen, self.n_kv_heads, self.head_dim))
        xv = jnp.reshape(xv, (seqlen, self.n_kv_heads, self.head_dim))

        xq = calculate_rope(xq, cos_freq, sin_freq)
        xk = calculate_rope(xk, cos_freq, sin_freq)

        if positions.shape[0] > 1:
            # prefill
            cache_k = cache_k.at[positions, :, :].set(xk[positions, :, :], mode="drop")
            cache_v = cache_v.at[positions, :, :].set(xv[positions, :, :], mode="drop")

            key = jnp.repeat(xk, self.kv_repeats, axis=1)
            value = jnp.repeat(xv, self.kv_repeats, axis=1)
            output = self.compute_scores_and_output(xq, key, value, mask, seqlen, None)
        else:
            # single-token generation
            one_hot_indices = jax.nn.one_hot(
                positions, self.sliding_window, dtype=cache_k.dtype
            ).reshape(self.sliding_window, 1, 1)
            # Ideally, we expect that you flush out the cache with the new prompt,
            # and start over. Why flushing out the cache values are necessary?
            # It ensures that we are not adding any values updated earlier with the
            # new updates, meaning we are always replacing the value not updating it.
            # For example, if prompt had a length of 6, and you want to generate 7th
            # token, this ensures that we are not adding the old value of 7th token
            # to the updated value as it would lead to wrong results.
            # In case, you are not flushing the cache after every prompt, add the
            # `jnp.where()` condition as shown below:
            # k_updates = cache_k + xk * one_hot_indices
            # v_updates = cache_v + xv * one_hot_indices
            # cache_k = jnp.where(cache_k, cache_k, k_updates)
            # cache_v = jnp.where(cache_v, cache_v, v_updates)
            cache_k = cache_k + xk * one_hot_indices
            cache_k = cache_v + xv * one_hot_indices

            cur_pos = positions[-1] + 1
            causal_mask = jnp.broadcast_to(
                jnp.arange(self.sliding_window) >= cur_pos, (1, 1, self.sliding_window)
            ).reshape(self.sliding_window, 1, 1)

            key = jnp.repeat(
                jnp.where(causal_mask, 0, cache_k), axis=1, repeats=self.kv_repeats
            )
            value = jnp.repeat(
                jnp.where(causal_mask, 0, cache_v), axis=1, repeats=self.kv_repeats
            )

            output = self.compute_scores_and_output(
                xq,
                key,
                value,
                mask,
                seqlen,
                causal_mask.reshape((1, 1, self.sliding_window)),
            )

        return output, cache_k, cache_v


class TransformerBlock(eqx.Module):
    dim: int
    n_heads: int
    attention: Attention
    attention_norm: RMSNorm
    feed_forward: FeedForward
    ffn_norm: RMSNorm

    def __init__(self, args, key, dtype=jnp.bfloat16):
        key1, key2 = jax.random.split(key, 2)
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention = Attention(args, key=key1, dtype=dtype)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

        self.feed_forward = FeedForward(args, key=key2, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

    def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
        normed_x = jax.vmap(self.attention_norm)(x)
        r, cache_k, cache_v = self.attention(
            normed_x, cos_freq, sin_freq, positions, mask, cache_k, cache_v
        )
        h = x + r
        r = jax.vmap(self.feed_forward)(jax.vmap(self.ffn_norm)(h))
        out = h + r
        return out, cache_k, cache_v


class Transformer(eqx.Module):
    tok_embeddings: eqx.nn.Embedding
    layers: TransformerBlock
    norm: RMSNorm
    output: eqx.nn.Linear
    vocab_size: int
    n_layers: int
    sliding_window: int

    def __init__(self, args, key, dtype=jnp.bfloat16):
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.sliding_window = args.sliding_window
        keys = jax.random.split(key, args.n_layers + 2)
        embed_key, linear_key, tf_layers_keys = keys[0], keys[1], keys[2:]

        self.tok_embeddings = eqx.nn.Embedding(
            args.vocab_size, args.dim, key=embed_key, dtype=dtype
        )
        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps, dtype=dtype)
        self.output = eqx.nn.Linear(
            args.dim, args.vocab_size, use_bias=False, key=linear_key, dtype=dtype
        )
        make_layers = lambda k: TransformerBlock(args, key=k, dtype=dtype)
        self.layers = eqx.filter_vmap(make_layers)(tf_layers_keys)
        del make_layers
        # Create a list of individual transformer block instances
        #self.layers = [TransformerBlock(args, key=k, dtype=dtype) for k in tf_layers_keys]


    def compute_mask(self, seqlen):
        t = jnp.full((seqlen, seqlen), dtype=jnp.bfloat16, fill_value=1)
        mask = jnp.tril(t, k=0)
        # make the mask banded to account for sliding window
        mask = jnp.log(jnp.triu(mask, k=-self.sliding_window))
        return mask

    def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
        # x is of shape (seqlen, )
        h = jax.vmap(self.tok_embeddings)(x)
        #h = jax.vmap(jax.vmap(self.tok_embeddings))(x)

        if x.shape[-1] > 1:
            seqlen = x.shape[-1]
            mask = self.compute_mask(seqlen)
        else:
            mask = None

        # Partition the TransformerLayers into static and dynamic parts
        dynamic_layers, static_layers = eqx.partition(self.layers, eqx.is_array)

        def f(_x, _dynamic_l):
            layer = eqx.combine(_dynamic_l, static_layers)
            h, cache_k, cache_v, layer_idx = _x
            h, cache_ki, cache_vi = layer(
                h,
                cos_freq,
                sin_freq,
                positions,
                mask,
                cache_k[layer_idx, ...],
                cache_v[layer_idx, ...],
            )
            cache_k = cache_k.at[layer_idx, :, :, :].set(cache_ki)
            cache_v = cache_v.at[layer_idx, :, :, :].set(cache_vi)
            return (h, cache_k, cache_v, layer_idx + 1), None

        layer_idx = 0
        #print(h.shape, cache_k.shape, cache_v.shape, layer_idx)
        (h, cache_k, cache_v, layer_idx), _ = jax.lax.scan(
            f, (h, cache_k, cache_v, layer_idx), dynamic_layers
        )

        h = jax.vmap(self.norm)(h).astype(jnp.float32)
        #h = jax.vmap(self.output)(h).astype(jnp.float32)
        return h, cache_k, cache_v
