"""Noisy MLP module"""

from typing import Callable, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class Linear(hk.Module):
  """Linear module."""

  def __init__(
            self,
            output_size: int,
            #rng: jax.random.PRNGKey,
            with_bias: bool = True,
            w_init: Optional[hk.initializers.Initializer] = None,
            b_init: Optional[hk.initializers.Initializer] = None,
            name: Optional[str] = None,
  ):
    """Constructs the Linear module.
    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev `1 / sqrt(fan_in)`. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super().__init__(name=name)
    #self.rng = hk.PRNGSequence(rng)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_mu_init
    self.b_init = b_mu_init or jnp.zeros

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    
    if w_init is None:
      stddev = 2. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
    
    out = jnp.dot(inputs, w)

    if self.with_bias:
      b = hk.get_parameter("b_mu", [self.output_size], dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b
      
    return out


class MLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      with_bias=True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
  ):
    """Constructs an MLP.
    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for Linear weights.
      b_init: Initializer for Linear bias. Must be `None` if `with_bias` is
        `False`.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between linear layers. Defaults
        to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.
    Raises:
      ValueError: If with_bias is False and b_init is not None.
    """

    #self.rng = jax.random.PRNGKey(seed)
    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_mu_init
    self.b_init = b_mu_init
    self.activation = activation
    self.activate_final = activate_final
    layers = []
    for index, output_size in enumerate(output_sizes):
      layers.append(Linear(
          output_size=output_size,
          w_init=w_init,
          b_init=b_init,
          with_bias=with_bias,
          name="noisy_linear_%d" % index))
    self.layers = tuple(layers)

  def __call__(
      self,
      inputs: jnp.ndarray,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jnp.ndarray:
    """Connects the module to some inputs.
    Args:
      inputs: A Tensor of shape `[batch_size, input_size]`.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.
    Returns:
      output: The output of the model of size `[batch_size, output_size]`.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return out

  def reverse(
      self,
      activate_final: Optional[bool] = None,
      name: Optional[str] = None,
  ) -> "NoisyMLP":
    """Returns a new NoisyMLP which is the layer-wise reverse of this NoisyMLP.
    NOTE: Since computing the reverse of an MLP requires knowing the input size
    of each linear layer this method will fail if the module has not been called
    at least once.
    The contract of reverse is that the reversed module will accept the output
    of the parent module as input and produce an output which is the input size
    of the parent.
    >>> mlp = hk.nets.NoisyMLP([1, 2, 3])
    >>> y = mlp(jnp.ones([1, 2]))
    >>> rev = mlp.reverse()
    >>> rev(y)
    DeviceArray(...)
    Args:
      activate_final: Whether the final layer of the NoisyMLP should be activated.
      name: Optional name for the new module. The default name will be the name
        of the current module prefixed with ``"reversed_"``.
    Returns:
      A NoisyMLP instance which is the reverse of the current instance. Note these
      instances do not share weights and, apart from being symmetric to each
      other, are not coupled in any way.
    """

    if activate_final is None:
      activate_final = self.activate_final
    if name is None:
      name = self.name + "_reversed"

    return NoisyMLP(
        output_sizes=(layer.input_size for layer in reversed(self.layers)),
        w_init=self.w_mu_init,
        b_init=self.b_mu_init,
        with_bias=self.with_bias,
        activation=self.activation,
        activate_final=activate_final,
        name=name)
