"""Noisy MLP module"""

from typing import Callable, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class NoisyLinear(hk.Module):
  """Noisy Linear module."""

  def __init__(
            self,
            output_size: int,
            with_bias: bool = True,
            w_mu_init: Optional[hk.initializers.Initializer] = None,
            b_mu_init: Optional[hk.initializers.Initializer] = None,
            w_sigma_init: Optional[hk.initializers.Initializer] = None,
            b_sigma_init: Optional[hk.initializers.Initializer] = None,
            name: Optional[str] = None,
            factorized_noise: bool = False
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
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_mu_init = w_mu_init
    self.b_mu_init = b_mu_init or jnp.zeros
    self.w_sigma_init = w_sigma_init
    self.b_sigma_init = b_sigma_init or jnp.zeros
    self.factorized = factorized_noise

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_mu_init = self.w_mu_init
    w_sigma_init = self.w_sigma_init
    
    # for initialization of weights
    if self.factorized:
        val_mu = 1. / np.sqrt(self.input_size)
        val_sigma = 0.5 / np.sqrt(self.input_size)
    else:
        val_mu = np.sqrt(3 / self.input_size)
        val_sigma = 0.017

    # init weights mu 
    if w_mu_init is None:
        w_mu_init = hk.initializers.RandomUniform(minval=-val_mu, maxval=val_mu)
    w_mu = hk.get_parameter("w_mu", [input_size, output_size], dtype, init=w_mu_init)
    
    # init weights sigma
    if w_sigma_init is None:
        w_sigma_init = hk.initializers.Constant(val_sigma)
    w_sigma = hk.get_parameter("w_sigma", [input_size, output_size], dtype, init=w_sigma_init)
    
    # calculate noise
    if self.factorized:
        e_noise_input = jax.random.normal(hk.next_rng_key(), (w_sigma.shape[0], 1))
        e_noise_output = jax.random.normal(hk.next_rng_key(), (1, w_sigma.shape[1]))
        e_noise_input = jnp.multiply(jnp.sign(e_noise_input), jnp.sqrt(jnp.abs(e_noise_input)))
        e_noise_output = jnp.multiply(jnp.sign(e_noise_output), jnp.sqrt(jnp.abs(e_noise_output)))
        w_noise = jnp.matmul(e_noise_input, e_noise_output)
    else:
        w_noise = jax.random.normal(hk.next_rng_key(), w_sigma.shape)
        
    # calculate output
    out_noisy = jnp.dot(inputs, jnp.add(w_mu, jnp.multiply(w_sigma, w_noise)))
    
    # add bias
    if self.with_bias:
        b_mu = hk.get_parameter("b_mu", [self.output_size], dtype, init=self.b_mu_init)
        b_sigma = hk.get_parameter("b_sigma", [self.output_size], dtype, init=self.b_sigma_init)
        b_mu = jnp.broadcast_to(b_mu, out_noisy.shape)
        b_sigma = jnp.broadcast_to(b_sigma, out_noisy.shape)
        b_noise = e_noise_output if self.factorized else jax.random.normal(hk.next_rng_key(), b_sigma.shape)
        out_noisy = out_noisy + jnp.add(b_mu, jnp.multiply(b_sigma, b_noise))
      
    return out_noisy


class NoisyMLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      with_bias=True,
      w_mu_init: Optional[hk.initializers.Initializer] = None,
      b_mu_init: Optional[hk.initializers.Initializer] = None,
      w_sigma_init: Optional[hk.initializers.Initializer] = None,
      b_sigma_init: Optional[hk.initializers.Initializer] = None,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
      factorized_noise: bool = False
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

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_mu_init = w_mu_init
    self.b_mu_init = b_mu_init
    self.w_sigma_init = w_sigma_init
    self.b_sigma_init = b_sigma_init
    self.activation = activation
    self.activate_final = activate_final
    self.factorized = factorized_noise
    layers = []
    for index, output_size in enumerate(output_sizes):
        layers.append(NoisyLinear(
            output_size=output_size,
            w_mu_init=w_mu_init,
            b_mu_init=b_mu_init,
            w_sigma_init=w_sigma_init,
            b_sigma_init=b_sigma_init,
            with_bias=with_bias,
            name="noisy_linear_%d" % index,
            factorized_noise=self.factorized)
        )
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
        w_mu_init=self.w_mu_init,
        b_mu_init=self.b_mu_init,
        w_sigma_init=self.w_sigma_init,
        b_sigma_init=self.b_sigma_init,
        with_bias=self.with_bias,
        activation=self.activation,
        activate_final=activate_final,
        name=name,
        factorized_noise=self.factorized)
