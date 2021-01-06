"""Basic definitions for the flows module."""


import torch.nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class RNNFlow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, rnn_dict, rnn_net: torch.nn.RNNBase = None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        self._rnn_dict = rnn_dict

        if rnn_net is not None:
            assert isinstance(rnn_net, torch.nn.Module), (
                "rnn_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "rnn_net=None"
            )
            self._rnn_net = rnn_net
        else:
            self._rnn_net = torch.nn.RNN()

    def _log_prob(self, inputs, context):
        h_0 = torch.zeros((self._rnn_dict["input_size"], context.shape[1], self._rnn_dict["hidden_size"]))
        embedded_context = self._rnn_net(context, h_0)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        h_0 = torch.zeros((self._rnn_dict["input_size"], context.shape[1], self._rnn_dict["hidden_size"]))
        embedded_context = self._rnn_net(context, h_0)
        noise = self._distribution.sample(num_samples, context=embedded_context)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=num_samples)

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(num_samples, context=embedded_context)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise
