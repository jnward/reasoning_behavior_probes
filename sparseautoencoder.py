import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_in=None,
        n_features=None,
        k=None,
        d_hidden=None,
        n_hidden=0,
        apply_preencoder_bias=True,
        dtype=torch.float32,
        device="cpu",
        encoder=None,
        decoder=None,
        normalize_activations=False,
    ):
        super().__init__()

        self.normalize_activations = normalize_activations

        # Initialize from provided components or parameters
        if encoder is not None and decoder is not None:
            self._init_from_encoder_decoder(encoder, decoder)
        elif encoder is not None:
            self._init_from_encoder(encoder, dtype, device)
        elif decoder is not None:
            self._init_from_decoder(
                decoder, k, d_hidden, n_hidden, apply_preencoder_bias, dtype, device
            )
        elif all(x is not None for x in [d_in, n_features, k]):
            self._init_from_params(
                d_in,
                n_features,
                k,
                d_hidden,
                n_hidden,
                apply_preencoder_bias,
                dtype,
                device,
            )
        else:
            raise ValueError(
                "Must either provide encoder and/or decoder, or all of: d_in, n_features, k"
            )

    def _init_from_params(
        self,
        d_in,
        n_features,
        k,
        d_hidden,
        n_hidden,
        apply_preencoder_bias,
        dtype,
        device,
    ):
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.d_hidden = d_hidden
        self.n_hidden = n_hidden
        self.apply_preencoder_bias = apply_preencoder_bias

        # Create decoder first to use its bias for encoder
        self.decoder = Decoder(n_features, d_in, dtype, device)
        self.encoder = Encoder(
            d_in,
            n_features,
            k,
            d_hidden,
            n_hidden,
            preencoder_bias=self.decoder.b_dec if apply_preencoder_bias else None,
            apply_preencoder_bias=apply_preencoder_bias,
            dtype=dtype,
            device=device,
        )
        # initialize encoder to be the transpose of the decoder
        self.encoder.W_enc.data = self.decoder.W_dec.T.clone()

    def _init_from_encoder_decoder(self, encoder, decoder):
        assert (
            encoder.n_features == decoder.n_features
        ), "Encoder and decoder must have the same number of features"
        assert (
            encoder.d_in == decoder.d_out
        ), "Encoder input dimension must match decoder output dimension"

        self.d_in = encoder.d_in
        self.n_features = encoder.n_features
        self.k = encoder.k
        self.d_hidden = encoder.d_hidden
        self.n_hidden = encoder.n_hidden
        self.apply_preencoder_bias = encoder.apply_preencoder_bias
        self.encoder = encoder
        self.decoder = decoder

    def _init_from_encoder(self, encoder, dtype, device):
        self.d_in = encoder.d_in
        self.n_features = encoder.n_features
        self.k = encoder.k
        self.d_hidden = encoder.d_hidden
        self.n_hidden = encoder.n_hidden
        self.apply_preencoder_bias = encoder.apply_preencoder_bias
        self.encoder = encoder

        # Create decoder normally
        self.decoder = Decoder(self.n_features, self.d_in, dtype=dtype, device=device)

        # If encoder has a preencoder bias, modify both components
        if encoder.preencoder_bias is not None:
            logging.warning(
                "When initializing an SEA from a given encoder and no given decoder, the existing encoder's preencoder_bias will become a buffer in the encoder, and the new decoder will adopt the trainable parameter as decoder.b_dec."
            )
            # Convert encoder's preencoder_bias from Parameter to buffer
            bias_data = encoder.preencoder_bias.data
            del encoder.preencoder_bias
            # Set decoder bias to match encoder's original bias
            self.decoder.b_dec.data.copy_(bias_data)
            # Make encoder use decoder's bias
            encoder.register_buffer("preencoder_bias", self.decoder.b_dec)

    def _init_from_decoder(
        self, decoder, k, d_hidden, n_hidden, apply_preencoder_bias, dtype, device
    ):
        if k is None:
            raise ValueError("Must provide k when initializing from decoder only")

        self.decoder = decoder
        self.d_in = decoder.d_out
        self.n_features = decoder.n_features
        self.k = k
        self.d_hidden = d_hidden
        self.n_hidden = n_hidden
        self.apply_preencoder_bias = apply_preencoder_bias

        # Create new encoder using decoder's bias
        self.encoder = Encoder(
            self.d_in,
            self.n_features,
            self.k,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
            preencoder_bias=self.decoder.b_dec if apply_preencoder_bias else None,
            apply_preencoder_bias=apply_preencoder_bias,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def from_pretrained(
        release: str,
        sae_id: str,
        k: int,
        d_hidden=None,
        n_hidden=0,
        device: str = "cpu",
    ):
        sae_lens_autoencoder, _, _ = SAELensAutoencoder.from_pretrained(
            release, sae_id, device
        )

        n_features = sae_lens_autoencoder.cfg.d_sae
        d_in = sae_lens_autoencoder.cfg.d_in
        apply_preencoder_bias = sae_lens_autoencoder.cfg.apply_b_dec_to_input

        decoder = Decoder(
            n_features,
            d_in,
            dtype=sae_lens_autoencoder.dtype,
            device=device,
        )
        decoder.W_dec.data = sae_lens_autoencoder.W_dec.data
        decoder.b_dec.data = sae_lens_autoencoder.b_dec.data

        encoder = Encoder(
            d_in,
            n_features,
            k,
            d_hidden,
            n_hidden,
            preencoder_bias=decoder.b_dec if apply_preencoder_bias else None,
            apply_preencoder_bias=apply_preencoder_bias,
            dtype=sae_lens_autoencoder.dtype,
            device=device,
        )
        encoder.W_enc.data = sae_lens_autoencoder.W_enc.data
        encoder.b_enc.data = sae_lens_autoencoder.b_enc.data

        return SparseAutoencoder(encoder=encoder, decoder=decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x: torch.Tensor):
        if not self.normalize_activations:
            return self.decode(self.encode(x))
        x_norm = x.norm(dim=1, keepdim=True)
        scale_factor = 1 / math.sqrt(self.d_in)
        x_scaled = x * scale_factor / x_norm
        reconstruction = self.decode(self.encode(x_scaled))
        return reconstruction * x_norm / scale_factor

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, n_features, d_out, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.n_features = n_features
        self.d_out = d_out
        self.dtype = dtype
        self.device = device
        self._trainable = True

        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(n_features, d_out, dtype=dtype, device=device)
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(d_out, dtype=dtype, device=device))
        self.normalize_weights()

    def forward(self, x):
        return x @ self.W_dec + self.b_dec

    def normalize_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

    def remove_parallel_gradient_component(self):
        parallel_component = torch.einsum(
            "nd, nd -> n",
            self.W_dec.grad,
            self.W_dec.data,
        )
        self.W_dec.grad -= torch.einsum(
            "n, nd -> nd",
            parallel_component,
            self.W_dec.data,
        )

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self.W_dec.requires_grad = value
        self.b_dec.requires_grad = value
        self._trainable = value


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class Encoder(nn.Module):
    def __init__(
        self,
        d_in,
        n_features,
        k,
        d_hidden=None,
        n_hidden=0,
        preencoder_bias=None,
        apply_preencoder_bias=True,
        # encoder_weights=None,
        # encoder_bias=None,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.k = k
        self.apply_preencoder_bias = apply_preencoder_bias
        self.dtype = dtype
        self.device = device
        self._trainable = True

        if preencoder_bias is None and apply_preencoder_bias:
            logging.warning(
                "No preencoder bias was probided, but apply_preencoder bias is True. An independent preencoder bias will be trained; set apply_preencoder_bias=False if this is not desired."
            )

        if preencoder_bias is not None and apply_preencoder_bias:
            self.register_buffer("preencoder_bias", preencoder_bias)
        elif apply_preencoder_bias:
            self.preencoder_bias = nn.Parameter(
                torch.zeros(d_in, dtype=dtype, device=device)
            )
        else:
            self.preencoder_bias = None

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(d_in, n_features, dtype=dtype, device=device)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(n_features, dtype=dtype, device=device))

        if n_hidden > 0:
            assert d_hidden is not None, "Hidden_dim must be provided if n_hidden > 0"
            mlp_layers = []
            mlp_layers.append(nn.Linear(d_in, d_hidden, bias=True, device=device))
            mlp_layers.append(nn.ReLU())
            for _ in range(n_hidden - 1):
                mlp_layers.append(
                    nn.Linear(d_hidden, d_hidden, bias=True, device=device)
                )
                mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(d_hidden, n_features, bias=True, device=device))
            self.mlp_path = nn.Sequential(*mlp_layers)
        else:
            self.mlp_path = None

        self.device = device

    def top_k(self, x):
        batch_size = x.size(0)
        total_k = self.k * batch_size
        
        # Flatten batch and feature dimensions
        x_flat = x.reshape(-1)
        
        # Get topk values and create a mask
        topk_values = torch.topk(x_flat, total_k)[0]
        threshold = topk_values[-1]

            
        # Apply threshold to original tensor
        return x * (x >= threshold)

    def forward(self, x):
        if self.preencoder_bias is not None:
            x = x - self.preencoder_bias
        y = x @ self.W_enc + self.b_enc
        if self.mlp_path is not None:
            y += self.mlp_path(x)
        activations = F.relu(y)
        return self.top_k(activations)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self.W_enc.requires_grad = value
        self.b_enc.requires_grad = value
        self._trainable = value
