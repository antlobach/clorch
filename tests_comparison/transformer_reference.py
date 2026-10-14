import json
import sys

import torch


torch.set_num_threads(1)
torch.manual_seed(0)

SPECS = {
    "in_proj_weight": (0.002, -0.10),
    "in_proj_bias": (0.003, -0.05),
    "out_proj.weight": (0.004, -0.10),
    "out_proj.bias": (0.005, 0.02),
    "linear1.weight": (0.001, -0.08),
    "linear1.bias": (0.002, 0.01),
    "linear2.weight": (0.0015, -0.06),
    "linear2.bias": (0.003, -0.02),
    "norm1.weight": (0.001, 1.0),
    "norm1.bias": (0.002, -0.01),
    "norm2.weight": (0.0015, 0.9),
    "norm2.bias": (0.001, 0.02),
}


def values_like(tensor, scale, offset):
    return (torch.arange(tensor.numel(), dtype=torch.float32) * scale + offset).reshape(tensor.shape)


def initialize(module):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            suffix = name.removeprefix("self_attn.")
            scale, offset = SPECS[suffix]
            parameter.copy_(values_like(parameter, scale, offset))


def interleaved_rope(x, cosines, sines):
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even, odd = pairs.unbind(-1)
    rotated = torch.stack((-odd, even), dim=-1).reshape_as(x)
    return x * cosines.reshape(1, cosines.shape[0], 1, -1) + rotated * sines.reshape(1, sines.shape[0], 1, -1)


def main(output_path):
    attention = torch.nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True)
    layer = torch.nn.TransformerEncoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    initialize(attention)
    initialize(layer)
    attention.eval()
    layer.eval()

    query = (torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) * 0.01) - 0.2
    context = (torch.arange(80, dtype=torch.float32).reshape(2, 5, 8) * 0.008) - 0.15
    padding_mask = torch.tensor(
        [[False, False, False, False, False],
         [False, False, False, True, True]]
    )
    rope_input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]],
                                [[-1.0, 0.5, 2.0, -3.0]]]])
    angles = torch.tensor([[0.2, 0.2, 0.6, 0.6],
                           [0.4, 0.4, 0.8, 0.8]])

    with torch.inference_mode():
        self_attention = attention(query, query, query, need_weights=False)[0]
        cross_attention = attention(
            query,
            context,
            context,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        transformer = layer(query)
        rope = interleaved_rope(rope_input, torch.cos(angles), torch.sin(angles))

    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(
            {
                "self_attention": self_attention.tolist(),
                "cross_attention": cross_attention.tolist(),
                "transformer": transformer.tolist(),
                "interleaved_rope": rope.tolist(),
            },
            output,
        )


if __name__ == "__main__":
    main(sys.argv[1])
