import runpy
import os
import sys
import types
from pathlib import Path

import torch
from torch import nn


transformers_stub = types.ModuleType("transformers")
transformers_stub.BertTokenizer = object
sys.modules["transformers"] = transformers_stub

module = runpy.run_path(
    os.environ.get("TRANSFORMER_SOURCE", str(Path(__file__).with_name("transformer.py"))),
    run_name="happy_llm_transformer_test",
)

ModelArgs = module["ModelArgs"]
MultiHeadAttention = module["MultiHeadAttention"]
Transformer = module["Transformer"]


class CaptureDecoder(nn.Module):
    def forward(self, decoder_input, encoder_output):
        self.decoder_input = decoder_input
        self.encoder_output = encoder_output
        return decoder_input


def main():
    args = ModelArgs(8, 2, 8, 0.0, 8, 32, 8, 1)
    model = Transformer(args)
    decoder = CaptureDecoder()
    model.transformer.decoder = decoder

    source = torch.tensor([[1, 2, 3, 4, 5]])
    shifted_target = torch.tensor([[6, 7, 8]])
    targets = torch.tensor([[7, 8, 9]])
    logits, loss = model(source, outputs=shifted_target, targets=targets)

    try:
        model(source, targets=targets)
    except ValueError as error:
        assert "outputs" in str(error)
    else:
        raise AssertionError("training without shifted outputs should fail clearly")

    assert decoder.encoder_output.shape[1] == source.shape[1]
    assert decoder.decoder_input.shape[1] == shifted_target.shape[1]
    assert logits.shape == (1, shifted_target.shape[1], args.vocab_size)
    assert torch.isfinite(loss)

    attention = MultiHeadAttention(args)
    query = torch.randn(1, 3, args.n_embd)
    key_value = torch.randn(1, 5, args.n_embd)
    assert attention(query, key_value, key_value).shape == query.shape

    print("Seq2Seq shifted-target and unequal-length attention regression passed")


if __name__ == "__main__":
    main()
