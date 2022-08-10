
import torch
from src.device import device
from src.transformer import Transformer
from src.dataset import tgt_word_index


class GreedyDecoder():
    def __init__(self, transformer: Transformer):
        self.transformer = transformer

    def decode(self, enc_input, start_symbol):
        """
        For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
        target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
        Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
        :param model: Transformer Model
        :param enc_input: The encoder input
        :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
        :return: The target input
        """

        enc_outputs, _ = self.transformer.encoder(enc_input)
        dec_input = torch.zeros(1, 0).type_as(enc_input.data)
        terminated = False
        next_symbol = start_symbol
        while not terminated:
            dec_input = torch.cat(
                [
                    dec_input.detach(),
                    torch.tensor(
                        [[next_symbol]],
                        dtype=enc_input.dtype
                    ).to(device)
                ],
                -1
            )
            dec_outputs, _, _ = self.transformer.decoder(
                dec_input,
                enc_input,
                enc_outputs
            )
            projected = self.transformer.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word

            if next_symbol == tgt_word_index["."]:
                terminated = True

            print(next_word)

        return dec_input
