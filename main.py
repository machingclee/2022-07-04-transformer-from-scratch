from src.transformer import Transformer
from src.greedy_decoder import GreedyDecoder
from src.device import device
from src.dataset import data_loader, tgt_word_index, tgt_index_word
import torch


def main():
    transformer = Transformer().to(device)
    model_path = "pths/model_epoch_30.pth"

    if model_path is not None:
        transformer.load_state_dict(torch.load(model_path))

    greedy_decoder = GreedyDecoder(transformer)
    enc_inputs, _, _ = next(iter(data_loader))
    enc_inputs = enc_inputs.to(device)
    for i in range(len(enc_inputs)):
        greedy_dec_input = greedy_decoder.decode(
            enc_inputs[i].view(1, -1),
            start_symbol=tgt_word_index["S"]
        )
        predict, _, _, _ = transformer(
            enc_inputs[i].view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        print(enc_inputs[i], '->', [tgt_index_word[n.item()]
                                    for n in predict.squeeze()])


if __name__ == "__main__":
    main()
