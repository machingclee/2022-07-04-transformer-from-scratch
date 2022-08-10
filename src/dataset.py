import torch
import torch.utils.data as Data


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_word_index = {'P': 0, 'ich': 1,
                  'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_word_index)

tgt_word_index = {'P': 0, 'i': 1, 'want': 2, 'a': 3,
                  'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
tgt_index_word = {i: w for i, w in enumerate(tgt_word_index)}
tgt_vocab_size = len(tgt_word_index)

src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input_, dec_input_, dec_output_ = sentences[i]
        # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        enc_input = [src_word_index[n] for n in enc_input_.split()]
        # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_input = [tgt_word_index[n] for n in dec_input_.split()]
        # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
        dec_output = [tgt_word_index[n] for n in dec_output_.split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


class MyDataSet(Data.Dataset):
    def __init__(self):
        super(MyDataSet, self).__init__()
        enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


data_loader = Data.DataLoader(MyDataSet(), batch_size=2, shuffle=True)
