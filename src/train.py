import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os
from src.transformer import Transformer
from src.device import device
from torchsummary import summary
from torch import optim
from src.dataset import data_loader


def train():
    transformer = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)
    epochs = 30

    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in data_loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
                device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(
                enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1),
                  'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    state_dict = transformer.state_dict()
    torch.save(state_dict, os.path.join("pths", f"model_epoch_{epochs}.pth"))
