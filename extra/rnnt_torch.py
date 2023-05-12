# a rough copy of https://github.com/sooftware/RNN-Transducer
import torch
from torch import nn


class EncoderRNNT(nn.Module):
    supported_rnns = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}

    def __init__(self, input_size, hidden_size, output_size, num_layers, rnn_type="lstm", bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True, bidirectional=bidirectional)
        self.out_proj = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def __call__(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs.transpose(0, 1), input_lengths.cpu())
        outputs, _ = self.rnn(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = self.out_proj(outputs.transpose(0, 1))
        return outputs, input_lengths


class DecoderRNNT(nn.Module):
    supported_rnns = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}

    def __init__(self, num_classes, hidden_size, output_size, num_layers, rnn_type="lstm", sos_id=1, eos_id=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_size)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True, bidirectional=False)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def __call__(self, inputs, input_lengths=None, hidden_states=None):
        embedded = self.embedding(inputs)
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded.transpose(0, 1), input_lengths.cpu())
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = self.out_proj(outputs.transpose(0, 1))
        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs = self.out_proj(outputs)
        return outputs, hidden_states


class RNNTransducer(nn.Module):
    def __init__(self, num_classes, input_size, num_encoder_layers=4, num_decoder_layers=1, encoder_hidden_size=320, decoder_hidden_size=512, output_size=512, rnn_type="lstm", bidirectional=True, sos_id=1, eos_id=2):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = EncoderRNNT(input_size=input_size, hidden_size=encoder_hidden_size, output_size=output_size, num_layers=num_encoder_layers, rnn_type=rnn_type, bidirectional=bidirectional)
        self.decoder = DecoderRNNT(num_classes=num_classes, hidden_size=decoder_hidden_size, output_size=output_size, num_layers=num_decoder_layers, rnn_type=rnn_type, sos_id=sos_id, eos_id=eos_id)
        # TODO: is the input size correct?
        self.fc = nn.Linear(output_size * 2, num_classes, bias=False)

    def joint(self, encoder_outputs, decoder_outputs):
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)
            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)
            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])
        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs)
        return outputs

    def __call__(self, inputs, input_lengths, targets, target_lengths):
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs

    @torch.no_grad()
    def decode(self, encoder_output, max_length):
        pred_tokens, hidden_state = [], None
        decoder_input = encoder_output.new_tensor([[self.decoder.sos_id]], dtype=torch.long)
        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_states=hidden_state)
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)
        return torch.tensor(pred_tokens, dtype=torch.long)

    @torch.no_grad()
    def recognize(self, inputs, input_lengths):
        outputs = []
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)
        for encoder_output in encoder_outputs:
            decoder_seq = self.decode(encoder_output, max_length)
            outputs.append(decoder_seq)
        outputs = torch.stack(outputs, dim=1).transpose(0, 1)
        return outputs


if __name__ == "__main__":
    batch_size, sequence_length, dim = 3, 12345, 80
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    inputs = torch.rand(batch_size, sequence_length, dim).to(device)
    input_lengths = torch.tensor([12345, 12300, 12000], dtype=torch.int)
    targets = torch.tensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2], [1, 3, 3, 3, 3, 3, 4, 5, 2, 0], [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]], dtype=torch.long).to(device)
    target_lengths = torch.tensor([9, 8, 7], dtype=torch.long)
    model = nn.DataParallel(RNNTransducer(10, 80)).to(device)
    outputs = model(inputs, input_lengths, targets, target_lengths)
    outputs = model.module.recognize(inputs, input_lengths)
    print(outputs)