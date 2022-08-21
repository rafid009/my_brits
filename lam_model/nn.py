import torch
import torch.nn as nn

class net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size):
        super(net, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 2)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul + 1, 1)  # Budbreak

    def forward(self, x, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul


        # out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50

        # out_lt_90 = self.linear6(out_s)  # LT90

        # out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_50, h_next