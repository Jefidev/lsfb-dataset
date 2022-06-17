import torch
from torch import jit, nn, Tensor
from torch.nn import init, Parameter
from typing import List, Tuple
import math


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()

        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.init_weights(hidden_size)

    def init_weights(self, hidden_size: int):
        stdv = 1.0 / math.sqrt(hidden_size)
        init.uniform_(self.weight_ih, -stdv, stdv)
        init.uniform_(self.weight_hh, -stdv, stdv)
        init.uniform_(self.bias_ih)
        init.uniform_(self.bias_hh)

    @jit.script_method
    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        gates = (
            torch.mm(x, self.weight_ih.t()) + self.bias_ih +
            torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        )
        forget_gate, in_gate, cell_gate, out_gate = gates.chunk(4, 1)

        forget_gate = torch.sigmoid(forget_gate)
        in_gate = torch.sigmoid(in_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    @jit.script_method
    def init_state(self, x: Tensor):
        batch_size = x.size(0)
        hx = x.new_zeros(batch_size, self.hidden_size)
        cx = x.new_zeros(batch_size, self.hidden_size)
        hx.requires_grad_(False)
        cx.requires_grad_(False)
        return hx, cx

    @jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        state = self.init_state(x)
        inputs = x.permute(1, 0, 2).unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for x_t in inputs:
            out, state = self.cell(x_t, state)
            outputs.append(out)

        return torch.stack(outputs).permute(1, 0, 2), state


class LSTMClassifier(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(LSTMClassifier, self).__init__()
        self.lstm = LSTMLayer(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    @jit.script_method
    def forward(self, x: Tensor):
        x, _ = self.lstm(x)
        return self.fc(x)