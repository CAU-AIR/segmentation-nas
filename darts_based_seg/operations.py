import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MixedOp(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride=2,
        dilation=1,
        output_padding=1,
        bias=False,
        d_model_reduction=4,
        nhead=4,
    ):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()

        # ConvTranspose2d operations
        self._ops.append(
            nn.ConvTranspose2d(
                C_in,
                C_out,
                3,
                stride=stride,
                padding=1,
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
            )
        )
        self._ops.append(
            nn.ConvTranspose2d(
                C_in,
                C_out,
                5,
                stride=stride,
                padding=2,
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
            )
        )
        self._ops.append(
            nn.ConvTranspose2d(
                C_in,
                C_out,
                7,
                stride=stride,
                padding=3,
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
            )
        )

        # Transformer Encoder
<<<<<<< HEAD
        # d_model = C_out * 8 * 8 // d_model_reduction
        d_model = C_out
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=1
        )

        # self.fc1 = nn.Linear(C_out * 8 * 8, d_model)
        # self.fc2 = nn.Linear(d_model, C_out * 8 * 8)
=======
        d_model = C_out * 8 * 8  # Adjust d_model as necessary for your input size
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=8)
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(C_out * 8 * 8, d_model)
        self.fc2 = nn.Linear(d_model, C_out * 8 * 8)
>>>>>>> a40728a3be156cbdcb436b509fbb0270ca500eee

        self.bn = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)
        self.alphas = nn.Parameter(
            torch.Tensor([1.0 / 4] * 4).cuda(), requires_grad=True
        )

    def clip_alphas(self):
        with torch.no_grad():
            self.alphas.clamp_(0, 1)
            alpha_sum = self.alphas.sum()
            self.alphas.div_(alpha_sum)

    def forward(self, x):
        # Apply ConvTranspose2d operations
<<<<<<< HEAD
        x_conv = sum(alpha * op(x) for alpha, op in zip(self.alphas[:3], self._ops[:3]))

        # Transformer operation
        b, c, h, w = x_conv.size()

        x_flat = x_conv.view(b, c, -1).permute(0, 2, 1)  # (batch_size, seq_len, d_model)
        x_transformed = self.transformer_encoder(x_flat)
        x_transformed = x_transformed.permute(0, 2, 1).view(b, c, h, w)
        
        # x_flat = x_conv.view(b, -1)
        # x_transformed = self.fc1(x_flat)
        # x_transformed = x_transformed.unsqueeze(1)
        # x_transformed = self.transformer_encoder(x_transformed)
        # x_transformed = x_transformed.squeeze(1)
        # x_transformed = self.fc2(x_transformed)
        # x_transformed = x_transformed.view(b, c, h, w)
=======
        x_conv = sum(alpha * op(x)
                     for alpha, op in zip(self.alphas[:3], self._ops[:3]))

        # Transformer operation
        b, c, h, w = x_conv.size()
        x_flat = x_conv.view(b, -1)
        x_transformed = self.fc1(x_flat)
        x_transformed = x_transformed.unsqueeze(1)
        x_transformed = self.transformer_encoder(x_transformed)
        x_transformed = x_transformed.squeeze(1)
        x_transformed = self.fc2(x_transformed)
        x_transformed = x_transformed.view(b, c, h, w)
>>>>>>> a40728a3be156cbdcb436b509fbb0270ca500eee

        # Combine results
        x = x_transformed
        x = self.relu(x)
        x = self.bn(x)
        return x

    def get_max_alpha_idx(self):
        # return the index of the maximum alpha
        return torch.argmax(self.alphas).item()

    def get_max_op(self):
        # return the operation with the maximum alpha
        if self.get_max_alpha_idx() == 3:
            return self.transformer_encoder
        else:
            return self._ops[self.get_max_alpha_idx()]
