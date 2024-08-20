from torch import nn


class TransformerModel(nn.Module):
    def __init__(self, input_size=7, d_model=64, nhead=8, num_layers=2, output_size=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    batch_first=True)  # 设置 batch_first=True
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # 只取最后一个时间步的输出
        return x