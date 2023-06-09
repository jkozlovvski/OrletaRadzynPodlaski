import torch


class Ensemble(torch.nn.Module):
    def __init__(self, image_embedding, text_embedding):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(image_embedding + text_embedding, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 21),
            torch.nn.Softmax(dim=1),  # (batch, logits)
        )

    def forward(self, img, text):
        input = torch.cat([img, text], axis=1)
        return self.mlp(input)
