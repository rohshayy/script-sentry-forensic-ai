import torch.nn as nn

class ScriptSentry(nn.Module):
    def __init__(self):
        super(ScriptSentry, self).__init__()
        # We split the model so we can access the "Style Layer" (128 neurons)
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, 47)

    def forward(self, x, return_features=False):
        # Flattening logic that works for both single images and batches
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        style_vector = self.features(x)
        if return_features:
            return style_vector
        return self.classifier(style_vector)