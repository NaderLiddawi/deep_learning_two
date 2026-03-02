"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        self.classifier = nn.Linear(3 * h * w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.classifier(x.view(x.size(0), -1))


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
            hidden_dim: int, size of the hidden layer
        """
        super().__init__()

        input_dim = 3 * h * w
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.network(x.view(x.size(0), -1))


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        """
        An MLP with multiple hidden layers and dropout regularization.

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
            dropout: float, dropout probability (0 = disabled)
        """
        super().__init__()

        input_dim = 3 * h * w

        # First layer: input -> hidden
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # LayerNorm works at any batch size; BatchNorm1d breaks at batch=1
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        # Subsequent hidden layers
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        # Final projection to num_classes (no activation — raw logits)
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.network(x.view(x.size(0), -1))


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,   # MUST stay 128 — 256 exceeds 10 MB size cap
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        """
        A deep MLP where each hidden layer has a residual (skip) connection
        and dropout regularization to prevent overfitting.

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int
            hidden_dim: int, size of hidden layers
            num_layers: int, number of residual hidden layers
            dropout: float, dropout probability
        """
        super().__init__()

        input_dim = 3 * h * w

        # Project raw flattened input up to hidden_dim once.
        # All residual blocks operate in hidden_dim space so dims always match.
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)   # LayerNorm works at any batch size; BatchNorm1d breaks at batch=1

        # Each residual block: two linear transforms with LN, plus a dropout
        # The identity skip is added *after* the block, then ReLU is applied.
        self.blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.blocks.append(block)
            self.dropouts.append(nn.Dropout(dropout))

        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten and project into the residual stream
        out = self.activation(self.input_norm(self.input_proj(x.view(x.size(0), -1))))

        # Each block: out = ReLU(F(out) + out), then dropout
        for block, drop in zip(self.blocks, self.dropouts):
            out = drop(self.activation(block(out) + out))

        return self.output(out)


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r