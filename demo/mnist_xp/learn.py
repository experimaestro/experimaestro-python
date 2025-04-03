import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim as optim
import torchvision.transforms as transforms
from experimaestro import (
    Config,
    Constant,
    Meta,
    Param,
    PathGenerator,
    Task,
    field,
    LightweightTask,
    tqdm,
)

from .data import LabelledImages


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --8<-- [start:cnn]
class CNN(Config, nn.Module):
    """Defines a CNN model"""

    n_layers: Param[int] = 2
    """Number of Hidden layers"""

    hidden_dim: Param[int] = 64
    """Number of hidden units"""

    kernel_size: Param[int] = 3
    """Kernel size of the CNN"""
    # --8<-- [end:cnn]

    def __post_init__(self):
        """Simple CNN module with n_layers hidden layers and hidden_dim hidden
        units"""
        # create a list of hidden CNN layers with ReLU activation
        self.layers = nn.Sequential()
        for i in range(self.n_layers):
            self.layers.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_channels=1 if i == 0 else self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding="same",
                ),
            )
            self.layers.add_module(f"relu{i}", nn.ReLU())

        # pooling layer to reduce the size of the output to 13x13
        self.layers.add_module("pool", nn.MaxPool2d(kernel_size=2))

        # output layer
        self.output = nn.Linear(self.hidden_dim * 14 * 14, 10)

        logging.info(
            f"Model created with {self.n_layers} layers "
            f"of dimension {self.hidden_dim}:"
            f"{count_parameters(self)} parameters in total"
        )

    def forward(self, x):
        # apply the CNN layers to the input
        x = self.layers(x)
        # flatten the output
        x = x.view(x.size(0), -1)
        # apply the output layer
        x = self.output(x)
        return torch.log_softmax(x, dim=1)


class ParameterLoader(LightweightTask):
    model: Param[CNN]
    path: Meta[Path]

    def execute(self):
        state_dict = torch.load(self.path)
        self.model.load_state_dict(state_dict)


class Evaluate(Task):
    model: Param[CNN]

    data: Param[LabelledImages]

    batch_size: Meta[int] = 64
    """Batch size"""

    results_path: Meta[Path] = field(default_factory=PathGenerator("results.csv"))
    """Path to store tensorboard logs"""

    version: Constant[str] = "2"
    """Task version (can be changed if the algorithm changes and invalidate
    previous runs)"""

    @torch.no_grad()
    def execute(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        dataset = self.data.torchvision_dataset(transform=transform)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        logging.info(f"Dataset loaded with {len(dataset)} samples\n")

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0

        for images, labels in tqdm(test_loader):
            output = self.model(images)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        # Compute mean results
        loss = test_loss / len(dataset)
        accuracy = correct / len(dataset)

        with self.results_path.open("w") as fh:
            print("loss,accuracy", file=fh)  # noqa: T201
            print(f"{loss},{accuracy}", file=fh)  # noqa: T201


# Note that we use docstrings to document the class and the parameters
# This can be used to generate automatically a documentation
class Learn(Task):
    """Learn to classify an image into a pre-defined set of classes"""

    runpath: Meta[Path] = field(default_factory=PathGenerator("runs"))
    """Path to store tensorboard logs"""

    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""

    data: Param[LabelledImages]
    """Train data are labelled images"""

    model: Param[CNN]
    """The model we are training"""

    epochs: Param[int] = 1
    """number of epochs to train the model"""

    n_val: Param[int] = 100
    """number of steps between validation and logging"""

    lr: Param[float] = 1e-2
    """learning rate"""

    batch_size: Param[int] = 64
    """Batch size"""

    # can be changed if needed to rerun the task with same parameters
    version: Constant[str] = "3"
    """Task version (can be changed if the algorithm changes and invalidate
    previous runs)"""

    def task_outputs(self, dep):
        return dep(ParameterLoader(model=self.model, path=self.parameters_path))

    def execute(self):
        # Tensorboard writer
        writer = SummaryWriter(self.runpath)

        # Load and preprocess the MNIST dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        trainset = self.data.torchvision_dataset(transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True
        )

        logging.info(f"MNIST loaded\n - Train: {len(trainset)} samples\n")
        logging.debug(f"Image shape: {trainset[0][0].shape}")

        logging.info("Creating the CNN model")
        # Create the CNN model
        model = self.model
        model.train()

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Train the model
        it = 0
        logging.info("Training the CNN model ...")

        for epoch in tqdm(range(self.epochs)):
            running_loss = 0
            for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}"):
                it += 1
                output = model(images)

                # Compute the loss and update the model parameters
                optimizer.zero_grad()
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                writer.add_scalar("loss", loss.item(), it)

        # Saves the model
        torch.save(self.model.state_dict(), self.parameters_path)
