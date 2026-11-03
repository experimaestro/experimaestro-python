import json
import math
import random
from pathlib import Path

from experimaestro import Config, Meta, Param, PathGenerator, Task, field


class Dataset(Config):
    """Noisy points on the line y = 2x + 1"""

    size: Param[int]
    """Number of points"""

    seed: Param[int]
    """Random seed (two seeds give two different datasets)"""

    noise: Param[float] = field(ignore_default=0.1)
    """Standard deviation of the noise"""

    def points(self) -> list[tuple[float, float]]:
        rng = random.Random(self.seed)
        xs = [rng.uniform(-1, 1) for _ in range(self.size)]
        return [(x, 2 * x + 1 + rng.gauss(0, self.noise)) for x in xs]


class LinearModel(Config):
    """A line y = a x + b, whose parameters are read from a file"""

    path: Meta[Path]
    """The file containing the parameters"""

    def __post_init__(self):
        parameters = json.loads(self.path.read_text())
        self.a, self.b = parameters["a"], parameters["b"]

    def predict(self, x: float) -> float:
        return self.a * x + self.b


class Learn(Task):
    """Fit a line to a dataset by gradient descent"""

    data: Param[Dataset]
    """Training data"""

    lr: Param[float]
    """Learning rate"""

    epochs: Param[int] = field(ignore_default=100)
    """Number of passes over the data"""

    model_path: Meta[Path] = field(default_factory=PathGenerator("model.json"))
    """Where the learned parameters are written"""

    def task_outputs(self, dep) -> LinearModel:
        return dep(LinearModel.C(path=self.model_path))

    def execute(self):
        a, b = 0.0, 0.0
        points = self.data.points()
        for epoch in range(self.epochs):
            for x, y in points:
                error = a * x + b - y
                a -= self.lr * error * x
                b -= self.lr * error
            print(f"epoch {epoch}: a={a:.3f}, b={b:.3f}")
            if not (math.isfinite(a) and math.isfinite(b)):
                raise RuntimeError(f"Training diverged: lr={self.lr} is too large")
        self.model_path.write_text(json.dumps({"a": a, "b": b}))


class Evaluate(Task):
    """Mean squared error of a model on a dataset"""

    model: Param[LinearModel]
    """The model to evaluate"""

    data: Param[Dataset]
    """Test data"""

    results_path: Meta[Path] = field(default_factory=PathGenerator("results.json"))

    def execute(self):
        points = self.data.points()
        mse = sum((self.model.predict(x) - y) ** 2 for x, y in points) / len(points)
        self.results_path.write_text(json.dumps({"mse": mse}))
