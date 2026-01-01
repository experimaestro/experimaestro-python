from abc import ABC, abstractmethod
from pathlib import Path
import logging
from torchvision.datasets import VisionDataset, MNIST
from experimaestro import Meta, Param
from datamaestro import dataset, Base, Context
from datamaestro.data.ml import Supervised
from datamaestro.download.custom import custom_download


class LabelledImages(Base, ABC):
    @abstractmethod
    def torchvision_dataset(self, **kwargs) -> VisionDataset: ...


class MNISTLabelledImages(LabelledImages):
    root: Meta[Path]
    train: Param[bool]

    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        return MNIST(self.root, train=self.train, **kwargs)


def download_mnist(context: Context, root: Path, force=False):
    logging.info("Downloading in %s", root)
    for train in [False, True]:
        MNIST(root, train=train, download=True)


@custom_download("root", download_mnist)
@dataset(id="com.lecun.mnist")
def mnist(root: Path) -> Supervised[LabelledImages, None, LabelledImages]:
    """This corresponds to a dataset with an ID `com.lecun.mnist`"""
    return Supervised.C(
        train=MNISTLabelledImages.C(root=root, train=True),
        test=MNISTLabelledImages.C(root=root, train=False),
    )
