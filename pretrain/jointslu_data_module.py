import pytorch_lightning as pl
from torch.utils.data import DataLoader
from os.path import exists

from pretrain.jointslu_dataset import JointSluDataset
from data.data_processing import read_jointslu_lines, \
    construct_jointslu_data, store_jointslu_labels


class JointsluDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, train_bsz=1, test_bsz=1):
        super().__init__()
        self.train_bsz = train_bsz
        self.test_bsz = test_bsz
        self.train_dataset, self.val_dataset, self.test_dataset = \
            None, None, None
        self.tokenizer = tokenizer

    def _create_data(self, split):
        return JointSluDataset.create_data(split, self.tokenizer)

    def prepare_data(self) -> None:
        """Read data from disk and convert it to
        train.json, test.json and val.json"""
        pass
        """
        train_val_lines = read_jointslu_lines("../data/atis.train.w-intent.iob")
        test_lines = read_jointslu_lines("../data/atis.test.w-intent.iob")
        length = len(train_val_lines)
        # split the data 8:2
        train_length = int(0.8 * length)
        construct_jointslu_data("train", train_val_lines[0:train_length])
        construct_jointslu_data("val", train_val_lines[train_length:])
        construct_jointslu_data("test", test_lines)
        store_jointslu_labels()
        """

    def setup(self, stage=None):
        """Load the data"""
        # Load train and validation data
        self.train_dataset = self._create_data('train')
        self.val_dataset = self._create_data('val')
        # Load test data
        self.test_dataset = self._create_data('test')
        assert self.train_dataset is not None
        assert self.val_dataset is not None
        assert self.test_dataset is not None

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_bsz,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.train_dataset.collate_dia_samples)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.train_bsz,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.val_dataset.collate_dia_samples)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_bsz,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.test_dataset.collate_dia_samples)
