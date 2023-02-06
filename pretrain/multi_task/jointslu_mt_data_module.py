import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pretrain.multi_task.jointslu_mt_dataset import MTDataset


class MTDataModule(pl.LightningDataModule):
    def __init__(self, dataset, labels_version, scenario, tokenizer, train_bsz=1, test_bsz=1, few_shot=False):
        super().__init__()
        self.dataset = dataset
        self.labels_version = labels_version
        self.scenario = scenario
        self.train_bsz = train_bsz
        self.test_bsz = test_bsz
        self.train_dataset, self.val_dataset, self.test_dataset = \
            None, None, None
        self.tokenizer = tokenizer
        self.few_shot = few_shot

    def _create_data(self, split):
        data = MTDataset.create_data(
            self.dataset,
            self.labels_version,
            self.scenario,
            split,
            self.tokenizer,
            self.few_shot
        )
        return data

    def prepare_data(self) -> None:
        """Read data from disk and convert it to
        train.json, test.json and val.json"""
        pass

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
        loader = DataLoader(self.train_dataset,
                            batch_size=self.train_bsz,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=self.train_dataset.collate_dia_samples)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.test_bsz,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=self.val_dataset.collate_dia_samples)
        return loader

    def test_dataloader(self):

        loader = DataLoader(self.test_dataset,
                            batch_size=self.test_bsz,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=self.test_dataset.collate_dia_samples)
        return loader

    def predict_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=self.test_bsz,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=self.test_dataset.collate_dia_samples)
        return loader
