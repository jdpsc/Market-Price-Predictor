import math
from typing import List

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import lightning as L
from tqdm import tqdm


class StockDataset(Dataset):
    def __init__(
        self,
        stock_data: pd.DataFrame,
        target_col: int,
        days_ahead: int = 1,
        days_lag: int = 7,
    ):
        self.stock_data = stock_data
        self.days_ahead = days_ahead
        self.days_lag = days_lag
        self.target_col = target_col

        self.max_seq_len = days_lag

    def __getitem__(self, index):

        x = self.stock_data[index : (index + self.days_lag), self.target_col].copy()
        y = self.stock_data[
            (index + self.days_lag) : (index + self.days_lag + self.days_ahead),
            self.target_col,
        ].copy()

        mask = np.isnan(x)
        x[mask] = 0

        x = x.reshape(-1)
        mask = mask.reshape(-1)
        y = y.reshape(-1)

        return_dict = {"x": x, "mask": mask, "y": y}

        return return_dict

    def __len__(self):
        return len(self.stock_data) - self.days_ahead - self.days_lag


def get_valid_indices(dataset: Dataset):
    indices = []

    for idx in tqdm(range(len(dataset))):
        item = dataset.__getitem__(idx)
        if np.isnan(item["y"]).any():
            continue
        if np.all(item["mask"]):
            continue

        indices.append(idx)

    return indices


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.pe = torch.zeros(1, max_seq_len, d_model)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.pe[: x.size(1)].to(device=x.device)


class Transformer(torch.nn.Module):

    def __init__(self, embed_dim, nhead, num_layers, max_seq_len, days_ahead):
        super().__init__()

        self.embed_layer = torch.nn.Linear(1, embed_dim)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.positional_encoder = PositionalEncoding(
            d_model=embed_dim, max_seq_len=max_seq_len
        )
        self.linear = torch.nn.Linear(embed_dim, days_ahead)

    def forward(self, x, mask):

        x = x.unsqueeze(-1)

        embeddings = self.embed_layer(x)
        embeddings_with_pos = self.positional_encoder(embeddings)
        transformer_embeddings = self.transformer_encoder(
            embeddings_with_pos, src_key_padding_mask=mask
        )
        final_embeddings = torch.mean(transformer_embeddings, dim=1)
        preds = self.linear(final_embeddings)
        return preds


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.learning_rate = learning_rate
        self.model = model
        self.loss = torch.nn.L1Loss()

    def forward(self, batch):
        return self.model(batch["x"], batch["mask"])

    def _shared_step(self, batch):
        preds = self(batch)
        loss = self.loss(preds, batch["y"])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
