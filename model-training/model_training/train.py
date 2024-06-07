import tempfile
from pathlib import Path
import os
import argparse

import hydra
from omegaconf import DictConfig
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from sklearn.preprocessing import StandardScaler
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dotenv import load_dotenv

from model_training.data import load_stock_data_from_feature_store
from model_training.feature_engineering import (
    DataFrameTransformer,
    add_returns,
    filter_columns,
)
from model_training.model import (
    StockDataset,
    get_valid_indices,
    Transformer,
    LightningModel,
)
from model_training.utils import get_logger

logger = get_logger(__name__)

load_dotenv()


class FullModelPipeline(mlflow.pyfunc.PythonModel):
    """Class to encapsulate the full model pipeline. Objects of this class will be logged to mlflow"""

    def __init__(
        self,
        data_transformer: DataFrameTransformer,
        model: Transformer,
        days_ahead: int,
        days_lag: int,
        target_col: int,
    ):
        self.data_transformer = data_transformer
        self.model = model
        self.days_ahead = days_ahead
        self.days_lag = days_lag
        self.target_col = target_col

    def predict(self, context, model_input):
        with torch.no_grad():
            infer_data = self.data_transformer.transform_input(model_input)
            inference_ds = StockDataset(
                stock_data=infer_data,
                target_col=self.target_col,
                days_ahead=self.days_ahead,
                days_lag=self.days_lag,
            )
            inference_loader = DataLoader(
                dataset=inference_ds, batch_size=1, num_workers=0, shuffle=False
            )
            preds = torch.cat(
                [self.model(batch["x"], batch["mask"]) for batch in inference_loader]
            )

            preds = preds.numpy()

            preds = self.data_transformer.inverse_target_transform(preds)

            return preds

@hydra.main(config_path="../configs", config_name="main")
def train_model(config: DictConfig):
    """Function to train the model"""

    logger.info(f"Training model for symbol: {config.data.symbol}")

    # Mflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = (
        f"market_price_predictor_model_{config.model.name}_symbol_{config.data.symbol}"
    )
    mlflow.set_experiment(experiment_name)
    # Start an mflow run, set the artifact path
    mlflow.start_run()

    # Log the configuration parameters to mlflow
    mlflow.log_params(config.model)

    data = load_stock_data_from_feature_store(
        symbol=config.data.symbol,
        feature_view_version=config.data.feature_view_version,
        train_dataset_version=config.data.train_dataset_version,
    )

    target_col = data.columns.get_loc(config.model.target_col)

    torch.set_default_dtype(torch.float64)

    # Set seeds for reproducibility
    torch.manual_seed(config.model.seed)
    np.random.seed(config.model.seed)

    data_transformer = DataFrameTransformer()
    data_transformer.set_scalers(StandardScaler(), StandardScaler())
    data_transformer.add_transformer(add_returns)
    data_transformer.add_transformer(filter_columns)

    # Get the train data, corresponding to the first train_val_split racio of the data
    train_data = data.iloc[: int(len(data) * config.model.train_val_split)]
    val_data = data.iloc[int(len(data) * config.model.train_val_split) :]

    train_data = data_transformer.fit_transform(train_data, target_col=config.model.target_col)
    val_data = data_transformer.transform_input(val_data)

    train_ds = StockDataset(
        stock_data=train_data,
        target_col=target_col,
        days_ahead=config.model.days_ahead,
        days_lag=config.model.days_lag,
    )
    val_ds = StockDataset(
        stock_data=val_data,
        target_col=target_col,
        days_ahead=config.model.days_ahead,
        days_lag=config.model.days_lag,
    )

    train_indices = get_valid_indices(train_ds)
    logger.info(f"Valid training indices: {len(train_indices)}")
    val_indices = get_valid_indices(val_ds)
    logger.info(f"Valid validation indices: {len(val_indices)}")

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=config.model.batch_size,
        sampler=SubsetRandomSampler(indices=train_indices),
        num_workers=config.model.train_num_workers,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=config.model.batch_size,
        sampler=SubsetRandomSampler(indices=val_indices),
        num_workers=config.model.val_num_workers,
    )

    logger.info("Data loaders created")

    pytorch_model = Transformer(
        embed_dim=config.model.embed_dim,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        max_seq_len=train_ds.max_seq_len,
        days_ahead=config.model.days_ahead,
    )

    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="min", monitor="train_loss", save_last=False
        ),
        EarlyStopping(monitor="val_loss", patience=config.model.patience, mode="min"),
    ]

    lightning_model = LightningModel(
        model=pytorch_model, learning_rate=config.model.learning_rate
    )

    logger.info("Model training starting...")

    mlflow_logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_id=mlflow.active_run().info.run_id,
    )

    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=config.model.max_epochs,
        accelerator=config.model.accelerator,
        deterministic=True,
        logger=mlflow_logger,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    full_model = FullModelPipeline(
        data_transformer=data_transformer,
        model=pytorch_model,
        days_ahead=config.model.days_ahead,
        days_lag=config.model.days_lag,
        target_col=target_col,
    )

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=full_model,
        code_paths=["data.py", "feature_engineering.py", "model.py", "train.py"],
    )

    # Make inference on the full dataset
    logger.info("Making inference on the full dataset")

    preds = full_model.predict(None, data)
    real = data.loc[:, config.model.target_col].values

    # We won't have predictions for the first days_lag days
    real = real[-len(preds):]

    generate_and_log_plot(real, preds, "predictions_vs_real")
    generate_and_log_plot(real[-30:], preds[-30:], "predictions_vs_real_recent")

    logger.info("Model training completed")


def generate_and_log_plot(real: np.array, preds: np.array, title: str):

    # New plot
    plt.figure(figsize=(10, 5))

    plt.plot(real.squeeze(), label="Real")
    plt.plot(preds.squeeze(), label="Predictions")
    plt.legend()

    with tempfile.TemporaryDirectory() as tmp_dir:
        plt.savefig(Path(tmp_dir, f"{title}.png"))
        mlflow.log_artifact(Path(tmp_dir, f"{title}.png"))


if __name__ == "__main__":

    train_model()
