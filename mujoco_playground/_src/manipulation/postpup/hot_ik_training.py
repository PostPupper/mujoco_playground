import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from mujoco_playground._src.manipulation.postpup import (
    model_utils,
    constants,
    data_generation,
)
import wandb


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000000):
        self.num_samples = num_samples

        self.model, self.data = model_utils.load_callback(
            xml_path=constants.PIPER_RENDERED_NORMAL_XML
        )

    def __len__(self):
        return self.num_samples

    def generate_data(self, arm_dofs: int = 6):
        # Generate random joint angles.
        joint_angles = np.random.rand(self.model.nq)
        # Compute forward kinematics for the "gripper_site_x_forward" site.
        pos_quat = data_generation.generate_fk(
            self.model, self.data, joint_angles, "gripper_site_x_forward"
        )
        return pos_quat, joint_angles[:arm_dofs]

    def __getitem__(self, idx):
        inp, lbl = self.generate_data()
        # Convert to torch tensors.
        inp = torch.tensor(inp, dtype=torch.float32)
        lbl = torch.tensor(lbl, dtype=torch.float32)
        return inp, lbl


class RegressionMLP(L.LightningModule):
    def __init__(
        self, hidden_sizes=[32, 32], batch_size=32, learning_rate=1e-3, t_max=50
    ):
        """
        Args:
            hidden_sizes (list of int): Number of neurons in each hidden layer.
            batch_size (int): Batch size for the dataloader.
            learning_rate (float): Learning rate for the optimizer.
            t_max (int): Maximum number of iterations for the CosineAnnealingLR scheduler.
        """
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Build a configurable MLP.
        layers = []
        input_size = 7  # Input vector size.
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ELU())
            input_size = hidden
        layers.append(
            nn.Linear(input_size, 6)
        )  # Output layer (6-element regression output)
        self.model = nn.Sequential(*layers)

        # Use Mean Squared Error for regression.
        self.criterion = nn.MSELoss()
        self.t_max = t_max

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Update scheduler every epoch.
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        dataset = GeneratedDataset(num_samples=1000000)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=12
        )


if __name__ == "__main__":
    config = {
        "hidden_sizes": [1024, 1024, 1024],
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "t_max": 500,
    }
    # Initialize wandb logger (log_model=True ensures that model checkpoints are uploaded to wandb cloud).
    wandb_logger = WandbLogger(
        project="regression_mlp_project",
        log_model=True,
    )
    wandb_logger.experiment.config.update(config)

    # Setup a model checkpoint callback that monitors the training loss.
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        filename="regression-mlp-{epoch:02d}-{train_loss:.4f}",
        save_top_k=3,
        verbose=True,
        dirpath="./checkpoints",  # Local directory for temporary checkpoint storage.
        save_weights_only=False,  # Save the full model.
    )

    # Log learning rate changes.
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Instantiate the model.
    model_instance = RegressionMLP(
        hidden_sizes=config["hidden_sizes"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        t_max=config["t_max"],
    )

    # Create the trainer with wandb logging and callbacks.
    trainer = L.Trainer(
        max_epochs=config["t_max"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # profiler="simple",
    )

    # Start training.
    trainer.fit(model_instance)
