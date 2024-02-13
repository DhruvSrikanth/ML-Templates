import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import callbacks


class SomeDataset(data.Dataset):
    """
    A custom dataset class for any data you want to use.
    """
    def __init__(self, root_dir: str) -> None:
        super(SomeDataset, self).__init__()
        """
        Args:
            root_dir (str): Directory with all the data.
        """
        self.root_dir = root_dir
        # Process data ...
        inputs, outputs = [], []
        # Make sure these are torch tensors
        self.inputs = inputs.float()
        self.outputs = outputs.float()

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (input, output) pair.
        """
        return self.inputs[idx], self.outputs[idx]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.inputs)


class SequencePredictorExample(nn.Module):
    """
    A simple sequence predictor model.

    Args:
        latent_dim (int): The dimension of the latent space.
        num_layers (int): The number of layers in the transformer encoder.
        num_heads (int): The number of heads in the multihead attention mechanism.

    Returns:
        torch.Tensor: The predicted sequence.
    """
    def __init__(self, latent_dim: int, num_layers: int, num_heads: int) -> None:
        super(SequencePredictorExample, self).__init__()
        """
        Args:
            latent_dim (int): The dimension of the latent space.
            num_layers (int): The number of layers in the transformer encoder.
            num_heads (int): The number of heads in the multihead attention mechanism.
        """
        self.d_model = latent_dim
        self.projection = nn.Linear(in_features=3 * 240 * 240, out_features=latent_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.feature_extractor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, batch_first=True, dropout=0.0),
            num_layers=num_layers
        )
        self.predictor = nn.Linear(latent_dim, 1)
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize the weights of the model.
        """
        initrange = 0.1
        self.projection.weight.data.uniform_(-initrange, initrange)
        self.predictor.bias.data.zero_()
        self.predictor.weight.data.uniform_(-initrange, initrange)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            seq_x (torch.Tensor): The input sequence.

        Returns:
            torch.Tensor: The predicted sequence.
        """
        seq_x = seq_x.flatten(start_dim=2)
        sequence_embedding = self.relu(self.projection(seq_x) * math.sqrt(self.d_model))
        sequence_features = self.feature_extractor(sequence_embedding)
        sequence_logits = self.predictor(sequence_features)
        return sequence_logits


class SequenceLearner(L.LightningModule):
    """
    A simple sequence learner framework.

    Args:
        predictor (torch.nn.Module): The sequence predictor model.
        lr (float): The learning rate.
        betas (tuple): The beta values for the Adam optimizer.
        eps (float): The epsilon value for the Adam optimizer.
        weight_decay (float): The weight decay value for the Adam optimizer.
        log_info (bool): Whether to log information about the model.

    Returns:
        dict: The loss of the model.
    """
    def __init__(self, predictor: nn.Module, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, log_info: bool = False) -> None:
        super(SequenceLearner, self).__init__()
        self.predictor = predictor
        self.learning_rate = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.log_info = log_info
        self.criterion = nn.MSELoss(reduction='mean')

    def step(self, batch: torch.Tensor, stage: str) -> dict:
        """
        A single step of the model.

        Args:
            batch (torch.Tensor): The input batch.
            stage (str): The stage of the model.

        Returns:
            dict: The loss of the model.
        """
        assert stage in ['train', 'val', 'test']
        b_x, b_y = batch
        logits = self.predictor(seq_x=b_x)
        loss = self.criterion(logits, b_y)
        self.log_dict({f'{stage}_loss': loss}, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def training_step(self, batch: torch.Tensor) -> dict:
        """
        A single training step of the model.

        Args:
            batch (torch.Tensor): The input batch.

        Returns:
            dict: The loss of the model.
        """
        return self.step(batch=batch, stage='train')

    def validation_step(self, batch: torch.Tensor) -> dict:
        """
        A single validation step of the model.

        Args:
            batch (torch.Tensor): The input batch.

        Returns:
            dict: The loss of the model.
        """
        return self.step(batch=batch, stage='val')

    def test_step(self, batch: torch.Tensor) -> dict:
        """
        A single testing step of the model.

        Args:
            batch (torch.Tensor): The input batch.

        Returns:
            dict: The loss of the model.
        """
        return self.step(batch=batch, stage='test')

    def configure_optimizers(self) -> dict:
        """
        Configure the optimizer for the model.

        Returns:
            dict: The optimizer and learning rate scheduler of the model.
        """
        opt = optim.AdamW(self.parameters(), lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)
        return {'optimizer': opt, 'lr_scheduler': sch, 'monitor': 'val_loss'}


def benchmark(trainer: L.Trainer, model: L.LightningModule, train_dataloaders: data.Dataloader, val_dataloaders: data.Dataloader, test_dataloaders: data.Dataloader, log_info: bool, log_freq: int) -> None:
    """
    Benchmark the model.

    Args:
        trainer (L.Trainer): The trainer for the model.
        model (nn.LightningModule): The model to benchmark.
        train_dataloaders (data.Dataloader): The training dataloader.
        val_dataloaders (data.Dataloader): The validation dataloader.
        test_dataloaders (data.Dataloader): The testing dataloader.
        log_info (bool): Whether to log information about the model.
        log_freq (int): The frequency at which to log information about the model.
    """
    if log_info: trainer.logger.watch(model, log='all', log_freq=log_freq)
    trainer.fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
    trainer.test(model=model, dataloaders=test_dataloaders)
    if log_info: trainer.logger.experiment.unwatch(model)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RANDOM_SEED = 100
    L.seed_everything(RANDOM_SEED)

    ROOT_DIR = 'path/to/data'
    BATCH_SIZE = 256
    SHUFFLE = True
    NUM_WORKERS = 32
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    dataloader = data.DataLoader(
        dataset=SomeDataset(root_dir=ROOT_DIR),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    LOG_INFO = True
    LOG_FREQ = 100
    LATENT_DIM = 512
    NUM_LAYERS = 4
    NUM_HEADS = 4
    LR = 1e-3
    model = SequenceLearner(
        predictor=SequencePredictorExample(latent_dim=LATENT_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS).float(),
        lr=LR,
        log_info=LOG_INFO
    )

    logger = WandbLogger(project="project-name", log_model='all')
    functional_callbacks = [
        callbacks.ModelCheckpoint(monitor='val_loss', mode='min'),            # Save best model (based on val_loss (lower is better)))
        callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min'),  # Stop training if val_loss doesn't improve for 5 epochs
    ]
    cosmetic_callbacks = [
        callbacks.RichProgressBar(),   # Progress bar
        callbacks.RichModelSummary(),  # Model summary
        callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True),  # Monitor optimziation params
    ]
    MAX_EPOCHS = 50
    GRADIENT_CLIP_VAL = 1
    N_GPUS = 1
    DEVICES = [i for i in range(N_GPUS)]
    acceleration_config = {'accelerator': 'gpu', 'devices': DEVICES}
    trainer = L.Trainer(gradient_clip_val=GRADIENT_CLIP_VAL, max_epochs=MAX_EPOCHS, logger=logger, callbacks=functional_callbacks + cosmetic_callbacks, **acceleration_config) \
              if LOG_INFO else \
              L.Trainer(max_epochs=MAX_EPOCHS, callbacks=cosmetic_callbacks, **acceleration_config)

    benchmark(trainer=trainer, model=model, train_dataloaders=dataloader, val_dataloaders=None, test_dataloaders=None, log_info=LOG_INFO, log_freq=LOG_FREQ)
