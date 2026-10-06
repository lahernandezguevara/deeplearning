from __future__ import annotations

import logging
import random
from itertools import product

import torch
import torch.nn as nn

from callbacks.early_stopping import EarlyStopping
from datasets.mnist import get_mnist_loaders
from engine.trainer import fit
from models.cnn import CNN
from utils.device import get_device

logger = logging.getLogger(__name__)

BATCH_SIZES = (64, 128)
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)
WEIGHT_DECAYS = (0.0, 1e-5, 1e-4, 1e-3)
NUM_TRIALS = 5
MAX_EPOCHS = 20
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-3
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 2

Hyperparameters = tuple[int, float, float]


def _sample_trials(num_trials: int, seed: int) -> list[Hyperparameters]:
    """Selecciona configuraciones sin repetición de forma reproducible."""
    search_space = list(product(BATCH_SIZES, LEARNING_RATES, WEIGHT_DECAYS))
    if not 1 <= num_trials <= len(search_space):
        raise ValueError(
            f"num_trials debe estar entre 1 y {len(search_space)}; "
            f"se recibió {num_trials}."
        )

    # Un generador local evita que la selección altere las semillas del entrenamiento.
    return random.Random(seed).sample(search_space, k=num_trials)


def _run_trial(configuration: Hyperparameters, seed: int) -> float:
    """Entrena una configuración y devuelve su mejor pérdida de validación."""
    batch_size, learning_rate, weight_decay = configuration

    # Reiniciar ambas fuentes permite comparar pruebas bajo las mismas condiciones.
    random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader, _ = get_mnist_loaders(
        "data",
        batch_size=batch_size,
    )
    device = get_device()
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )
    history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=MAX_EPOCHS,
        early_stopping=early_stopping,
        scheduler=scheduler,
    )
    return min(history["val_loss"])


def _run_search(num_trials: int, seed: int) -> None:
    """Ejecuta la búsqueda y registra la mejor configuración encontrada."""
    trials = _sample_trials(num_trials, seed)
    best_configuration = trials[0]
    best_score = float("inf")

    logger.info("Semilla de la búsqueda: %d", seed)
    for trial_number, configuration in enumerate(trials, start=1):
        batch_size, learning_rate, weight_decay = configuration
        logger.info(
            "Prueba %d/%d: batch_size=%d, learning_rate=%.1e, weight_decay=%.1e",
            trial_number,
            num_trials,
            batch_size,
            learning_rate,
            weight_decay,
        )
        score = _run_trial(configuration, seed)
        logger.info("Resultado de la prueba %d: val_loss=%.6f", trial_number, score)

        if score < best_score:
            best_score = score
            best_configuration = configuration

    batch_size, learning_rate, weight_decay = best_configuration
    logger.info(
        "Mejor configuración: batch_size=%d, learning_rate=%.1e, "
        "weight_decay=%.1e, val_loss=%.6f",
        batch_size,
        learning_rate,
        weight_decay,
        best_score,
    )


def main() -> None:
    """Configura el logging y ejecuta la búsqueda aleatoria mínima."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_search(NUM_TRIALS, RANDOM_SEED)


if __name__ == "__main__":
    main()
