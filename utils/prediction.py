import os

import numpy as np
import pymedphys
import tensorflow as tf

from tqdm import tqdm

from models.unet import UNet

from utils.constants import (
    PIXEL_SPACING,
    DEFAULT_FILTERS,
    DEFAULT_FOLDS,
    DEFAULT_GAMMA_DOSE,
    DEFAULT_GAMMA_DISTANCE,
    DEFAULT_GAMMA_CUTOFF,
)


# =============================================================================
# Portal Dose prediction
# =============================================================================

def pd_prediction(
    x_test,
    weights_dir,
    best_filters=DEFAULT_FILTERS,
    n_folds=DEFAULT_FOLDS,
):
    """
    Predict Portal Dose images from preprocessed EPID images using the
    trained ensemble.

    Parameters
    ----------
    x_test : np.ndarray
        Preprocessed EPID images.

    weights_dir : str
        Directory containing the trained model weights.

    best_filters : int
        Number of filters of the U-Net architecture.

    n_folds : int
        Number of cross-validation folds.

    Returns
    -------
    np.ndarray
        Predicted Portal Dose images.
    """

    ensemble_predictions = []

    for fold in tqdm(
        range(1, n_folds + 1),
        desc="Loading ensemble models",
        ncols=80,
        ascii=" >=",
    ):

        fold_dir = os.path.join(
            weights_dir,
            f"cv_{fold}",
        )

        if not os.path.isdir(fold_dir):
            raise FileNotFoundError(
                f"Fold directory not found:\n{fold_dir}"
            )

        fold_predictions = []

        # -------------------------------------------------------------
        # Models belonging to the current fold
        # -------------------------------------------------------------

        for model_idx in range(1, 6):

            model_dir = os.path.join(
                fold_dir,
                f"model_{model_idx}",
            )

            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"Model directory not found:\n{model_dir}"
                )

            weight_files = sorted(
                f
                for f in os.listdir(model_dir)
                if f.endswith(".h5")
            )

            if len(weight_files) == 0:
                raise FileNotFoundError(
                    f"No weight file found in:\n{model_dir}"
                )

            # ---------------------------------------------------------
            # Select checkpoint with the lowest validation MAE
            # ---------------------------------------------------------

            def extract_val_mae(filename):
                return float(
                    filename.split("-")[-1].replace(
                        ".weights.h5",
                        "",
                    )
                )

            best_weight = min(
                weight_files,
                key=extract_val_mae,
            )

            best_weight_path = os.path.join(
                model_dir,
                best_weight,
            )

            # ---------------------------------------------------------
            # Load model
            # ---------------------------------------------------------

            model = UNet(
                input_size=(256, 256, 1),
                num_filters=best_filters,
            )

            model.load_weights(best_weight_path)

            predictions = model.predict(
                x_test,
                verbose=0,
            )

            fold_predictions.append(predictions)

        # -------------------------------------------------------------
        # Average predictions within the fold
        # -------------------------------------------------------------

        fold_predictions = np.asarray(fold_predictions)

        fold_average = np.mean(
            fold_predictions,
            axis=0,
        )

        ensemble_predictions.append(fold_average)

    # -----------------------------------------------------------------
    # Average predictions across folds
    # -----------------------------------------------------------------

    ensemble_predictions = np.asarray(
        ensemble_predictions
    )

    final_predictions = np.mean(
        ensemble_predictions,
        axis=0,
    )

    final_predictions = tf.squeeze(
        final_predictions,
        axis=-1,
    )

    print("Inference completed.")

    return final_predictions


# =============================================================================
# Gamma-index analysis
# =============================================================================

def calculate_gamma_index(
    reference,
    prediction,
    dose_percent_threshold=DEFAULT_GAMMA_DOSE,
    distance_mm_threshold=DEFAULT_GAMMA_DISTANCE,
):
    """
    Compute the gamma-index map between a reference and a predicted
    Portal Dose image.
    """

    reference = np.squeeze(reference)
    prediction = np.squeeze(prediction)

    axes = (
        PIXEL_SPACING * np.arange(reference.shape[0]),
        PIXEL_SPACING * np.arange(reference.shape[1]),
    )

    gamma = pymedphys.gamma(
        axes,
        reference,
        axes,
        prediction,
        dose_percent_threshold,
        distance_mm_threshold,
        lower_percent_dose_cutoff=DEFAULT_GAMMA_CUTOFF,
    )

    return gamma