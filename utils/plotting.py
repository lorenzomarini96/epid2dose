import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.constants import (
    EPID_MIN,
    EPID_MAX,
    PD_MIN,
    PD_MAX,
    PIXEL_SPACING,
)


# =============================================================================
# Utility functions
# =============================================================================

def denormalize_epid(image):
    """Convert normalized EPID image back to detector units."""
    return image * (EPID_MAX - EPID_MIN) + EPID_MIN


def denormalize_pd(image):
    """Convert normalized Portal Dose prediction back to cGy."""
    return image * (PD_MAX - PD_MIN) + PD_MIN


def get_extent(pixel_spacing_mm=PIXEL_SPACING):
    """Return image extent in cm centered at the origin."""

    pixel_spacing_cm = pixel_spacing_mm / 10.0

    return [
        -128 * pixel_spacing_cm,
         128 * pixel_spacing_cm,
        -128 * pixel_spacing_cm,
         128 * pixel_spacing_cm,
    ]


# =============================================================================
# Export predictions
# =============================================================================

def export_predictions(
    x_test,
    final_predictions,
    x_filenames,
    prediction_dir,
    pdf_path,
    save_denormalized=True,
):
    """
    Generate a PDF overview of the predictions and export the predicted
    Portal Dose images as NumPy arrays.

    Parameters
    ----------
    x_test : np.ndarray
        Normalized EPID images.

    final_predictions : np.ndarray
        Normalized Portal Dose predictions.

    x_filenames : list[str]
        Original EPID filenames.

    prediction_dir : str
        Directory where the predicted Portal Dose (.npy) files are saved.

    pdf_path : str
        Full path of the PDF overview.

    save_denormalized : bool
        If True, predictions are saved in physical units (cGy).
    """

    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    extent = get_extent()

    with PdfPages(pdf_path) as pdf:

        for epid_img, pd_pred, filename in zip(
            x_test,
            final_predictions,
            x_filenames,
        ):

            epid_img_denorm = denormalize_epid(epid_img)
            pd_pred_denorm = denormalize_pd(pd_pred)

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # ==========================================================
            # EPID
            # ==========================================================

            im0 = axs[0].imshow(
                epid_img_denorm,
                cmap="jet",
                extent=extent,
            )

            axs[0].set_title(f"EPID: {filename}")
            axs[0].set_xlabel("X [cm]")
            axs[0].set_ylabel("Y [cm]")
            axs[0].axhline(0, color="white", ls="--", lw=0.5)
            axs[0].axvline(0, color="white", ls="--", lw=0.5)

            cbar0 = fig.colorbar(
                im0,
                ax=axs[0],
                fraction=0.046,
                pad=0.04,
            )

            cbar0.set_label("[a.u.]")

            # ==========================================================
            # Predicted Portal Dose
            # ==========================================================

            im1 = axs[1].imshow(
                pd_pred_denorm,
                cmap="jet",
                extent=extent,
                vmin=PD_MIN,
                vmax=PD_MAX,
            )

            axs[1].set_title("Predicted Portal Dose")
            axs[1].set_xlabel("X [cm]")
            axs[1].set_ylabel("Y [cm]")
            axs[1].axhline(0, color="white", ls="--", lw=0.5)
            axs[1].axvline(0, color="white", ls="--", lw=0.5)

            cbar1 = fig.colorbar(
                im1,
                ax=axs[1],
                fraction=0.046,
                pad=0.04,
            )

            cbar1.set_label("[cGy]")

            plt.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

            # ==========================================================
            # Save prediction
            # ==========================================================

            base_name = os.path.splitext(filename)[0]
            base_name = (
                base_name.replace("EPID", "")
                .replace("epid", "")
                .strip("_-")
            )

            prediction_path = os.path.join(
                prediction_dir,
                f"PD_{base_name}.npy",
            )

            if save_denormalized:
                np.save(prediction_path, pd_pred_denorm)
            else:
                np.save(prediction_path, pd_pred)

    print(f"Prediction overview saved to: {pdf_path}")