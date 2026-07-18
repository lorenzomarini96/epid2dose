import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from utils.io import load_reference_pd, load_predictions
from utils.prediction import calculate_gamma_index
from utils.plotting import get_extent


def generate_pdf_report(
    pdf_path,
    reference_dir,
    prediction_dir,
    epid_filenames,
    study_name="epid2dose",
    pd_min_val=0.0,
    pd_max_val=55.0,
):
    """
    Generate a PDF validation report based on gamma-index analysis.

    The report compares the predicted Portal Dose distributions against the
    corresponding reference Portal Dose images and summarizes the agreement
    using the gamma-index (3%/3 mm criterion).

    For each evaluated case, the report includes:

    - the reference Portal Dose distribution;
    - the predicted Portal Dose distribution;
    - the gamma-index histogram;
    - the gamma passing rate.

    A final summary page reports the mean, median, minimum, and maximum
    gamma passing rates across all evaluated cases.

    Parameters
    ----------
    pdf_path : str
        Output path of the generated PDF report.

    reference_dir : str
        Directory containing the reference Portal Dose images exported
        from the Treatment Planning System (.txt).

    prediction_dir : str
        Directory containing the predicted Portal Dose images (.npy).

    epid_filenames : list[str]
        List of EPID filenames corresponding to each evaluated case.

    study_name : str, optional
        Study identifier reported in the PDF header.
        Default is ``"epid2dose"``.

    pd_min_val : float, optional
        Minimum Portal Dose value displayed in the dose maps.
        Default is ``0.0`` cGy.

    pd_max_val : float, optional
        Maximum Portal Dose value displayed in the dose maps.
        Default is ``55.0`` cGy.

    Returns
    -------
    None

    Notes
    -----
    The gamma-index is computed using the implementation provided by
    PyMedPhys with the default criteria adopted by the package.
    """

    reference_images, _ = load_reference_pd(reference_dir)
    predicted_images, _ = load_predictions(prediction_dir)

    if len(reference_images) != len(predicted_images):
        raise RuntimeError(
            "The number of reference and predicted Portal Dose images does not match."
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ==========================================================
    # Header
    # ==========================================================

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "epid2dose", ln=True, align="C")

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Validation Report", ln=True)

    pdf.set_font("Arial", size=11)

    pdf.cell(
        0,
        8,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True,
    )

    pdf.cell(0, 8, f"Study: {study_name}", ln=True)

    pdf.cell(
        0,
        8,
        f"Evaluated images: {len(reference_images)}",
        ln=True,
    )

    pdf.cell(
        0,
        8,
        "Gamma criterion: 3% / 3 mm",
        ln=True,
    )

    pdf.ln(5)

    gamma_summary = []

    # ==========================================================
    # One page per case
    # ==========================================================

    for i, (reference, prediction) in enumerate(
        zip(reference_images, predicted_images)
    ):

        gamma = calculate_gamma_index(reference, prediction)

        valid_gamma = gamma[~np.isnan(gamma)]

        gamma_pass_rate = (
            np.sum(valid_gamma <= 1)
            / len(valid_gamma)
            * 100
        )

        gamma_summary.append(
            (epid_filenames[i], gamma_pass_rate)
        )

        fig, axs = plt.subplots(
            1,
            3,
            figsize=(15, 5),
            gridspec_kw={"width_ratios": [1, 1, 1]},
            constrained_layout=True,
        )

        # ------------------------------------------------------
        # Reference Portal Dose
        # ------------------------------------------------------

        im0 = axs[0].imshow(
            reference,
            cmap="jet",
            vmin=pd_min_val,
            vmax=pd_max_val,
            extent=get_extent(),
        )

        axs[0].set_title("Reference Portal Dose")
        axs[0].set_xlabel("X [cm]")
        axs[0].set_ylabel("Y [cm]")

        # ------------------------------------------------------
        # Predicted Portal Dose
        # ------------------------------------------------------

        im1 = axs[1].imshow(
            prediction,
            cmap="jet",
            vmin=pd_min_val,
            vmax=pd_max_val,
            extent=get_extent(),
        )

        axs[1].set_title("Predicted Portal Dose")
        axs[1].set_xlabel("X [cm]")
        axs[1].set_ylabel("Y [cm]")

        # ------------------------------------------------------
        # Gamma histogram
        # ------------------------------------------------------

        axs[2].hist(
            valid_gamma,
            bins=30,
            color="cyan",
            edgecolor="black",
        )

        axs[2].axvline(
            x=1,
            color="red",
            linestyle="--",
        )

        axs[2].set_title(
            f"Gamma Histogram\nPassing rate = {gamma_pass_rate:.2f}%"
        )

        axs[2].set_xlabel("Gamma index")
        axs[2].set_ylabel("Frequency")

        cbar = fig.colorbar(
            im1,
            ax=[axs[0], axs[1]],
            fraction=0.046,
            pad=0.04,
        )

        cbar.set_label("Dose [cGy]")

        tmp_img = f"temp_gamma_{i}.png"

        plt.savefig(tmp_img, dpi=200)
        plt.close(fig)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(
            0,
            8,
            f"Case {i+1}: {epid_filenames[i]}",
            ln=True,
        )

        pdf.image(tmp_img, w=180)

        pdf.ln(5)

        os.remove(tmp_img)

    # ==========================================================
    # Summary
    # ==========================================================

    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.cell(
        0,
        10,
        "Gamma Passing Rate Summary",
        ln=True,
    )

    pdf.set_font("Arial", size=11)

    pdf.cell(120, 8, "Case", border=1)
    pdf.cell(50, 8, "Passing rate (%)", border=1, ln=True)

    rates = []

    for filename, rate in gamma_summary:

        pdf.cell(120, 8, filename, border=1)
        pdf.cell(50, 8, f"{rate:.2f}", border=1, ln=True)

        rates.append(rate)

    pdf.set_font("Arial", "B", 11)

    pdf.cell(120, 8, "Mean", border=1)
    pdf.cell(50, 8, f"{np.mean(rates):.2f}", border=1, ln=True)

    pdf.cell(120, 8, "Median", border=1)
    pdf.cell(50, 8, f"{np.median(rates):.2f}", border=1, ln=True)

    pdf.cell(120, 8, "Minimum", border=1)
    pdf.cell(50, 8, f"{np.min(rates):.2f}", border=1, ln=True)

    pdf.cell(120, 8, "Maximum", border=1)
    pdf.cell(50, 8, f"{np.max(rates):.2f}", border=1, ln=True)

    pdf.output(pdf_path)

    print(f"Validation report saved to: {pdf_path}")