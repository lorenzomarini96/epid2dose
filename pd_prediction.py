#!/usr/bin/env python3

"""
epid2dose

Deep learning-based Portal Dose prediction from transit EPID images.
"""

import argparse
import os

from utils.preprocessing import load_and_preprocess
from utils.prediction import pd_prediction
from utils.plotting import export_predictions
from utils.report import generate_pdf_report


# =============================================================================
# Command line interface
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict Portal Dose images from transit EPID images."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Directory containing EPID DICOM images.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--weights",
        default="models/weights",
        help="Directory containing the trained model weights.",
    )

    parser.add_argument(
        "--reference_dir",
        default=None,
        help="Directory containing reference Portal Dose (.txt) images.",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a gamma-index validation report.",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_arguments()

    print()
    print("------------------------------------------------------------")
    print("epid2dose")
    print("Deep learning-based Portal Dose prediction")
    print("------------------------------------------------------------")
    print()

    # ==========================================================
    # Create output folders
    # ==========================================================

    prediction_dir = os.path.join(args.output, "predictions")
    report_dir = os.path.join(args.output, "reports")

    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # ==========================================================
    # Load EPID images
    # ==========================================================

    print("Loading EPID images...")

    x_test, filenames = load_and_preprocess(args.input)

    print(f"Found {len(filenames)} EPID image(s).")

    # ==========================================================
    # Inference
    # ==========================================================

    print("\nRunning inference...")

    predictions = pd_prediction(
        x_test=x_test,
        weights_dir=args.weights,
    )

    # ==========================================================
    # Export predictions
    # ==========================================================

    print("\nExporting predictions...")

    export_predictions(
        x_test=x_test,
        final_predictions=predictions,
        x_filenames=filenames,
        prediction_dir=prediction_dir,
        pdf_path=os.path.join(
            report_dir,
            "prediction_overview.pdf",
        ),
    )

    # ==========================================================
    # Validation report
    # ==========================================================

    if args.report:

        if args.reference_dir is None:
            raise ValueError(
                "--report requires --reference_dir"
            )

        print("\nGenerating validation report...")

        generate_pdf_report(
            pdf_path=os.path.join(
                report_dir,
                "validation_report.pdf",
            ),
            reference_dir=args.reference_dir,
            prediction_dir=prediction_dir,
            epid_filenames=filenames,
            study_name="epid2dose",
        )

    print()
    print("Prediction completed successfully.")
    print()
    print(f"Predictions : {prediction_dir}")

    if args.report:
        print(f"Reports     : {report_dir}")

    print()


if __name__ == "__main__":
    main()