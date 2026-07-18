import os
import numpy as np
import cv2

from utils.constants import IMAGE_SIZE

def load_reference_pd(reference_dir, target_size=IMAGE_SIZE):
    """
    Load reference Portal Dose images.

    The reference Portal Dose distributions are expected as plain-text files
    exported from the Treatment Planning System (TPS). Each image is resized
    to the network input resolution.

    Parameters
    ----------
    reference_dir : str
        Directory containing the reference Portal Dose (.txt) files.

    target_size : tuple[int, int], optional
        Output image size in pixels.
        Default is (256, 256).

    Returns
    -------
    list[np.ndarray]
        List of resized Portal Dose images.

    list[str]
        Corresponding filenames without extension.

    Raises
    ------
    RuntimeError
        If one or more files cannot be loaded.
    """

    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(
            f"Reference directory not found: {reference_dir}"
        )

    images = []
    filenames = []

    for fname in sorted(os.listdir(reference_dir)):
        if fname.endswith(".txt"):

            path = os.path.join(reference_dir, fname)

            try:
                img = np.loadtxt(path)
                img = cv2.resize(
                    img,
                    target_size,
                    interpolation=cv2.INTER_CUBIC
                )

                images.append(img)
                filenames.append(os.path.splitext(fname)[0])

            except Exception as e:
                raise RuntimeError(
                    f"Unable to load reference file '{fname}': {e}"
                )

    return images, filenames


def load_predictions(prediction_dir):
    """
    Load predicted Portal Dose distributions.

    Parameters
    ----------
    prediction_dir : str
        Directory containing the predicted Portal Dose images saved as
        NumPy (.npy) files.

    Returns
    -------
    list[np.ndarray]
        List of predicted Portal Dose images.

    list[str]
        Corresponding filenames without extension.

    Raises
    ------
    FileNotFoundError
        If the prediction directory does not exist.
    """

    if not os.path.isdir(prediction_dir):
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_dir}"
        )

    pred_filenames = sorted(
        f for f in os.listdir(prediction_dir)
        if f.endswith(".npy")
    )

    pred_images = [
        np.load(
            os.path.join(prediction_dir, fname),
            allow_pickle=False
        )
        for fname in pred_filenames
    ]

    return pred_images, [
        os.path.splitext(f)[0]
        for f in pred_filenames
    ]
