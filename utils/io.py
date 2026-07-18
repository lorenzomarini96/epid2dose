import os
import numpy as np
import cv2

from utils.constants import IMAGE_SIZE

def load_reference_pd(reference_dir, target_size=IMAGE_SIZE):
    """
    Load reference Portal Dose (.txt) images and resize them.

    Args:
        reference_dir (str): Directory containing the reference Portal Dose images.
        target_size (tuple): Output image size.

    Returns:
        images (list[np.ndarray]): Reference Portal Dose images.
        filenames (list[str]): Corresponding filenames.
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
    Load predicted Portal Dose images (.npy).
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
