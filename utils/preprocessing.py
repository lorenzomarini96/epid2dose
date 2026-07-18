import os
import numpy as np
import pydicom
import cv2


# =============================================================================
# Normalization constants
# =============================================================================

from utils.constants import (
    EPID_MIN,
    EPID_MAX,
    IMAGE_SIZE,
)

TARGET_SIZE = (256, 256)

def crop_epid_images_to_match_pd(epid_images, pd_shape=(345, 345),
                                  pixel_spacing_pd=1.0, pixel_spacing_epid=0.405):
    """
    Crop EPID images to match the physical field of view of the Portal Dose images.

    The original EPID images have a smaller pixel spacing than the reference
    Portal Dose images. This function crops the EPID images so that both
    modalities represent the same physical area before resizing them to the
    network input resolution.

    Parameters
    ----------
    epid_images : list[np.ndarray]
        List of raw EPID images.

    pd_shape : tuple[int, int], optional
        Original Portal Dose image size. Default is (345, 345).

    pixel_spacing_pd : float, optional
        Pixel spacing of the Portal Dose images in millimetres.
        Default is 1.0 mm.

    pixel_spacing_epid : float, optional
        Pixel spacing of the EPID images in millimetres.
        Default is 0.405 mm.

    Returns
    -------
    list[np.ndarray]
        List of cropped EPID images.
    """

    target_physical_x = pd_shape[1] * pixel_spacing_pd
    target_physical_y = pd_shape[0] * pixel_spacing_pd

    target_size_x = int(target_physical_x / pixel_spacing_epid)
    target_size_y = int(target_physical_y / pixel_spacing_epid)

    cropped_epid_images = []
    for img in epid_images:
        original_y, original_x = img.shape
        crop_x = max((original_x - target_size_x) // 2, 0)
        crop_y = max((original_y - target_size_y) // 2, 0)
        cropped_img = img[crop_y:original_y - crop_y, crop_x:original_x - crop_x]
        cropped_epid_images.append(cropped_img)

    return cropped_epid_images

def load_and_preprocess(directory_epid, apply_crop=True,
                        pd_shape=(345, 345), pixel_spacing_pd=1.0, pixel_spacing_epid=0.405):
    """
    Load and preprocess transit EPID images.

    The preprocessing pipeline consists of four consecutive steps:

    1. Read EPID DICOM images.
    2. Apply detector response correction using the PSF calibration value.
    3. Optionally crop the images to match the physical field of view of the
       Portal Dose images.
    4. Resize the images to 256 × 256 pixels and normalize the pixel values
       using the statistics adopted during network training.

    Parameters
    ----------
    directory_epid : str
        Directory containing the EPID DICOM images.

    apply_crop : bool, optional
        Whether to crop the EPID images before resizing.
        Default is True.

    pd_shape : tuple[int, int], optional
        Original Portal Dose image size.
        Default is (345, 345).

    pixel_spacing_pd : float, optional
        Pixel spacing of the Portal Dose images in millimetres.
        Default is 1.0 mm.

    pixel_spacing_epid : float, optional
        Pixel spacing of the EPID images in millimetres.
        Default is 0.405 mm.

    Returns
    -------
    np.ndarray
        Array of normalized EPID images with shape
        (N, 256, 256).

    list[str]
        List containing the original EPID filenames.

    Notes
    -----
    The detector response correction is performed using the PSF value stored
    in the DICOM private tag (0021,1002). Image normalization uses the same
    minimum and maximum values employed during model training to ensure
    consistency between training and inference.
    """

    epid_images_corrected = []
    epid_filenames = []

    dicom_files = sorted(
        f for f in os.listdir(directory_epid)
        if f.lower().endswith(".dcm")
    )

    for filename in dicom_files:
        path = os.path.join(directory_epid, filename)

        if os.path.isfile(path):
            ds = pydicom.dcmread(path)
            img = ds.pixel_array

            # 1. Detector response correction
            if (0x0021, 0x1002) not in ds:
                 raise ValueError(
                      f"Missing PSF tag in {filename}"
                )

            psf_value = ds[(0x0021, 0x1002)].value
            img_corrected = np.rint((65535 - img) / psf_value)

            epid_images_corrected.append(img_corrected)
            epid_filenames.append(filename)

    # 2. Crop to match the Portal Dose field of view
    if apply_crop:
        epid_images_corrected = crop_epid_images_to_match_pd(
            epid_images_corrected,
            pd_shape=pd_shape,
            pixel_spacing_pd=pixel_spacing_pd,
            pixel_spacing_epid=pixel_spacing_epid
        )

    # 3. Resize
    epid_images_processed = []

    for img in epid_images_corrected:
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        # 4. Normalize
        img_normalized = (img_resized - EPID_MIN) / (EPID_MAX - EPID_MIN)
        epid_images_processed.append(img_normalized)

    return np.array(epid_images_processed), epid_filenames
