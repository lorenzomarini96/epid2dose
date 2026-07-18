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
    Ritaglia tutte le immagini EPID per farle corrispondere alla dimensione fisica delle immagini PD.

    Args:
        epid_images (list or np.array): Lista o array di immagini EPID (shape: HxW).
        pd_shape (tuple): Shape originale delle immagini PD (default: (345, 345)).
        pixel_spacing_pd (float): Spaziatura pixel delle immagini PD.
        pixel_spacing_epid (float): Spaziatura pixel delle immagini EPID.

    Returns:
        list: Lista di immagini EPID ritagliate.
    
    COMMENTO:
    Le EPID passano da 1024x1024 a 852x852 con stesso pixel spacing ([0.405, 0.405]mm).
    Dopo il resize, 852x852 --> 256x256.
    le EPID passano da 852x852 --> 256x256.
    Se il pixel spacing era 0.405 mm/pixel, dopo il resize sarà 852/256 * 0.405 mm = 1,347 mm/pixel
    le PD passano da 345x345 --> 256x256.
    Se il pixel spacing era 1.00 mm/pixel, dopo il resize sarà 345/256 * 1mm = 1,34 mm/pixel
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
    Carica le immagini EPID da file DICOM, (1) corregge, (2) opzionalmente croppa,
    (3) ridimensiona, (4) normalizza.

    Args:
        directory_epid (str): Percorso alla cartella contenente le immagini DICOM EPID.
        apply_crop (bool): Se True, applica il crop per matchare il pixel spacing con le PD.
        pd_shape (tuple): Shape originale delle immagini PD (default: (345, 345)).
        pixel_spacing_pd (float): Spaziatura pixel immagini PD.
        pixel_spacing_epid (float): Spaziatura pixel immagini EPID.

    Returns:
        tuple: (epid_images_processed, epid_filenames)
            - epid_images_processed (np.array): Array delle immagini EPID preprocessate.
            - epid_filenames (list): Lista dei nomi dei file EPID caricati.
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

            # 1) Correzione PSF
            if (0x0021, 0x1002) not in ds:
                 raise ValueError(
                      f"Missing PSF tag in {filename}"
                )

            psf_value = ds[(0x0021, 0x1002)].value
            img_corrected = np.rint((65535 - img) / psf_value)

            epid_images_corrected.append(img_corrected)
            epid_filenames.append(filename)

    # 2) Crop opzionale
    if apply_crop:
        epid_images_corrected = crop_epid_images_to_match_pd(
            epid_images_corrected,
            pd_shape=pd_shape,
            pixel_spacing_pd=pixel_spacing_pd,
            pixel_spacing_epid=pixel_spacing_epid
        )

    # 3) Resize + 4) Normalizzazione
    epid_images_processed = []

    for img in epid_images_corrected:
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img_normalized = (img_resized - EPID_MIN) / (EPID_MAX - EPID_MIN)
        epid_images_processed.append(img_normalized)

    return np.array(epid_images_processed), epid_filenames
