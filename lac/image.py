import numpy as np
import skimage.io
import pandas as pd 

def compute_mean_bg(phase_image, fluo_image, method='isodata', obj_dark=True):
    """
    Computes the mean background fluorescence of the inverted segmentation
    mask.

    Parameters
    ----------
    phase_image : 2d-array, int or float.
        The phase contrast image used for generating the inverse segmentation
        mask. If this image is not a float with pixel values in (0, 1), it
        will be renormalized.
    fluo_image : 2d-array, int
        The fluorescence image used to calculate the mean pixel value. If
        flatfield correction is necessary, it should be done before this
        sending to this function.
    method: string, ['otsu', 'yen', 'li', 'isodata'], default 'isodata'
        Automated thresholding method to use. Default is 'isodata' method.
    obj_dark : bool, default True
        If True, objects will be **darker** than the automatically generated
        threshold value. If False, objects are deemed to be brighter.

    Returns
    -------
    mean_bg: float
        The mean background fluorescence of the image.
    """

    # Ensure that the image is renormalized.
    if (phase_image > 1.0).any():
        phase_image = (phase_image - phase_image.min()) /\
                      (phase_image.max() - phase_image.min())
    # Perform the background subtraction.
    im_blur = skimage.filters.gaussian(phase_image, sigma=50)
    im_sub = phase_image - im_blur

    # Determine the method to use.
    methods = {'otsu': skimage.filters.threshold_otsu,
               'yen': skimage.filters.threshold_yen,
               'li': skimage.filters.threshold_li,
               'isodata': skimage.filters.threshold_isodata}

    # Determine the threshold value.
    thresh_val = methods[method](im_sub)

    # Generate the inverted segmentation mask and dilate.
    if obj_dark is True:
        im_thresh = im_sub < thresh_val
    else:
        im_thresh = im_sub > thresh_val

    selem = skimage.morphology.disk(20)
    im_dil = skimage.morphology.dilation(im_thresh, selem=selem)

    # Mask onto the fluroescence image and compute the mean background value.
    mean_bg = np.mean(fluo_image[im_dil < 1])
    return mean_bg


def median_flatfield(image_stack, medfilter=True, selem='default',
                     return_profile=False):
    """
    Computes a illumination profile from the median of all images
    and corrects each individual image.

    Parameters
    ----------
    image_stack: scikit-image ImageCollection
        Series of images to correct. The illumination profile is created
        from computing the median filter of all images in this collection.
    medfilter: bool, default True
        If True, each individiual image will be prefiltered using a median
        filter with  a given selem.
    selem : string or structure, default 3x3 square
        Structural element to use for the median filtering. Default  is
        a 3x3 pixel square.
    return_profile: bool, default False
        If True, the illumination profiled image will be returned.

    Returns
    -------
    ff_ims : list of 2d-array
        Flatfield corrected images.
    med_im : 2d-array
        Illumination profile produced from the median of all images in
        image stack.
    """

    # Determine if the prefiltering should be performed.
    if medfilter is True:

        # Define the structural element.
        if selem is 'default':
            selem = skimage.morphology.square(3)
        image_stack = [scipy.ndimage.median_filter(
            im, footprint=selem) for im in image_stack]

    # Compute the median filtered image.
    med_im = np.median(image_stack, axis=0)

    # Perform the correction.
    ff_ims = [(i / med_im) * np.mean(med_im) for i in image_stack]

    if return_profile is True:
        return [ff_ims, med_im]
    else:
        return ff_ims


def average_stack(im, median_filt=True):
    """
    Computes an average image from a provided array of images.

    Parameters
    ----------
    im : list or arrays of 2d-arrays
        Stack of images to be filtered.
    median_filt : bool
        If True, each image will be median filtered before averaging.
        Median filtering is performed using a 3x3 square structural element.

    Returns
    -------
    im_avg : 2d-array
        averaged image with a type of int.
    """

    # Determine if the images should be median filtered.
    if median_filt is True:
        selem = skimage.morphology.square(3)
        im_filt = [scipy.ndimage.median_filter(i, footprint=selem) for i in im]
    else:
        im = im_filt

    # Generate and empty image to store the averaged image.
    im_avg = np.zeros_like(im[0]).astype(int)
    for i in im:
        im_avg += i
    im_avg = im_avg / len(im)
    return im_avg


def generate_flatfield(im, im_field, median_filt=True):
    """
    Corrects illumination of a given image using a dark image and an image of
    the flat illumination.

    Parameters
    ----------
    im : 2d-array
        Image to be flattened.
    im_field: 2d-array
        Average image of fluorescence illumination.
    median_filt : bool
        If True, the image to be corrected will be median filtered with a
        3x3 square structural element.

    Returns
    -------
    im_flat : 2d-array
        Image corrected for uneven fluorescence illumination. This is performed
        as

        im_flat = (im  / im_field ) * mean(im_field)

    Raises
    ------
    RuntimeError
        Thrown if bright image and dark image are approximately equal. This
        will result in a division by zero.
    """
    # Compute the mean field value.
    mean_diff = np.mean(im_field)

    if median_filt is True:
        selem = skimage.morphology.square(3)
        im_filt = scipy.ndimage.median_filter(im, footprint=selem)
    else:
        im_filt = im

    # Compute and return the flattened image.
    im_flat = (im_filt / im_field) * mean_diff
    return im_flat


def normalize_image(im, sub_bg=True):
    """
    Rescales the values of an image between 0 and 1. Can also perform a
    background subtraction.

    Parameters
    ----------
    im : 2d-array
        Image to be normalized.
    sub_bg: bool, default True.
        If True, a gaussian background subtraction is performed with
        a small sd.
    Returns
    -------
    im_norm : 2d-array
        Normalized image. If sub_bg is True, these values are on
        the domain [-1, 1]. If sub_bg is False, values are on [0, 1]
    """
    im_norm = (im - im.min()) / (im.max() - im.min())
    if sub_bg is True:
        im_blur = skimage.filters.gaussian(im_norm, sigma=5)
        im_norm = im_norm - im_blur
    return im_norm


def threshold_phase(im, min_int=0.15):
    """
    Performs an intensity based segmentation of a phase contrast image.
    This function uses Otsu's method to determine the threshold value.

    Parameters
    ----------
    im: 2d-array
        Image to be segmented. Desired objects in this image are assumed
        to be dark.
    min_int : float
        The maximum mean pixel intensity of a segmented object. This
        value must be between 0 and 1. Default is 0.15

    Returns
    -------
    mask: 2d-array, int
        Segmented image with labeled regions.
    """

    # Preprocess the phase image.
    im_sub = normalize_image(im)
    im_float = normalize_image(im, sub_bg=False)

    # Use Otsu's method.
    thresh = skimage.filters.threshold_otsu(im_sub)

    # Clean initial segmentation.
    seg = skimage.segmentation.clear_border(im_sub < thresh)
    seg = skimage.morphology.remove_small_objects(seg)
    mask = skimage.measure.label(seg)

    # Oversegment to correct for slight drift.
    selem = skimage.morphology.disk(2)
    mask = skimage.morphology.dilation(mask, selem)
    lab = skimage.measure.label(mask)

    # Impose minimum intensity filter.
    props = skimage.measure.regionprops(lab, im_float)
    final_im = np.zeros_like(mask)
    for prop in props:
        mean_int = prop.min_intensity
        if mean_int <= min_int:
            final_im += (lab == prop.label)
    mask = skimage.measure.label(final_im)
    return mask