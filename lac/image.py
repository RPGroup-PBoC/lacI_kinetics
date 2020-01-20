import numpy as np
import skimage.io
import skimage.segmentation
import skimage.morphology
import skimage.measure
import skimage.filters
import scipy.ndimage
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

def contour_seg(image, level=0.3, selem='default', perim_bounds=(5, 1E3),
                ip_dist=0.160, ecc_bounds=(0.7, 1), area_bounds=(1, 50),
                return_conts=False, min_int=0.2):
    """
    Identifies contours around dark objects in a phase contrast image.

    Parameters
    ----------
    image: 2d-array
        Phase contrast image of interest.
    level: float
        Level at which to draw contours on black top-hat filtered image.
        Default value is 0.3.
    selem: 2d-array or string
        Structuring element to use for the black top-hat filtering procedure
        Default value is a disk with a diameter of 20 pixels.
    perim_bounds: length 2 tuple
        Lower and upper perimeter bounds of approved objects. This should be
        in units of microns. The default values are 5 and 25 microns for the
        lower and upper bound, respectively.
    ip_dist : float
        Interpixel distance of the image in units of microns per pixel. The
        default value is 0.160 microns per pixel.
    area_bounds : tuple of float
        Upper and lower bounds for selected object areas. These should be
        given in units of square microns.
    ecc_bounds : tuple of float
        Bounds for object eccentricity. Default values are between 0.5 and 1.0.
    return_conts : bool
        If True, the x and y coordinates of the individual contours will be
        returned. Default value is False

    Returns
    -------
    im_lab : 2d-array, int
        Two dimensional image where each individual object is labeled.

    conts : 1d-array
        List of contour coordinates. Each entry of this array comes as
        an x,y pair of arrays. Has the same length as the number of
        contoured objects. This is only returned if `return_conts` is
        True.

    """

    # Apply the white top-hat filter.
    if selem == 'default':
        selem = skimage.morphology.disk(20)

    # Normalize the image.
    image = (image - image.min()) / (image.max() - image.min())

    # Blur and background subtract the image.
    im_blur = skimage.filters.gaussian(image, sigma=5)
    im_sub = image - im_blur

    # Apply the black tophat filter.
    im_filt = skimage.morphology.black_tophat(im_sub, selem)

    # Find the contours and return.
    conts = skimage.measure.find_contours(im_filt, level)

    # Make an empty image for adding the approved objects.
    objs = np.zeros_like(image)

    # Loop through each contour.
    for _, c in enumerate(conts):
        perim = 0
        for j in range(len(c) - 1):
            # Compute the distance between points.
            distance = np.sqrt((c[j + 1, 0] - c[j, 0])**2 +
                               (c[j + 1, 1] - c[j, 1])**2)
            perim += distance * ip_dist

        # Test if the perimeter is allowed by the user defined bounds.
        if (perim > perim_bounds[0]) & (perim < perim_bounds[1]):

            # Round the contours.
            c_int = np.round(c).astype(int)

            # Color the image with the contours and fill.
            objs[c_int[:, 0], c_int[:, 1]] = 1.0

    # Fill and label the objects.
    objs_fill = scipy.ndimage.binary_fill_holes(objs)
    objs_fill = skimage.morphology.remove_small_objects(objs_fill)
    im_lab = skimage.measure.label(objs_fill)

    # Apply filters.
    approved_obj = np.zeros_like(im_lab)
    props = skimage.measure.regionprops(im_lab, image)
    for prop in props:
        area = prop.area * ip_dist**2
        ecc = prop.eccentricity
        if (area < area_bounds[1]) & (area > area_bounds[0]) &\
            (ecc < ecc_bounds[1]) & (ecc > ecc_bounds[0]) &\
                (prop.mean_intensity < min_int):
            approved_obj += (im_lab == prop.label)
    im_lab = skimage.measure.label(approved_obj)

    if return_conts is True:
        return conts, im_lab
    else:
        return im_lab

