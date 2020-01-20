import os
import skimage.io
import skimage.measure
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import bokeh.plotting
import bokeh.io
import bokeh.palettes
import bokeh.layouts


# ---------------------------------------------------------------------------
# Plotting styles
# ---------------------------------------------------------------------------
def set_plotting_style(return_colors=True):
    """
    Sets the plotting style.

    Parameters
    ----------
    return_colors: Bool
        If True, this will also return a palette of eight color-blind safe
        colors with the hideous yellow replaced by 'dusty purple.'
    """
    rc = {'axes.facecolor': '#E3DCD0',
          'font.family': 'Lucida Sans Unicode',
          'grid.linestyle': '-',
          'grid.linewidth': 0.5,
          'grid.alpha': 0.75,
          'grid.color': '#ffffff',
          'mathtext.fontset': 'stixsans',
          'mathtext.sf': 'sans',
          'legend.frameon': True,
          'legend.facecolor': '#FFEDCE',
          'figure.dpi': 150}

    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('darkgrid', rc=rc)
    # colors = sns.color_palette('colorblind', n_colors=8)
    # colors[4] = sns.xkcd_palette(['dusty purple'])[0]
    colors = {'green': '#7AA974', 'light_green': '#BFD598',
              'pale_green': '#DCECCB', 'yellow': '#EAC264',
              'light_yellow': '#F3DAA9', 'pale_yellow': '#FFEDCE',
              'blue': '#738FC1', 'light_blue': '#A9BFE3',
              'pale_blue': '#C9D7EE', 'red': '#D56C55', 'light_red': '#E8B19D',
              'pale_red': '#F1D4C9', 'purple': '#AB85AC',
              'light_purple': '#D4C2D9'}
    if return_colors:
        return colors

def save_seg(fname, image, mask, fill_contours=True, ip_dist=0.160,
             bar_length=10, title=None, colormap='hls'):
    """
    Saves a merge of a segmentation mask and the original image for a
    sanity check.

    Parameters
    ----------
    fname : str
        The file will be saved with this path.
    image : 2d-array, float
        The original image on which the segmentation mask will be overlaid.
    mask : 2d-array, bool
        Boolean segmentation mask of the original image.
    contours: bool
        If True, contours of segmented objects will be filled.
    ip_dist : float
        Interpixel distance for the image. This is used for computing the
        scalebar length.  This should be in units of microns. Default
        value is 0.160 microns per pixel.
    bar_length : int
        The length of the desired scalebar in units of microns.
    title : str, optional
        Title for the image.
    colormap : str
        Colormap for labeling the objects. Default is the high-contrast
        'hls'. This can take any standard colormap string.

    Return
    ------
    fig : Matplotlib Figure object
        Figure containing the axis of the plotted image.
    """

    # Make copies of the image and mask.
    image_copy = np.copy(image)
    mask_copy = np.copy(mask)

    # Burn the scalebar into the upper-left hand  of th image.
    num_pix = int(bar_length / ip_dist)
    image = (image_copy - image_copy.min()) /\
            (image_copy.max() - image_copy.min())
    image[10:20, 10:10 + num_pix] = 1.0

    # Make sure the mask is a boolean image.
    if type(mask) != bool:
        mask = mask_copy > 0

    # Find the contours of the mask.
    conts = skimage.measure.find_contours(mask, 0)

    # Plot the image and generate the contours.
    with sns.axes_style('white'):
        fig = plt.figure()
        plt.imshow(image, cmap=plt.cm.Greys_r)

        # Plot all of the contours
        colors = sns.color_palette(colormap, n_colors=len(conts))
        for i, c in enumerate(conts):
            plt.plot(c[:, 1], c[:, 0], color=colors[i], lw=0.75)
            if fill_contours is True:
                plt.fill(c[:, 1], c[:, 0], color=colors[i], alpha=0.5)

        # Remove the axes.
        plt.xticks([])
        plt.yticks([])

        # Add title if provided.
        if title is not None:
            plt.title(title)

        # Tighten up and save the image.
        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
    return fig

