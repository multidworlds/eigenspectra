import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from matplotlib import cm
from matplotlib import colors


def generate_map2d(hue_quantity, lightness_quantity, hue_cmap,
                   hue_range=None, lightness_range=None,
                   scale_min=0, scale_max=100):
    '''
    Given an existing color map, and two ranges of values, map the first range
    to hue, and the second to lightness by scaling the first component of the
    CIELab expression for the colors representing each hue.

    Parameters
    ----------

    hue_quantity       : array of floats, values to map to hue
    lightness_quantity : array of floats, values to map to lightness
    hue_cmap           : colormap object,
                         gradient of colors defining the range of hues
    hue_range          : list,
                         minimum and maximum of normalization for hue mapping
    lightness_range    : list,
                         minimum and maximum of normalization for hue mapping
    scale_min          : float, minimum perceived lightness (0-100)
    scale_max          : float, maximum perceived lightness (0-100)

    Returns
    -------

    rgb_maps : array of RGB colors, same shape as quantity
    '''

    if hue_range is None:
        hue_range = [np.nanmin(hue_quantity),
                     np.nanmax(hue_quantity)]
    if lightness_range is None:
        lightness_range = [np.nanmin(lightness_quantity),
                           np.nanmax(lightness_quantity)]

    if (scale_min < 0 or scale_min > 100):
        raise Exception('The scale minimum is invalid!')
    if (scale_max < 0 or scale_max > 100):
        raise Exception('The scale maximum is invalid!')
    if (scale_min > scale_max):
        raise Exception('The scale minimum should be less than ' +
                        'or equal to the scale maximum.')

    hue_scale = colors.Normalize(*hue_range)
    hue_map = cm.ScalarMappable(hue_scale, cmap=hue_cmap).get_cmap()
    hues = hue_map(hue_quantity)

    lightness = lightness_quantity / \
        (np.ptp(lightness_range) / (scale_max - scale_min)) - \
        (np.nanmin(lightness_range) - scale_min)
    lightness[lightness < scale_min] = scale_min
    lightness[lightness > scale_max] = scale_max

    # colors_CIElab = np.zeros(np.r_[hues.shape[:-1], 3])
    rgb_maps = np.zeros(np.r_[hues.shape[:-1], 3])
    for index in np.ndindex(hues.shape[:-1]):
        # colors_CIElab[index] = colormodels.lab_from_xyz(
        #     colormodels.xyz_from_rgb(hues[index][:-1]))
        lab_base = convert_color(sRGBColor(*(hues[index][:-1])), LabColor)
        lab_scale = LabColor(lab_a=lab_base.lab_a,
                             lab_b=lab_base.lab_b,
                             lab_l=lightness[index])
        rgb_scale = np.asarray(
                convert_color(lab_scale, sRGBColor).get_value_tuple()
                )
        rgb_scale[rgb_scale < 0] = 0
        rgb_scale[rgb_scale > 1] = 1
        rgb_maps[index] = rgb_scale

    return rgb_maps
