# jm_utils/data/jm_matplotlib
"""
¡?
"""

## TO-DO
## pie - paretto - en lo posible mismos parámetros


__version__ = "0.1.0"
__description__ = "Custom pandas functions for data cleaning and manipulation."
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-30"


## Standard Libs
import random
import textwrap
from typing import Union, Optional, Any, Literal, Sequence, TypeAlias

# Third-Party Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors             # for get_color_list()
from matplotlib.ticker import PercentFormatter  # for pareto chart() and... ?
from matplotlib import colormaps                # for show_matplotlib_palettes()
# import seaborn as sns

# Local Libs
from jm_utils.data.pd_functions import to_series, get_fdt

## Custom types for non-included typing annotations
IndexElement: TypeAlias = Union[str, int, float, pd.Timestamp]
# IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', np.str_, np.int64, np.float64, np.datetime64, pd.Timestamp, ...]

MPL_CMAP_NAMES = {
             "Cyclic": ([
            "hsv", "hsv_r", "twilight", "twilight_r", "twilight_shifted", "twilight_shifted_r"
        ], 'For values that wrap around at the endpoints, such as phase angle, wind direction, or time of day'),
        
        "Diverging": ([
            "BrBG", "BrBG_r", "bwr", "bwr_r", "coolwarm", "coolwarm_r", "PiYG", "PiYG_r",
            "PRGn", "PRGn_r", "PuOr", "PuOr_r", "RdBu", "RdBu_r", "RdGy", "RdGy_r",
            "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r", "seismic", "seismic_r", "Spectral", "Spectral_r"
        ], 'When the information being plotted has a critical middle value, such as topography or when the data deviates around zero'),
        
        "Miscellaneous": ([
            "brg", "brg_r", 'CMRmap', 'CMRmap_r', 'cubehelix', 'cubehelix_r', "flag", "flag_r",
            'gist_earth', 'gist_earth_r', "gist_ncar", "gist_ncar_r", "gist_rainbow", "gist_rainbow_r", "gist_stern", "gist_stern_r",
            "gnuplot", "gnuplot_r", "gnuplot2", "gnuplot2_r", "jet", "jet_r", "nipy_spectral", "nipy_spectral_r",
            'ocean', 'ocean_r', "prism", "prism_r", 'terrain', 'terrain_r', 'turbo', 'turbo_r',
            "rainbow", "rainbow_r"
        ], 'Particular uses for which they have been created. E.G gist_earth, ocean, and terrain for plotting topography'),
        
        "Perceptually Uniform Sequential": ([
            "cividis", "cividis_r", "inferno", "inferno_r", "magma", "magma_r", "plasma", "plasma_r",
            "viridis", "viridis_r"
        ], 'For representing information that has ordering'),
        
        "Single-Hue Sequential": ([
            "binary", "binary_r", "Blues", "Blues_r", "bone", "bone_r", "gist_gray", "gist_gray_r",
            "gist_yarg", "gist_yarg_r", "gray", "gray_r", "Grays", "Grays_r", "Greens", "Greens_r",
            "grey", "grey_r", "Greys", "Greys_r", "Oranges", "Oranges_r", "Purples", "Purples_r",
            "Reds", "Reds_r"
        ], 'For representing information that has ordering'),
        
        "Multi-Hue Sequential": ([
            "autumn", "autumn_r", "BuGn", "BuGn_r", "BuPu", "BuPu_r", "cool", "cool_r",
            "GnBu", "GnBu_r", "OrRd", "OrRd_r", "PuBu", "PuBu_r", "PuBuGn", "PuBuGn_r",
            "PuRd", "PuRd_r", "spring", "spring_r", "summer", "summer_r", "winter", "winter_r",
            "YlGn", "YlGn_r", "YlGnBu", "YlGnBu_r", "YlOrBr", "YlOrBr_r", "YlOrRd", "YlOrRd_r"
        ], 'For representing information that has ordering'),
        
        "Other Sequential": ([
            "afmhot", "afmhot_r", "berlin", "berlin_r", "CMRmap", "CMRmap_r", "copper", "copper_r",
            "crest", "crest_r", "cubehelix", "cubehelix_r", "flare", "flare_r", "gist_earth", "gist_earth_r",
            "gist_grey", "gist_grey_r", "gist_heat", "gist_heat_r", "gist_yarg", "gist_yarg_r", "gist_yerg",
            "gist_yerg_r", "hot", "hot_r", "icefire", "icefire_r", "mako", "mako_r", "managua", "managua_r",
            "ocean", "ocean_r", "pink", "pink_r", "rocket", "rocket_r", "terrain", "terrain_r",
            "vanimo", "vanimo_r", "vlag", "vlag_r", "Wistia", "Wistia_r"
        ], 'For representing information that has ordering'),
        
        "Qualitative": ([
            "Accent", "Accent_r", 'colorblind', "Dark2", "Dark2_r", "Paired", "Paired_r", "Pastel1",
            "Pastel1_r", "Pastel2", "Pastel2_r", "Set1", "Set1_r", "Set2", "Set2_r", "Set3",
            "Set3_r", "tab10", "tab10_r", "tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c",
            "tab20c_r"
        ], 'To represent information which does not have ordering or relationships. Also for categorical data')
}

#--------------------------------------------------------------------------------------------------------------------------------#
#  CHARTs Functions:
#--------------------------------------------------------------------------------------------------------------------------------#
#   - Aux: get_colorblind_palette_list(), get_colors_list(),  _validate_numeric_series()
# Common parameters for categorical charts:
#   - data: Union[pd.Series, pd.DataFrame], | One or two col DF. Case two cols 1se col is index (categories) and 2nd values
#   - value_counts: Optional[bool] = False, | You can plot native values or aggregated ones by categories
#   - scale: Optional[int] = 1,             | All sizes, widths, etc. are scaled from this number (from 1 to 9)
#   - ...


def get_color_list(palette: str, n_colors: Optional[int] = 10) -> list[str]:
    """
    Returns a list of hex color codes from a specified Matplotlib colormap or a named palette.

    This function generates a list of colors suitable for data visualization. It supports
    custom Matplotlib colormaps and a predefined 'colorblind' palette optimized for
    accessibility.

    Parameters:
        palette (str): Name of the colormap or palette to use. Special value:
            - 'colorblind': Returns a predefined colorblind-safe palette.
            - Any other string: Interpreted as a Matplotlib colormap (e.g., 'viridis', 'plasma').

        n_colors (int, optional): Number of colors to generate from the colormap.
            Ignored if palette is 'colorblind' (which returns a fixed set).
            Default is 10.

    Returns:
        list[str]: A list of hexadecimal color codes (e.g., '#0173B2').

    Raises:
        ValueError: If the specified Matplotlib colormap does not exist.
        TypeError: If `n_colors` is not a number.

    Notes:
        - For the 'colorblind' palette, the function returns a fixed set of 30 colors.
          If more than 30 are requested, they will be truncated.
        - Uses `matplotlib.pyplot.get_cmap` and `matplotlib.colors.rgb2hex` internally.
        - Ideal for use in custom plotting functions requiring consistent, accessible color schemes.

    Example:
        >>> get_color_list('viridis', 3)
        ['#440154', '#21908C', '#FDE725']

        >>> get_color_list('colorblind', 5)
        ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC']
    """
    if palette == 'colorblind':
        return [
            '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
            '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9',
            '#5D8C3B', '#A93967', '#888888', '#FFC107', '#7C9680',
            '#E377C2', '#BCBD22', '#AEC7E8', '#FFBB78', '#98DF8A',
            '#FF9896', '#C5B0D5', '#C49C94', '#F7B6D2', '#DBDB8D',
            '#9EDAE5', '#D68E3A', '#A65898', '#B2707D', '#8E6C87'
        ]
    else:
        cmap = plt.get_cmap(palette)                        # Get the colormap
        colors_normalized = np.linspace(0, 1, n_colors)     # Generate equidistant points between 0 and 1
        colors_rgba = cmap(colors_normalized)               # Get the colors from colormap
        return [mcolors.rgb2hex(color[:3]) for color in colors_rgba]
    

def get_matplotlib_palettes_dict():
    """ matplotlib v. 3.10.3 """
    matplotlib_palettes = {
        "Cyclic": ([
            "hsv", "hsv_r", "twilight", "twilight_r", "twilight_shifted", "twilight_shifted_r"
        ], 'For values that wrap around at the endpoints, such as phase angle, wind direction, or time of day'),
        
        "Diverging": ([
            "BrBG", "BrBG_r", "bwr", "bwr_r", "coolwarm", "coolwarm_r", "PiYG", "PiYG_r",
            "PRGn", "PRGn_r", "PuOr", "PuOr_r", "RdBu", "RdBu_r", "RdGy", "RdGy_r",
            "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r", "seismic", "seismic_r", "Spectral", "Spectral_r"
        ], 'When the information being plotted has a critical middle value, such as topography or when the data deviates around zero'),
        
        "Miscellaneous": ([
            "brg", "brg_r", 'CMRmap', 'CMRmap_r', 'cubehelix', 'cubehelix_r', "flag", "flag_r",
            'gist_earth', 'gist_earth_r', "gist_ncar", "gist_ncar_r", "gist_rainbow", "gist_rainbow_r", "gist_stern", "gist_stern_r",
            "gnuplot", "gnuplot_r", "gnuplot2", "gnuplot2_r", "jet", "jet_r", "nipy_spectral", "nipy_spectral_r",
            'ocean', 'ocean_r', "prism", "prism_r", 'terrain', 'terrain_r', 'turbo', 'turbo_r',
            "rainbow", "rainbow_r"
        ], 'Particular uses for which they have been created. E.G gist_earth, ocean, and terrain for plotting topography'),
        
        "Perceptually Uniform Sequential": ([
            "cividis", "cividis_r", "inferno", "inferno_r", "magma", "magma_r", "plasma", "plasma_r",
            "viridis", "viridis_r"
        ], 'For representing information that has ordering'),
        
        "Single-Hue Sequential": ([
            "binary", "binary_r", "Blues", "Blues_r", "bone", "bone_r", "gist_gray", "gist_gray_r",
            "gist_yarg", "gist_yarg_r", "gray", "gray_r", "Grays", "Grays_r", "Greens", "Greens_r",
            "grey", "grey_r", "Greys", "Greys_r", "Oranges", "Oranges_r", "Purples", "Purples_r",
            "Reds", "Reds_r"
        ], 'For representing information that has ordering'),
        
        "Multi-Hue Sequential": ([
            "autumn", "autumn_r", "BuGn", "BuGn_r", "BuPu", "BuPu_r", "cool", "cool_r",
            "GnBu", "GnBu_r", "OrRd", "OrRd_r", "PuBu", "PuBu_r", "PuBuGn", "PuBuGn_r",
            "PuRd", "PuRd_r", "spring", "spring_r", "summer", "summer_r", "winter", "winter_r",
            "YlGn", "YlGn_r", "YlGnBu", "YlGnBu_r", "YlOrBr", "YlOrBr_r", "YlOrRd", "YlOrRd_r"
        ], 'For representing information that has ordering'),
        
        "Other Sequential": ([
            "afmhot", "afmhot_r", "berlin", "berlin_r", "CMRmap", "CMRmap_r", "copper", "copper_r",
            "crest", "crest_r", "cubehelix", "cubehelix_r", "flare", "flare_r", "gist_earth", "gist_earth_r",
            "gist_grey", "gist_grey_r", "gist_heat", "gist_heat_r", "gist_yarg", "gist_yarg_r", "gist_yerg",
            "gist_yerg_r", "hot", "hot_r", "icefire", "icefire_r", "mako", "mako_r", "managua", "managua_r",
            "ocean", "ocean_r", "pink", "pink_r", "rocket", "rocket_r", "terrain", "terrain_r",
            "vanimo", "vanimo_r", "vlag", "vlag_r", "Wistia", "Wistia_r"
        ], 'For representing information that has ordering'),
        
        "Qualitative": ([
            "Accent", "Accent_r", 'colorblind', "Dark2", "Dark2_r", "Paired", "Paired_r", "Pastel1",
            "Pastel1_r", "Pastel2", "Pastel2_r", "Set1", "Set1_r", "Set2", "Set2_r", "Set3",
            "Set3_r", "tab10", "tab10_r", "tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c",
            "tab20c_r"
        ], 'To represent information which does not have ordering or relationships. Also for categorical data')
    }

    return matplotlib_palettes


def show_matplotlib_palettes(
        palette_group: Union[str, list[str]] = 'Sample',
        n_colors: Optional[int] = 64,
        discrete: Optional[bool] = True
) -> plt.Figure:
    """
    Displays a visual comparison of Matplotlib colormaps (palettes) in a grid layout.

    This function creates a figure showing color swatches for a selected group of
    Matplotlib colormaps. It supports built-in categories, a representative sample,
    or a custom list of palettes. The display can be either discrete color bars or
    continuous gradient strips.

    Parameters:
        palette_group (Union[str, list[str]]): Specifies which palettes to display.
            - If str: one of the built-in groups (e.g., 'Qualitative', 'Sequential')
              or special options:
              - 'Sample': Shows a selection from all main categories.
              - 'Names': Displays a text list of all available palettes by category.
            - If list: A custom list of colormap names to display.
            Case is insensitive for string inputs. Default is 'Sample'.

        n_colors (int, optional): Number of discrete color swatches to show per palette.
            Used only when `discrete=True`. Must be between 1 and 99.
            Default is 64.

        discrete (bool, optional): If True, displays palettes as discrete color bars.
            If False, displays them as continuous color gradients.
            Default is True.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing all palette views.
            This allows further customization, saving, or inspection after display.

    Raises:
        TypeError: If `palette_group` is not a string or list of strings, or if `n_colors`
            is not a number.
        ValueError: If `n_colors` is not in the valid range (1–99).
        ValueError: If `palette_group` is a string but not a recognized category or option.

    Notes:
        - Invalid or deprecated colormap names are handled gracefully and labeled in red.
        - The 'colorblind' palette (custom) is excluded from continuous mode as it's not a
          standard Matplotlib colormap.
        - The layout adapts to the number of palettes, using one or two columns for discrete mode.
        - Uses `get_color_list` internally for discrete color extraction.
        - Ideal for exploring, comparing, and selecting appropriate color schemes for
          data visualization.

    Example:
        >>> show_plt_palettes('Sequential', n_colors=12)
        # Displays 12-color samples for all Sequential palettes.

        >>> show_plt_palettes(['viridis', 'plasma', 'coolwarm'], discrete=False)
        # Shows continuous gradients for three specific palettes.

        >>> show_plt_palettes('Names')
        # Prints a list of all available Matplotlib colormap categories and names.

        >>> show_plt_palettes()
        # Shows a default sample of palettes from various categories.
    """
    
    # First verified n_colors parameter (cause validation and preprocess palette_group parameter need more data)
    if not isinstance(n_colors, (int, float)):
        raise TypeError(f"'n_colors' parameter not valid. Must be an int or float. Got '{type(n_colors)}'.")

    if n_colors < 1 or n_colors > 99:
        raise ValueError(f"'n_colors' parameter not valid. Must be > 1 and < 99 . Got '{n_colors}'.")
    n_colors = int(n_colors) + 1
    
    # Get the known matplotlib palettes in a dict by categories plus addtion of 'Sample' key-value (later we must add also 'Custom' key-value if pallete_group is a list)
    palettes_by_category_dic = get_matplotlib_palettes_dict()                                   # dict_keys(['Cyclic', 'Diverging', 'Miscellaneous', 'Perceptually Uniform Sequential', 'Single-Hue Sequential', 'Multi-Hue Sequential', 'Special Sequential', 'Qualitative'])
    
    list_of_pltt_lists = [value[0] for value in palettes_by_category_dic.values()]              # Nedeed as source of data to get a random sample of 4 palettes of e/category
    palettes_by_category_dic['Sample'] = (                                                      # Added 'Sample' dict_key
        [pltt for p_g in list_of_pltt_lists for pltt in random.sample(p_g, k=4)],               # A random sample of four of e/category
        "4 Cyclic, 4 Diverging, 4 Miscellaneous, 4 Perceptually Uniform Sequential,"
        "4 Single-Hue Sequential, 4 Multi-Hue Sequential, 4 Other Sequential, 4 Qualitative")

    # Internal auxiliary function that generates a figure containing the names of the palettes according to their type.
    def _show_dic(dic):
        all_text =""
        for group_name, (palette_list, description) in dic.items():
            if group_name == 'Sample':
                continue
            all_text += f"* {group_name}.- ({len(palette_list)}) {description}:\n"
            wrapped = textwrap.fill(", ".join(palette_list), width=140, initial_indent="    ", subsequent_indent="    ")
            all_text += wrapped + "\n\n"
        # Build de Figure showing all text
        fig, ax = plt.subplots(figsize=(8, len(all_text.splitlines()) * 0.2))
        ax.set_axis_off()                          # Hide x and y axis
        ax.set_title('Matplotlib colormaps by category  - palette_group(s)', fontsize=12, fontweight='medium', family="monospace")
        ax.text(0.025, 0.42, all_text, fontsize=10, va="center", ha="left", family="monospace")
        plt.show()
        return fig            
    
    # Validate and preprocess palette_group parameter: get the palette_group_key, print lists/names if selected, or fill custom list
    if isinstance(palette_group, str):          
        palette_group_key = palette_group.strip().title()
        if palette_group_key == 'Names':
            fig = _show_dic(palettes_by_category_dic)
            return fig
        elif palette_group_key not in palettes_by_category_dic.keys():
            raise ValueError(f"Invalid value for 'palette_group': {repr(palette_group)}. Expected one of:"
                             "'Cyclic', 'Diverging', 'Miscellaneous', 'Perceptually Uniform Sequential', 'Single-Hue Sequential'," 
                             "'Multi-Hue Sequential', 'Other Sequential', 'Qualitative', 'Custom', 'Sample'.")
        else:
            # Get the palette_group_list and palette_group_desc of the selected palette category (group)
            palette_group_list, palette_group_desc = palettes_by_category_dic[palette_group_key]
    elif isinstance(palette_group, list):   
        palette_group_key = 'Custom'            # Only for title, no a new entry to de dictionary
        palette_group_list = palette_group      # The list of entered palettes to be shown
        palette_group_desc = 'User selected palettes'
    else:
        raise TypeError(f"Invalid type for 'palette_group': {repr(palette_group)}. Expected one of: 'str' or 'list'.")

    if discrete:                                                # Displays n_colors from the palette slightly separated by a thin white line
        # Build a Series of n_items elements to show colors
        sr = to_series({str(i): 1 for i in range(1, n_colors)})

        # Create a figure with two columns for the palettes - Bar charts showing palette colors
        rows = len(palette_group_list) // 2 if len(palette_group_list) % 2 == 0 else (len(palette_group_list) // 2) + 1
        width = 12                                              # Fixed width at 12 for now (can we look into making it proportional to n_colors?)
        height = rows / 1.75 if rows > 5 else rows / 1.375      # To avoid overlapping axes when there are few rows
        
        fig, axs = plt.subplots(rows, 2, figsize=(width, height), sharex=True, gridspec_kw=dict(wspace=0.1), constrained_layout=True)

        # Set the figure title and subtitle with the palette group key and description
        fig.suptitle(f"* Matplolib {palette_group_key} colormaps (palettes) - {len(palette_group_list)} *\n{palette_group_desc}",
                    fontsize=12, fontweight='medium')

        # Iterate over the axes and palette group to plot each palette                                           
        for ax, pltt in zip(axs.flatten(), palette_group_list):
            try:
                color_list = get_color_list(pltt, n_colors)
                ax.bar(sr.index, sr, color=color_list, width=1, edgecolor='white', linewidth=0.2)
                ax.set_xlim(-0.5, n_colors - 1.5)
                ax.set_ylim(0, 0.1)
                ax.set_title(pltt, loc='left', fontsize=10, fontweight='medium')
            except ValueError:
                err_msg = f"'{pltt}' is not currently a valid Matplotlib palette (cmap)"
                ax.set_title(err_msg, loc='left', fontsize=10, fontweight='medium', color='red')

            ax.set_axis_off()
    else:                                               # Displays a continuous strip of colors
        try:    # Remove 'colorblind' if exist in the selected list (Qualitative or Sample)
            palette_group_list.remove('colorblind')     # 'colorblind' in Matplotlib is jm construction
        except ValueError:
            pass

        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # Create figure and adjust figure height to number of colormaps
        nrows = len(palette_group_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                            left=0.2, right=0.99)
        axs[0].set_title(f"{palette_group_key} colormaps\n{palette_group_desc}", fontsize=12)

        for ax, name in zip(axs, palette_group_list):
            ax.imshow(gradient, aspect='auto', cmap=colormaps[name])
            ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()

    plt.show()
    return fig                  # Return the current figure for further manipulation if needed


    
if __name__ == "__main__":

    # Data
    dic = {'1603 SW': [21, 'No POE'], '1608 SW': [6, 'Headset compatible'], 
       '1616 SW': [3, 'Telefonista'], '9611 G': [8, 'Gerencial Gigabit']}
    df = pd.DataFrame.from_dict(dic, orient='index', columns=['Stock', 'Obs'])

    # Show palettes
    show_plt_palettes()



