# jm_utils/data/matplotlib_tints
"""
Diccionarios y funciones relacionadas con colores y paletas matplotlib

Utilizamos 'tints' en vez de color en el nombre para que no se confunda con el módulo 'colors' original de matplot lib
-> import jm_utils.data.matplotlib_tints as mpl_tints
"""

## TO-DO
## pie - paretto - en lo posible mismos parámetros


__version__ = "0.1.0"
__description__ = "Diccionarios y funciones relacionadas con colores y paletas matplotlib."
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


def ax_plt_pie(
    ax: plt.Axes,
    data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
    value_counts: Optional[bool] = False,
    dropna: Optional[bool] = True,
    order: Optional[str] = 'desc',
    scale: Optional[int] = 1,
    title: Optional[str] = None,
    kind: Optional[str] = 'pie',
    pct_label_place: Optional[str] = 'ext',     # mix, mixlgd, apart, ext
    palette: Optional[list] = 'colorblind',
    startangle: Optional[float] = 90,
    pct_decimals: Optional[int] = 1,
    labels_rotate: Optional[float] = 0,
    labels_color: Optional[str] = 'black',
    legends_loc: Optional[str] = 'best',
    legends_title: Optional[str] = None,
    show_stats_subtitle = True
) -> plt.Axes:
    """
    Plots a pie or donut chart on a given matplotlib Axes with advanced label and layout options.

    This function is designed to be used internally or within subplot grids. It draws a pie or donut
    chart on a pre-existing Axes object, allowing for integration into complex figure layouts.

    Parameters:
        ax (plt.Axes): The matplotlib Axes object on which to draw the chart.

        data (Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]): Input data.
            Will be converted to a frequency distribution using `get_fdt`.

        value_counts (bool, optional): If True, computes frequency counts of raw data.
            Default is False.

        dropna (bool, optional): If True, excludes NaN values from the chart.
            If False, includes NaN as a category. Default is True.

        order (str, optional): Sorting order for categories:
            - 'desc': descending by frequency.
            - 'asc': ascending by frequency.
            - 'ix_asc': ascending by index.
            - 'ix_desc': descending by index.
            Default is 'desc'.

        scale (int, optional): Scaling factor (1 to 16) affecting font sizes.
            Larger values produce larger text. Default is 1.

        title (str, optional): Chart title. If not provided, a default title is generated.

        kind (str, optional): Type of chart:
            - 'pie': standard pie chart.
            - 'donut': donut chart with a hole in the center.
            Default is 'pie'.

        pct_label_place (str, optional): Placement and style of labels:
            - 'ext': external labels with arrows.
            - 'mix': internal percentages and absolute values, labels outside.
            - 'mixlgd': internal values, legend with labels.
            - 'apart': no internal labels, full details in legend.
            Default is 'ext'.

        palette (list, optional): List of colors or a named palette (e.g., 'colorblind').
            If None, defaults to a standard palette.

        startangle (float, optional): Starting angle in degrees for the first wedge.
            Default is 90 (top, vertical).

        pct_decimals (int, optional): Number of decimal places in percentage labels.
            Default is 1.

        labels_rotate (float, optional): Rotation angle for internal labels.
            Default is 0.

        labels_color (str, optional): Color for internal text labels.
            Default is 'black'.

        legends_loc (str, optional): Position of the legend (e.g., 'best', 'upper right').
            Default is 'best'.

        legends_title (str, optional): Title for the legend.
            Default is None.

        show_stats_subtitle (bool, optional): If True, adds a subtitle with summary statistics:
            total count, number of categories, contribution of top 2, and null count.
            Default is True.

    Returns:
        plt.Axes: The modified Axes object with the pie/donut chart drawn.

    Raises:
        ValueError: If `kind` is not 'pie' or 'donut', or if more than 12 categories are provided.
        ValueError: If `scale` is not between 1 and 16.
        ValueError: If `pct_label_place` is invalid.

    Notes:
        - This function uses `get_fdt` to compute the frequency distribution table.
        - NaN handling is flexible: can be included, excluded, or sorted with values.
        - Designed for reuse and integration into larger figures.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax_plt_pie(ax, data, kind='donut', pct_label_place='mixlgd')
        >>> plt.show()
    """
    # Get the data to graph: use controls and processing that get_fdt() does to obtain the series to graph (first column: 'Frequency')
    fdt = get_fdt(data, value_counts=value_counts, order=order,
                  dropna=False, na_position='value', na_aside_calc=False, include_flat_relatives=False, include_pcts=False)
        # - dropna=False            -> So that it doesn't remove NaNs, and then handle them
        # - na_aside_calc=False     -> So that it allows me to sort the nan value with na_position='value'
        # - na_position='value'     -> So that it allows me to sort the nan value within the list of values either desc or asc (according to order)
        # - We do not include relative flat rates or percentages. We calculate the percentages before presenting them.

    # cat_name: lo puedo usar luego in title y en legend titel
    cat_name = fdt.index.name                       # Category name <- from fdt.index.name (could be 'Index' Warn!)

    sr = fdt.iloc[:, 0]                             # Get the Series with the frequencies (count)
    # As sr.index build the legends: If I want to change the legends, I'll have to see how I modify this sr.index (must be done previously of run this funct)
         
    # Handling of nans since they are presented in the subtitle, whether or not they appear in the graph
    total_label = "Total (w/nulls)"                # total_label to be presented in subtitle (inital value w/nulls)
    
    if pd.isna(sr.index).any():                     # There is np.nan [NaN] index, nans values
        n_nans = sr[np.nan]
        if dropna:                                  # No NaNs in the graph
            sr = sr.drop(np.nan, errors='ignore')   # Drop NaN row from the DataFrame
            total_label = "Total (wo/nulls)"       # The total will be calculated wo/NaNs (likewise, n_nans will appear in the subtitle.)
    else:                                           # No np.nan row
        n_nans = 0

    # Validate kind parameter
    if kind.lower() not in ['pie', 'donut']:
        raise ValueError(f"Invalid 'kind' parameter: '{kind}'. Must be 'pie' or 'donut'.")
    
    # Validate maximum categories
    if len(sr) > 12:
        raise ValueError(f"Data contains {len(sr)} categories. "
                        "Maximum allowed is 12 categories.")
    
    # Build graphs size, and fonts size from scale, and validate scale from 1 to 9.
    if scale < 1 or scale > 16:
        raise ValueError(f"[ERROR] Invalid 'scale' value. Must between '1' and '16', not '{scale}'.")
    else:
        scale = round(scale)

    ## OJO! possible need to validate a lot of other parameters
    
    # Calculate font sizes based scale
    multiplier= scale + 5
    labels_size = multiplier * 1.1
    title_size = multiplier * 1.6

    # Configure wedge properties for donut or pie chart
    if kind.lower() == 'donut':
        wedgeprops = {'width': 0.55, 'edgecolor': 'white', 'linewidth': 0.6}
    else:
        wedgeprops = {'edgecolor': 'white', 'linewidth': 0.3}

    # Define colors
    if palette:
        color_list = get_color_list(palette, len(sr))
    else:
        color_list = None

    # Build the different pies according pct_label_place. Previous get the total = frequencies sum
    total = sr.sum()

    if pct_label_place == 'ext':                                # External pct and labels using ax.annotate()

        wedges, _ = ax.pie(sr, wedgeprops=wedgeprops, colors=color_list, startangle=startangle)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        # Build the labels. Annotations and legend in same label (External)
        labels = [
            f"{sr.loc[sr == value].index[0]}\n{value:,}\n({round(value / total * 100, pct_decimals)} %)"
            for value in sr.values
        ]
        
        # Draw the annotations (labels)
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.1*np.sign(x), 1.1*y),
                    horizontalalignment=horizontalalignment, fontsize=labels_size, **kw)
            
    elif pct_label_place == 'mix' or pct_label_place == 'mixlgd' or pct_label_place == 'apart':
        # Initlal values for some varibales to use in this section
        labels = None                   # Labels will only exist for the 'mix' case. Other cases None (labels in legends or 'ext')
        legends_size = labels_size        # It allows me to change the size of the legend other than the label.
        
        # Set autopct: for 'mix' and 'mixlgd' internal w/ absolute and pcts. Others None (all info outside the chart)
        if pct_label_place == 'mix' or pct_label_place == 'mixlgd':
            
            def _make_autopct(values):             # A python Closure
                value_iterator = iter(values)    
                def my_autopct(pct):
                    return f"{next(value_iterator):,}\n{pct:.{pct_decimals}f}%"  
                return my_autopct
            
            autopct_func = _make_autopct(sr.values)

            # Set labels props for 'mix'. The only case labels differento to None
            if pct_label_place == 'mix':
                labels = sr.index

        else:                       # elif apart
            autopct_func = None     # No data inside de pie or donut

        # Build the graph
        ax.pie(x=sr,
               colors=color_list,
               labels=labels,
               startangle=startangle,
               autopct=autopct_func,
               wedgeprops=wedgeprops,
               textprops={'size': labels_size,
                        'color': labels_color,
                        'rotation': labels_rotate,
                        'weight': 'medium'})
        
        # Legends only in case of 'mixlgd' or 'apart'
        if pct_label_place == 'mixlgd' or pct_label_place == 'apart':
            if pct_label_place == 'mixlgd':
                legends = sr.index
            else:                    # elif apart: all the info (pcts, labels, cat_names) in the legends
                legends = [f"{sr.index[i]} \n| {value:,} | {round(value / total * 100, pct_decimals)} %"
                           for i, value in enumerate(sr.values)] 
        
            ax.legend(legends,
                      title=legends_title,
                      title_fontproperties = {'size':legends_size, 'weight': 'bold'},
                      loc=legends_loc,
                    #   bbox_to_anchor=(1, 0, 0.2, 1),
                      bbox_to_anchor=(1, 0.9),
                      prop={'size': legends_size})

    else:
        raise ValueError(f"Invalid labe_place parameter. Must be 'mix, 'mixlgd', 'ext' or 'aoart', not '{pct_label_place}'.")
            
    # Build title and set title
    if not title:
        title = f"Pie/Donut Chart ({cat_name} - {sr.name})"
    ax.set_title(title, fontsize=title_size, fontweight='bold')

    if show_stats_subtitle:                     # Enhanced subtitle with statistics
        n_categories = len(sr)                  # len(categories)
        top_2_pct = (sr.head(2).sum() / total * 100) if n_categories >= 2 else 100

        subtitle = f"{total_label} {total:,} | Categories: {n_categories} | First 2: {top_2_pct:.1f}% | Nulls (nan): {n_nans}"
        ax.text(0, 1.18, subtitle, ha='center', va='center', fontsize=title_size * 0.7, color='dimgray')

    return ax


def plt_pie(
    data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
    value_counts: Optional[bool] = False,
    dropna: Optional[bool] = True,
    order: Optional[str] = 'desc',
    scale: Optional[int] = 1,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    kind: Optional[str] = 'pie',
    pct_label_place: Optional[str] = 'mix',
    palette: Optional[list] = None,
    startangle: Optional[float] = 90,
    pct_decimals: Optional[int] = 1,
    labels_rotate: Optional[float] = 0,
    labels_color: Optional[str] = 'black',
    legends_loc: Optional[str] = 'best',
    legends_title: Optional[str] = None,
    show_stats_subtitle = True   
) -> tuple[plt.Figure, plt.Axes]:
    """
    Creates a pie or donut chart with customizable layout, labels, and styling.

    This high-level function generates a complete pie or donut chart figure. It is ideal for
    standalone visualizations and quick data exploration. Internally, it uses `ax_plt_pie`
    to draw the chart on a newly created Axes.

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]): Input data.
            If not a Series, it will be converted using `to_series`.

        value_counts (bool, optional): If True, computes frequency counts of raw data.
            Default is False.

        dropna (bool, optional): If True, excludes NaN values from the chart.
            If False, includes NaN as a category. Default is True.

        order (str, optional): Sorting order for categories (see `ax_plt_pie`).
            Default is 'desc'.

        scale (int, optional): Scaling factor (1 to 16) affecting figure and font sizes.
            Default is 1.

        figsize (tuple, optional): Width and height of the figure in inches.
            If not provided, size is calculated from `scale`.

        title (str, optional): Chart title. If not provided, a default title is generated.

        kind (str, optional): Type of chart: 'pie' or 'donut'. Default is 'pie'.

        pct_label_place (str, optional): Label placement strategy (see `ax_plt_pie`).
            Default is 'mix'.

        palette (list, optional): Color palette to use. Default is None.

        startangle (float, optional): Starting angle for the first wedge. Default is 90.

        pct_decimals (int, optional): Decimal places in percentage labels. Default is 1.

        labels_rotate (float, optional): Rotation for internal labels. Default is 0.

        labels_color (str, optional): Color for internal text. Default is 'black'.

        legends_loc (str, optional): Legend position. Default is 'best'.

        legends_title (str, optional): Title for the legend. Default is None.

        show_stats_subtitle (bool, optional): If True, displays a subtitle with key statistics.
            Default is True.

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing:
            - fig: The matplotlib Figure object.
            - ax: The Axes object with the chart.

    Raises:
        ValueError: If `scale` is not between 1 and 16.
        ValueError: If `kind` or `pct_label_place` are invalid (delegated to `ax_plt_pie`).

    Notes:
        - This function is a wrapper around `ax_plt_pie`, providing a simple interface
          for creating full figures.
        - Ideal for interactive use, reports, and exploratory data analysis.

    Example:
        >>> fig, ax = plt_pie(data, kind='donut', title='Distribution of Categories')
        >>> plt.show()
    """
    # Build graphs size, and fonts size from scale, and validate scale.
    if scale < 1 or scale > 16:
        raise ValueError(f"[ERROR] Invalid 'scale' value. Must between '1' and '16', not '{scale}'.")
    else:
        scale = round(scale)

    # Calculate figure dimensions
    if figsize is None:
        multiplier = scale + 5
        w_base, h_base = 1.1, 0.6
        width, height = w_base * multiplier, h_base * multiplier
        figsize = (width, height)
    else:
        width, height = figsize
        scale = (width + height) / 2.5

    # Base fig definitions
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

    ax_plt_pie(ax, data, value_counts, dropna, order, scale, title, kind, pct_label_place, palette, startangle,
                pct_decimals, labels_rotate, labels_color, legends_loc, legends_title, show_stats_subtitle)    
                
    return fig, ax


def plt_pie0(
    data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
    value_counts: Optional[bool] = False,
    dropna: Optional[bool] = True,
    order: Optional[str] = 'desc',
    scale: Optional[int] = 1,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    kind: Optional[str] = 'pie',
    label_place: Optional[str] = 'ext',
    palette: Optional[list] = 'colorblind',
    startangle: Optional[float] = 90,
    pct_decimals: Optional[int] = 1,
    label_rotate: Optional[float] = 0,
    legend_loc: Optional[str] = 'best',
    show_stats_subtitle = True
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generates a pie or donut chart with customizable label placement and styling.

    This function creates a pie or donut chart from categorical data using matplotlib.
    It supports internal, external, or aside label placement with optional percentage
    and value annotations.

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],):
            Input data be converted to a Series using `to_series`.
        value_counts (bool, optional): If True, counts occurrences of each category.
            Default is False.
        sort (bool, optional): If True and `value_counts=True`, sorts categories by frequency.
            Default is True.
        nans (bool, optional): If True, includes NaN values in the count. Default is False.
        scale (int, optional): Chart scaling factor (1 to 9). Affects figure size and font sizes.
            Default is 1.
        figsize (tuple, optional): Width and height of the figure in inches. Overrides `scale`.
            Default is None.
        title (str, optional): Chart title. If not provided, a default title is used.
        kind (str, optional): Type of chart to generate. Options:
            - 'pie': standard pie chart.
            - 'donut': donut chart with a hollow center.
        label_place (str, optional): Placement of labels. Options:
            - 'ext': external labels connected by arrows.
            - 'int': internal labels within each segment (shows absolute values and percentages).
            - 'aside': internal labels and a legend with extended labels on the side.
        palette (list or str, optional): Color palette for segments. If a string, uses a predefined
            palette (e.g., 'set2' or 'viridis'). Default is 'colorblind'.
        startangle (float, optional): Starting angle (in degrees) for the first wedge.
            Default is -40.
        pct_decimals (int, optional): Number of decimal places to display in percentage values.
            Default is 1.
        label_rotate (float, optional): Rotation angle for internal labels (only applies if
            `label_place='int'`). Default is 0.
        legend_loc (str, optional): Position of the legend (if displayed). See valid options in
            `matplotlib.legend`. Default is 'best'.

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing:
            - fig: The Matplotlib Figure object.
            - ax: The Matplotlib Axes object for further customization.

    Raises:
        TypeError: If input data is not a pandas Series or DataFrame.
        ValueError: If `kind` is not 'pie' or 'donut'.
        ValueError: If more than 12 categories are provided.
        ValueError: If `scale` is not between 1 and 9.

    Notes:
        - This function uses `to_series` to convert DataFrame or other data types into a Series.
        - It supports rich annotations and color palettes for better visual clarity.
        - Maximum of 12 categories allowed for readability.

    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'B', 'A', 'A', 'B', 'C'])
        >>> fig, ax = plt_pie3(data, kind='donut', label_place='aside', title='Distribution of Categories')
        >>> plt.show()
    """
    # Get the data to graph: use controls and processing that get_fdt() does to obtain the series to graph (first column: 'Frequency')
    fdt = get_fdt(data, value_counts=value_counts, order=order,
                  dropna=False, na_position='value', na_aside_calc=False, include_flat_relatives=False, include_pcts=False)
        # - dropna=False            -> So that it doesn't remove NaNs, and then handle them
        # - na_aside_calc=False     -> So that it allows me to sort the nan value with na_position='value'
        # - na_position='value'     -> So that it allows me to sort the nan value within the list of values either desc or asc (according to order)
        # - We do not include relative flat rates or percentages. We calculate the percentages before presenting them.

    cat_name = fdt.index.name                       # Category name <- from fdt.index.name (could be 'Index' Warn!)

    sr = fdt.iloc[:, 0]                             # Get the Series with the frequencies (count)
    # As sr.index build the legends: If I want to change the legends, I'll have to see how I modify this sr.index (must be done previously aside)
         
    # Handling of nans since they are presented in the subtitle, whether or not they appear in the graph
    total_label = "Total (w/ nulls)"                # Default total_label to be presented in subtitle
    
    if pd.isna(sr.index).any():                     # There is np.nan [NaN] index, nans values
        n_nans = sr[np.nan]
        if dropna:                                  # No NaNs in the graph
            sr = sr.drop(np.nan, errors='ignore')   # Drop NaN row from the DataFrame
            total_label = "Total (wo/ nulls)"       # The total will be calculated wo/NaNs (likewise, n_nans will appear in the subtitle.)
    else:                                           # No np.nan row
        n_nans = 0

    # Validate kind parameter
    if kind.lower() not in ['pie', 'donut']:
        raise ValueError(f"Invalid 'kind' parameter: '{kind}'. Must be 'pie' or 'donut'.")
    
    # Validate maximum categories
    if len(sr) > 12:
        raise ValueError(f"Data contains {len(sr)} categories. "
                        "Maximum allowed is 12 categories.")
    
    # Build graphs size, and fonts size from scale, and validate scale from 1 to 9.
    if scale < 1 or scale > 9:
        raise ValueError(f"[ERROR] Invalid 'scale' value. Must between '1' and '9', not '{scale}'.")
    else:
        scale = round(scale)

    # Calculate figure dimensions
    if figsize is None:
        multiplier = scale + 7.5
        w_base, h_base = 1, 0.56
        width, height = w_base * multiplier, h_base * multiplier
        figsize = (width, height)
    else:
        width, height = figsize
    
    # Calculate font sizes based on figure width
    label_size = width * 1.25
    title_size = width * 1.57

    # Base fig definitions
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

    # Configure wedge properties for donut or pie chart
    wedgeprops = {}
    if kind.lower() == 'donut':
        wedgeprops = {'width': 0.54, 'edgecolor': 'white', 'linewidth': 1}
    else:
        wedgeprops = {'edgecolor': 'white', 'linewidth': 0.5}

    # Define colors
    color_palette = get_color_list(palette, len(sr))

    if label_place == 'ext':

        wedges, texts = ax.pie(sr, wedgeprops=wedgeprops, colors=color_palette, startangle=startangle)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        # Build the labels. Annotations and legend in same label (External)
        labels = [
            f"{sr.loc[sr == value].index[0]}\n{value}\n({round(value / sr.sum() * 100, pct_decimals)} %)"
            for value in sr.values
        ]
        
        # Draw the annotations (labels)
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, fontsize=label_size, **kw)
            
    elif label_place == 'int' or label_place == 'aside':
        label_size = label_size * 0.8
        legend_size = label_size * 1.1
        
        # Set autopct and legends, different for 'int' and 'aside' label_place
        if label_place == 'int':
            # autopct for internal annotations. A funtion to show both: absolute an pcts.
            format_string = f'%.{pct_decimals}f%%'

            def _make_autopct(values, fmt_str):     # A python Closuer
                value_iterator = iter(values)
                
                def my_autopct(pct):
                    absolute_value = next(value_iterator)
                    percentage_string = fmt_str % pct
                    return f"{absolute_value}\n({percentage_string})"  
                
                return my_autopct
            
            autopct_function = _make_autopct(sr.values, format_string)

            legends = sr.index

        else:                           # elif aside:  Valid autopct and legends in case of 'aside' label_place
            autopct_function = None     # No data inside de pie or donut
            # Custom legends w/labels values and pct aside of the pie or donut
            total = sr.values.sum()         
            legends = [f"{sr.index[i]} \n| {value} | {round(value / total * 100, pct_decimals)} %"
                    for i, value in enumerate(sr.values)] 

        ax.pie(x=sr,
            colors=color_palette,
            startangle=startangle,
            autopct=autopct_function,
            wedgeprops=wedgeprops,
            textprops={'size': label_size,
                        'color': 'w',
                        'rotation': label_rotate,
                        'weight': 'bold'})
        
        ax.legend(legends,
                loc=legend_loc,
                bbox_to_anchor=(1, 0, 0.2, 1),
                prop={'size': legend_size})

    else:
        raise ValueError(f"Invalid labe_place parameter. Must be 'ext', 'int' or 'aside', not '{label_place}'.")
            
    # Build title and set title
    if not title:
        title = f"Pie/Donut Chart ({cat_name} - {sr.name})"
    ax.set_title(title, fontsize=title_size, fontweight='bold')

    if show_stats_subtitle:                     # Enhanced subtitle with statistics
        total_items = sr.sum()                  # Total items in the series
        n_categories = len(sr)                  # len(categories)
        top_2_pct = (sr.head(2).sum() / total_items * 100) if n_categories >= 3 else (sr.sum() / total_items * 100)

        subtitle = f"{total_label} {total_items:,} | Categories: {n_categories} | Top 2: {top_2_pct:.1f}% | Nulls (nan): {n_nans}"
        ax.text(0, 1.18, subtitle, ha='center', va='center', fontsize=title_size * 0.6, color='dimgray')

    return fig, ax


def plt_pareto(
    data: Union[pd.Series, pd.DataFrame],
    value_counts: Optional[bool] = False,
    scale: Optional[int] = 2,
    title: Optional[str] = 'Pareto Chart',
    x_label: Optional[str] = None,
    y1_label: Optional[str] = None,
    y2_label: Optional[str] = None,
    palette: Optional[list] = None,
    color1: Optional[str] = 'midnightblue',
    color2: Optional[str] = 'darkorange',
    pct_decimals: Optional[int] = 1,
    label_rotate: Optional[float] = 45,
    figsize: Optional[tuple] = None,
    fig_margin: Optional[float] = 1.1,
    show_grid: Optional[bool] = True,
    bars_alpha: Optional[float] = 0.8,
    reference_pct: Optional[float] = 80,
    reference_linewidth: float = 1,
    reference_color: str = 'red',
    reference_alpha: Optional[float] = 0.6,
    show_reference_lines: bool = True,
    scaled_cumulative: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Generates a Pareto chart with frequency bars and cumulative percentage line.

    This function creates a dual-axis chart showing category frequencies as bars
    and their cumulative percentage as a line. It supports custom styling, scaling,
    and automatic formatting of data.

    Parameters:
        data (Union[pd.Series, pd.DataFrame]): Input data. If DataFrame, it will be
            converted to a Series using `to_series`.
        value_counts (bool, optional): Whether to treat the input as raw categories
            and count frequencies. Default is False.
        scale (int, optional): Chart scaling factor (1 to 9). Affects figure size and font sizes.
            Default is 2.
        title (str, optional): Chart title. Default is 'Pareto Chart'.
        x_label (str, optional): Label for the x-axis. Default is the index name of the data.
        y1_label (str, optional): Label for the primary y-axis (frequencies). Default is the first column name.
        y2_label (str, optional): Label for the secondary y-axis (cumulative percentages).
            Default is the last column name.
        palette (list, optional): List of color names or hex codes for bar colors.
            Overrides `color1` if provided.
        color1 (str, optional): Color for the bars and primary y-axis labels. Default is 'midnightblue'.
        color2 (str, optional): Color for the cumulative percentage line and secondary y-axis labels.
            Default is 'darkorange'.
        pct_decimals (int, optional): Number of decimal places to display in percentage labels.
            Default is 1.
        label_rotate (float, optional): Rotation angle for x-axis labels. Default is 45.
        figsize (tuple, optional): Width and height of the figure in inches. If not provided,
            it is calculated based on `scale`.
        fig_margin (float, optional): Margin multiplier for y-axis limits. Default is 1.1.
        show_grid (bool, optional): Whether to show grid lines. Default is True.
        bars_alpha (float, optional): Transparency level for bars. Default is 0.8.
        reference_pct (float, optional): Reference percentage line to draw on the chart.
            Must be between 0 and 100. Default is 80.
        reference_linewidth (float, optional): Width of the reference line. Default is 1.
        reference_color (str, optional): Color of the reference line. Default is 'red'.
        reference_alpha (float, optional): Transparency of the reference line. Default is 0.6.
        show_reference_lines (bool, optional): Whether to show the reference percentage line.
            Default is True.
        scaled_cumulative (bool, optional): Whether to scale the cumulative line to match the bar axis.
            If False, uses a separate percentage axis. Default is False.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]: A tuple containing:
            - fig: The Matplotlib Figure object.
            - (ax, ax2): Primary and secondary Axes objects for further customization.

    Raises:
        TypeError: If input data is not a pandas Series or DataFrame.
        ValueError: If scale is not between 1 and 9 or reference_pct is invalid.

    Notes:
        - This function uses `get_fdt` to compute frequency distribution tables.
        - It supports rich annotations, custom palettes, and reference lines for better insights.
        - The chart includes a subtitle with summary statistics: total items, number of categories,
          top 3 contribution, and null count.

    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'B', 'A', 'A', 'B', 'C'])
        >>> fig, (ax, ax2) = plt_pareto(data, title='Product Defects Distribution')
        >>> plt.show()
    """

    # Convert to serie en case of DF
    if isinstance(data, pd.DataFrame):
        data = to_series(data)

    # Validate data parameter a pandas object
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"Input data must be a pandas Series or DataFrame. Got {type(data)} instead."
        )
    
    # Validate and process scale parameter
    if not (1 <= scale <= 9):
        raise ValueError(f"Invalid 'scale' value. Must be between 1 and 9, got {scale}.")
    
    scale = round(scale)
    
    # Validate reference percentage
    if reference_pct is not None and not (0 < reference_pct <= 100):
        raise ValueError(f"reference_pct must be between 0 and 100, got {reference_pct}")
    
    # Validate reference linewidth
    if reference_linewidth < 0:
        raise ValueError(f"reference_linewidth must be non-negative, got {reference_linewidth}")

    # Before getting the Frequency Distribution Table get the nulls
    nulls = data.isna().sum()

    # Get de fdt. categories=fdt.index; frequencies=fdt.iloc[:, 0]; relative_pcts=fdt.iloc[:, -2]; cumulative_pcts=fdt.iloc[:, -1]
    fdt = get_fdt(data, value_counts=value_counts, plain_relatives=False)

    # Calculate figure dimensions
    if figsize is None:
        multiplier = 1.33333334 ** scale
        w_base, h_base = 4.45, 2.25
        width, height = w_base * multiplier, h_base * multiplier
        figsize = (width, height)
    else:
        width, height = figsize
    
    # Calculate font sizes based on figure width
    bar_label_size = width
    axis_label_size = width * 1.25
    title_size = width * 1.57

    # Calculate cumulative_line sizes
    markersize = width * 0.3
    linewidth = width * 0.1

    # Set up colors
    if palette:
        color_palette = get_color_list(palette, fdt.shape[0])
        color1 = color_palette[0]                                   # In this case don't consider color1 parameter
    else:
        color_palette = color1

    # Create figure and primary axis
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    
    # Create bar plot
    bars = ax.bar(fdt.index, fdt.iloc[:, 0], 
                  color=color_palette,
                  width=0.95, 
                  alpha=bars_alpha,
                  edgecolor='white', 
                  linewidth=0.5)

    # Add value labels on bars
    labels = [f"[{fdt.iloc[ix, 0]}]  {fdt.iloc[ix, -2]:.1f} %" for ix in range(fdt.shape[0])]
    ax.bar_label(bars,
                labels=labels,
                fontsize=bar_label_size * 0.9,
                fontweight='bold',
                color=color1,
                label_type='edge',  # Etiqueta fuera de la barra
                padding=2)          #, rotation=90)  # opcional

    # Create secondary y-axis for cumulative percentage
    ax2 = ax.twinx()        # create another y-axis sharing a common x-axis
    
    # Calculate cumulative values
    cumulative_percentages = fdt.iloc[:, -1]            # Last column: ['Cumulative Freq. [%]']
    
    if scaled_cumulative:                               # Scaling mode fixed
        total_sum = fdt.iloc[:, 0].sum()
        
        # Convert cumulative percentages to scaled heightsdas
        scaled_values = (cumulative_percentages / 100) * total_sum
        
        # Draw the scaled line on the main axis (x=index, y=scaled_values)
        line = ax.plot(fdt.index, scaled_values,
                       color=color2,
                       marker="D",
                       markersize=markersize,
                       linewidth=linewidth,
                       markeredgecolor='white',
                       markeredgewidth=0.2)
        
        # Adjust main axis limits to include the line
        max_freq = fdt.iloc[:, 0].max()
        max_scaled = scaled_values.max()
        # Use the maximum between the bars and the scaled line, with margin
        ax.set_ylim(0, max(max_freq, max_scaled) * fig_margin)
        
        # CORRECCIÓN: Configurar ax2 para que coincida con la escala del eje principal
        ax2.set_ylim(0, max(max_freq, max_scaled) * fig_margin)
        
        # Create custom stickers for ax2 that show percentages, corresponding to the climbed heights
        ax2_ticks = []
        ax2_labels = []
        for pct in [0, 20, 40, 60, 80, 100]:
            scaled_tick = (pct / 100) * total_sum
            if scaled_tick <= max(max_freq, max_scaled) * fig_margin:
                ax2_ticks.append(scaled_tick)
                ax2_labels.append(f'{pct}%')
        
        ax2.set_yticks(ax2_ticks)
        ax2.set_yticklabels(ax2_labels)
        
        # % point labels
        formatted_weights = [f'{x:.{pct_decimals}f}%' for x in cumulative_percentages]
        for i, txt in enumerate(formatted_weights):
            if i == 0:              # To change only % annotate of the first bar         
                distance = 0.08     # first % annotate, away from the bar
            else:
                distance = 0.025    # The others % annotates, not so far
            ax.annotate(txt,
                       (fdt.index[i], scaled_values.iloc[i] + (max(max_freq, max_scaled) * distance)),
                       color=color2,
                       fontsize=bar_label_size,
                       ha='center')
        
        # Reference lines in scaled mode
        if show_reference_lines and reference_pct is not None:
            reference_scaled_height = (reference_pct / 100) * total_sum
            
            # AXHLINE and its text
            ax.axhline(y=reference_scaled_height, color=reference_color, linestyle='--', 
                      alpha=reference_alpha, linewidth=reference_linewidth)
            
            ax.text(0.01, reference_scaled_height + (max(max_freq, max_scaled) * 0.02), 
                   f'{reference_pct}%', 
                   transform=ax.get_yaxis_transform(), 
                   color=reference_color, fontsize=bar_label_size*0.8)
    
    else:                                           # Native scaling
        ax2.set_ylim(0, 100 * fig_margin)
        
        line = ax2.plot(fdt.index, cumulative_percentages,
                        color=color2,
                        marker="D",
                        markersize=markersize,
                        linewidth=linewidth,
                        markeredgecolor='white',
                        markeredgewidth=0.2)
        
        ax2.yaxis.set_major_formatter(PercentFormatter())

        formatted_weights = [f'{x:.{pct_decimals}f}%' for x in cumulative_percentages]  
        for i, txt in enumerate(formatted_weights):
                ax2.annotate(txt,
                            (fdt.index[i], cumulative_percentages.iloc[i] - 6),
                            color=color2,
                            fontsize=bar_label_size,
                            ha='center')
        
        if show_reference_lines and reference_pct is not None:
            ax2.axhline(y=reference_pct, color=reference_color, linestyle='--', 
                       alpha=reference_alpha, linewidth=reference_linewidth)
            
            ax2.text(0.01, reference_pct + 3, f'{reference_pct}%', 
                        transform=ax2.get_yaxis_transform(), 
                        color=reference_color, fontsize=bar_label_size*0.8)

    # Configure tick parameters
    ax.tick_params(axis='y', colors=color1, labelsize=bar_label_size)
    ax.tick_params(axis='x', rotation=label_rotate, labelsize=bar_label_size)
    ax2.tick_params(axis='y', colors=color2, labelsize=bar_label_size)

    # Set y-axis limits (solo para modo original)
    if not scaled_cumulative:
        max_freq = fdt.iloc[:, 0].max()
        ax.set_ylim(0, max_freq * fig_margin)

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    # Set title and labels
    if not x_label:
         x_label = fdt.index.name
    
    if not y1_label:
         y1_label = fdt.columns[0]

    if not y2_label:
         y2_label = fdt.columns[-1]

    # Enhanced subtitle with statistics
    total_items = fdt.iloc[:, 0].sum()      # frequencies.sum()
    n_categories = len(fdt.index)           # len(categories)
    top_3_pct = cumulative_percentages.iloc[min(2, len(cumulative_percentages)-1)]      # if len(cum_pcts) < 2
    subtitle = f"Total: {total_items:,} | Categories: {n_categories} | Top 3: {top_3_pct:.1f}% | Nulls: {nulls}"

    # Apply title and labels
    fig.suptitle(title, fontsize=title_size, fontweight='bold')
    ax.set_title(subtitle, fontsize=axis_label_size*0.8, color=color1, pad=10)
    ax.set_xlabel(x_label, fontsize=axis_label_size, fontweight='medium')
    ax.set_ylabel(y1_label, fontsize=axis_label_size, color=color1, fontweight='medium')
    ax2.set_ylabel(y2_label, fontsize=axis_label_size, color=color2, fontweight='medium')

    return fig, (ax, ax2)






if __name__ == "__main__":

    # Data
    dic = {'1603 SW': [21, 'No POE'], '1608 SW': [6, 'Headset compatible'], 
       '1616 SW': [3, 'Telefonista'], '9611 G': [8, 'Gerencial Gigabit']}
    df = pd.DataFrame.from_dict(dic, orient='index', columns=['Stock', 'Obs'])

    # Show palettes
    show_plt_palettes()



