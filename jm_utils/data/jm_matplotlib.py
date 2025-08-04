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
from typing import Union, Optional, Any, Literal, Sequence, TypeAlias
import random

# Third-Party Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter  # for pareto chart and ?
import seaborn as sns

# Local Libs
from jm_utils.data.jm_pandas import to_series, get_fdt

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


def get_colorblind_color_list() -> list[str]:
    """
    Retorna una lista de colores (hexadecimales) amigables para personas
    con daltonismo, equivalentes a sns.color_palette('colorblind').
    """
    return [
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
        '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9',
        '#5D8C3B', '#A93967', '#888888', '#FFC107', '#7C9680',
        '#E377C2', '#BCBD22', '#AEC7E8', '#FFBB78', '#98DF8A',
        '#FF9896', '#C5B0D5', '#C49C94', '#F7B6D2', '#DBDB8D',
        '#9EDAE5', '#D68E3A', '#A65898', '#B2707D', '#8E6C87'
    ]


def get_color_list(palette: str, n_items: Optional[int] = 10) -> list[str] | list[tuple[float, float, float, float]]:
    """
    | Return a valid matplotlib palette list    | 'colorblind' is a kind of sns.colorblind 
    - Qualitatives (Cat) = ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Dark2', 'Paired', 'Accent', 'colorblind']
    - Sequential (Order) = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    - Diverging (MidPoint) = ['coolwarm', 'bwr', 'seismic', 'PiYG', 'PRGn', 'BrBG', 'RdGy', 'RdBu', 'Spectral', 'RdYlGn', 'PuOr', 'RdYlBu']
    - Cyclic (Repeat)= ['twilight', 'twilight_shifted', 'hsv', 'turbo', 'cubehelix', 'gist_rainbow', 'jet', 'nipy_spectral', 'rainbow_r']
    - Mix = ['rainbow', 'flag', 'prism', 'ocean', 'terrain', 'gnuplot', 'CMRmap', 'hot', 'afmhot', 'gist_heat', 'copper', 'bone', 'pink']
    """
    if palette == 'colorblind':
        color_list = get_colorblind_color_list()
    else:
        cmap = plt.get_cmap(palette, n_items)             # Use palette colormap
        color_list = [cmap(i) for i in range(n_items)]    # Get colors from the colormap

    return color_list


def show_plt_palettes(
        palette_group: Union[str, list[str]] = 'Sample',
        n_items: Optional[int]=14,
) -> plt.Figure:
    """
    Displays a visual comparison of Matplotlib color palettes in a two-column layout.

    This function creates a grid of bar charts, each showing the color progression of a
    specific Matplotlib colormap. It supports built-in palette groups (Qualitative, Sequential,
    Diverging, Cyclic), a default 'Sample' view, and custom lists of palettes.

    Parameters:
        palette_group (Union[str, list[str]]): Specifies which palettes to display:
            - If str: one of 'Qualitative', 'Sequential', 'Diverging', 'Cyclic', or 'Sample'.
              Case-insensitive; will be capitalized.
            - If list: a custom list of colormap names to display.
            - Default is 'Sample', which shows a representative selection from all groups.
        n_items (int, optional): Number of color swatches (bars) to display per palette.
            Must be between 1 and 25 (inclusive). Default is 16.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing all subplots.
            This allows further customization, saving, or inspection after display.

    Raises:
        TypeError: If `palette_group` is not a string or list of strings, or if `n_items`
            is not a number.
        ValueError: If `n_items` is not in the valid range (1–25).

    Notes:
        - Invalid or deprecated colormap names are handled gracefully and labeled accordingly.
        - The layout adapts to the number of palettes, using two columns for better readability.
        - Uses `get_colors_list` internally to extract colors from each colormap.
        - Ideal for exploring and selecting appropriate color schemes for data visualization.

    Example:
        >>> show_plt_palettes('Sequential', n_items=10)
        # Displays 10-color samples for all Sequential palettes.

        >>> show_plt_palettes(['viridis', 'plasma', 'coolwarm', 'rainbow'], n_items=12)
        # Shows a custom comparison of four specific palettes.

        >>> show_plt_palettes()
        # Shows a default sample of 4 palettes from each category.
    """
    
    # Verified n_times parameter
    if not isinstance(n_items, (int, float)):
        raise TypeError(f"'n_items' parameter not valid. Must be an int or float. Got '{type(n_items)}'.")

    if n_items < 1 or n_items > 25:
        raise ValueError(f"'n_items' parameter not valid. Must be > 1 and < 26. Got '{n_items}'.")
    n_items = int(n_items) + 1

    # Palette_group selection + custom palette_group and palette_group parameter validation
    Custom = []                             # Default empty list for custom palettes
    if isinstance(palette_group, str):
        palette_group_key = palette_group.strip().capitalize()
    elif isinstance(palette_group, list):
        palette_group_key = 'Custom'
        Custom = palette_group
    else:
        raise TypeError(f"'palette_group' parameter not valid. Must be a string or a list. Got {type(palette_group)}.")
    
    # 1. Native palette Group lists
    Qualitative = ['Accent', 'colorblind', 'Dark2', 'Dark2_r', 'flag', 'Paired', 'Pastel1', 'Pastel2',
                    'prism', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']

    Sequential = ['autumn', 'binary', 'Blues', 'brg', 'BuPu', 'cividis', 'cool', 'GnBu',
              'Greens', 'Greys', 'Greys_r', 'gnuplot', 'inferno', 'magma', 'ocean', 'Oranges',
                'OrRd', 'plasma', 'PuRd', 'Purples', 'Reds', 'terrain', 'viridis', 'Wistia']

    Diverging = ['BrBG', 'bwr', 'bwr_r', 'coolwarm', 'PiYG', 'PiYG_r', 'PRGn', 'PRGn_r',
                 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic', 'Spectral', 'Spectral_r']

    Cyclic = ['berlin', 'berlin_r', 'cubehelix', 'cubehelix_r', 'flag_r', 'gist_rainbow', 'hsv', 'jet_r',
              'managua', 'nipy_spectral', 'rainbow', 'rainbow_r', 'twilight', 'twilight_shifted', 'turbo', 'vanimo']

    # 2. Get the palette group (and _desc) based on the input string (the one selected by the user)
    palette_group_dic = {
        'Qualitative': (Qualitative, 'for categorical data'),
        'Sequential': (Sequential, 'for data that has an order'),
        'Diverging': (Diverging, 'for data that have a significant midpoint'),
        'Cyclic': (Cyclic, 'for data that repeats, such as angles or phases'),
        'Custom': (Custom, 'user selected palettes'),
    }

    defaults_for_sample = ('Sample', 'a sample of four of each category')
    # Get the list of palettes for the selected group and its description
    selected_palettes, palette_group_desc = palette_group_dic.get(palette_group_key, defaults_for_sample)

    # Adjust the palette_group for the 'Sample' (four from each group)) case. Any value different from the main groups
    if palette_group_key not in ('Qualitative', 'Sequential', 'Diverging', 'Cyclic', 'Custom'):
        palette_group_key = 'Sample'                      
        palette_group = [palette for p_g in [Qualitative, Sequential, Diverging, Cyclic] for palette in random.sample(p_g, k=4)]
    else:                                                   # If not 'Sample' (all others, custom included), use the selected palettes  
        palette_group = selected_palettes                   # Assign the actual list if not 'Sample'

    # Build a Series of n_items elements to show colors
    sr = to_series({str(i): 1 for i in range(1, n_items)})

    # Create a figure with two columns for the palettes - Bar charts showing palette colors
    rows = len(palette_group) // 2 if len(palette_group) % 2 == 0 else (len(palette_group) // 2) + 1
    width = 12
    height = rows / 1.25 if rows > 6 else rows / 1.05 
    
    fig, axs = plt.subplots(rows, 2, figsize=(width, height), tight_layout=True, sharex=True)

    # Set the figure title with the palette group key and description
    fig.suptitle(f"Matplolib {palette_group_key} palettes (cmap): {palette_group_desc}", fontsize=14, fontweight='medium', y=1.001)

    if palette_group_key == 'Sample':
        fig.text(0.15, 0.95, "4 Qualitative (for categorical data), 4 Sequential (for ordered data),"
                             "4 Diverging (significant midpoint), and 4 Cyclic (for repeated data)",
                    fontsize=10, transform=fig.transFigure)

    # Iterate over the axes and palette group to plot each palette                                           
    for ax, pltt in zip(axs.flatten(), palette_group):
        try:
            color_list = get_color_list(pltt, n_items=n_items)
            ax.bar(sr.index, sr, color=color_list, width=1, edgecolor='white', linewidth=0.2)
            ax.set_xlim(-0.5, n_items - 1.5)
            ax.set_ylim(0, 0.1)
            ax.set_title(pltt, loc='left', fontsize=10, fontweight='medium')
        except ValueError:
            err_msg = f"'{pltt}' is not currently a valid Matplotlib palette (cmap)"
            ax.set_title(err_msg, loc='left', fontsize=10, fontweight='medium', color='red')

        ax.set_yticks([])       # Hide y-ticks for cleaner look
        ax.set_xticks([])       # Hide x-ticks

    plt.show()
    return fig                  # Return the current figure for further manipulation if needed


def plt_pie(
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



