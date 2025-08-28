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
# import matplotlib.colors as mcolors             # for get_color_list()
from matplotlib.ticker import PercentFormatter  # for pareto chart() and... ?
# from matplotlib import colormaps                # for show_matplotlib_palettes()
# import seaborn as sns

# Local Libs
from jm_utils.data.pd_functions import to_series, get_fdt
from jm_utils.data.mpl_tints import CMAP_NAMES_BY_CAT, get_hexcolor_list_from_pltt

## Custom types for non-included typing annotations
IndexElement: TypeAlias = Union[str, int, float, pd.Timestamp]
# IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', np.str_, np.int64, np.float64, np.datetime64, pd.Timestamp, ...]



def ax_plt_pie(
    ax: plt.Axes,
    data: pd.Series | np.ndarray | dict | list | set | pd.DataFrame,
    value_counts: Optional[bool] = False,
    dropna: Optional[bool] = True,
    order: Optional[str] = 'desc',
    scale: Optional[int] = 3,
    title: Optional[str | None] = None,
    kind: Optional[str] = 'pie',
    palette: Optional[list] = 'Blues_r',
    startangle: Optional[float] = 90,
    pcts_labels: Optional[str | None] = 'inside',
    pct_decimals: Optional[int] = 1,
    labels_rotatation: Optional[float] = 0,
    labels_color: Optional[str] = 'black',
    legends: Optional[str | None] = None,
    legends_title: Optional[str] = None,
    show_stats_subtitle: Optional[bool]= True,
    footer: Optional[str | None] = "Pie/Donut Chart from jm_utils.data.plt_charts",
) -> plt.Axes:
    """
    Plots a pie or donut chart on a given matplotlib Axes with advanced customization options.

    This function is designed for internal use or integration into subplot layouts. It draws
    a pie or donut chart on a pre-existing Axes object, offering fine control over labels,
    colors, and styling.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object on which to draw the chart.

        data (pd.Series, np.ndarray, dict, list, set, pd.DataFrame): Input data.
            Will be processed using `get_fdt` to generate frequency counts.

        value_counts (bool, optional): If True, treats input data as raw categories and
            computes frequency counts. Default is False.

        dropna (bool, optional): If True, excludes NaN values from the chart.
            If False, includes NaN as a category. Default is True.

        order (str, optional): Sorting order for categories:
            - 'desc': Descending by frequency.
            - 'asc': Ascending by frequency.
            - 'ix_asc': Ascending by index.
            - 'ix_desc': Descending by index.
            - None: No sorting.
            Default is 'desc'.

        scale (int, optional): Scaling factor (1 to 16) affecting font sizes.
            Larger values produce larger text. Default is 1.

        title (str or None, optional): Chart title. If None, a default title is generated.
            Default is None.

        kind (str, optional): Type of chart:
            - 'pie': Standard pie chart.
            - 'donut': Donut chart with a hole in the center.
            Default is 'pie'.

        palette (list or str, optional): Color palette to use.
            - If str: Name of a Matplotlib colormap (e.g., 'viridis', 'Blues_r') or a string
              of single-character color codes ('bgrcmykw').
            - If list: A list of color names or hex codes.
            - If None: Uses the default Matplotlib color cycle.
            Default is 'Blues_r'.

        startangle (float, optional): Starting angle in degrees for the first wedge.
            Default is 90 (top, vertical).

        pcts_labels (str or None, optional): Controls the display of labels and percentages:
            - None: No labels or percentages inside or outside the chart.
            - 'inside': Category names, values, and percentages inside segments.
            - 'invalues': Only values and percentages inside segments.
            - 'mixed': Category names as labels, values and percentages inside segments.
            - 'outside': Category names, values, and percentages connected by arrows outside segments.
            Default is 'inside'.

        pct_decimals (int, optional): Number of decimal places for percentage labels.
            Default is 1.

        labels_rotatation (float, optional): Rotation angle for internal text labels.
            Default is 0.

        labels_color (str, optional): Color for internal text labels.
            Default is 'black'.

        legends (str or None, optional): Controls the display of a legend:
            - None: No legend.
            - 'base': Legend with category names only.
            - 'full': Legend with category names, values, and percentages.
            Default is None.

        legends_title (str, optional): Title for the legend.
            Default is None.

        show_stats_subtitle (bool, optional): If True, adds a subtitle with summary statistics:
            total count, number of categories, contribution of top 2, and null count.
            Default is True.

        footer (str or None, optional): Text to display as a footer below the chart.
            If None, no footer is shown. Default is a standard jm_utils footer.

    Returns:
        matplotlib.axes.Axes: The modified Axes object with the pie/donut chart drawn.

    Raises:
        ValueError: If `scale` is not between 1 and 16.
        ValueError: If `kind` is not 'pie' or 'donut'.
        ValueError: If `pcts_labels` or `legends` receive invalid string values.
        ValueError: If more than 9 categories are provided in the data.

    Notes:
        - This function uses `get_fdt` internally to process the input data.
        - NaN handling is flexible: can be included, excluded, or sorted with values.
        - Designed for reuse and integration into larger figures or dashboards.
        - Supports a wide range of color customization options.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax_plt_pie6(ax, data, kind='donut', pcts_labels='mixed', legends='full')
        >>> plt.show()
    """

    ## ----------------------- Data Preparation ------------------------------------------------------------------------------------------------------------------ 
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
    total_label = "Total (w/nulls)"                 # total_label to be presented in subtitle (inital value w/nulls)
    
    if pd.isna(sr.index).any():                     # There is np.nan [NaN] index, nans values
        n_nans = sr[np.nan]
        if dropna:                                  # No NaNs in the graph
            sr = sr.drop(np.nan, errors='ignore')   # Drop NaN row from the DataFrame
            total_label = "Total (wo/nulls)"        # The total will be calculated wo/NaNs (likewise, n_nans will appear in the subtitle.)
    else:                                           # No np.nan row
        n_nans = 0

    total = sr.sum()                                # Get the total sum() of frequency (count) of all categories to display

    # Validate maximum categories
    if len(sr) > 9:
        raise ValueError(f"Data contains {len(sr)} categories. Maximum allowed is 9 categories.")
    
    ## ----------------------- Set sizes (based on scale), wedgeprops (based on kind) and colors (based on palette) ------------------------------------------------- 
    # Build graphs size, and fonts size from scale, and validate scale from 1 to 9.
    if scale < 1 or scale > 16:
        raise ValueError(f"Invalid value for 'scale': {repr(scale)}. Expected a value between 1 and 16 inclusive.")
    else:
        scale = round(scale)
    
    # Calculate font sizes based scale
    multiplier= scale + 5
    labels_size = multiplier * 1.1
    title_size = multiplier * 1.6

    # Configure wedge properties for 'pie' or for 'donut' chart
    if kind.lower() == 'pie':
        wedgeprops = {'edgecolor': 'white', 'linewidth': 1}
    elif kind.lower() == 'donut':
        wedgeprops = {'width': 0.55, 'edgecolor': 'white', 'linewidth': 1}
    else:
        ValueError(f"Invalid value for 'kind': {repr(kind)}. Expected one of: 'pie', 'donut'.")

    # Define colors: str -> palette name or 'one_char_color_str'; lst -> list of user colors
    if isinstance(palette, str):                            
        pltts_by_cat = CMAP_NAMES_BY_CAT.copy()      # Dict of full known colormaps 
        all_cmaps_lst = [cmap for key in pltts_by_cat.keys() for cmap in pltts_by_cat[key][0]]
        if palette in all_cmaps_lst:                        # If the str is a colormap
            not_qualitative_cmaps = list(filter(lambda pltt: pltt not in pltts_by_cat['Qualitative'][0], all_cmaps_lst))
            if palette in not_qualitative_cmaps:            # Different treatment for qualitative palettes and custom palettes (avoid ends)
                color_list = get_hexcolor_list_from_pltt(palette, len(sr) + 2)[1:]
            else:                                           # The case of Qualitative or custom user-defined colormaps
                color_list = get_hexcolor_list_from_pltt(palette, len(sr))
        elif any(char in 'bgrcmykw' for char in palette):   # One-char-color_names as string instead of a list
            color_list = [char for char in palette if char in 'bgrcmykw']
        else:
            color_list = None                               # Default system colors (colors in the currently active cycle.)
    elif isinstance(palette, list):
        color_list = palette                                # User-defined color list
    else:
        color_list = None                                   # Default system colors (colors in the currently active cycle.)

    ## ----------------------- Build the chart (pie or donut) ------------------------------------------------- 
    # _make_autopct() auxiliar function for different autopcts
    def _make_autopct(values, inside_lanels):             # A python Closure
        value_iterator = iter(values)    
        def my_autopct(pct):
            next_value = next(value_iterator)
            info = f"{next_value:,}\n{pct:.{pct_decimals}f}%"
            if inside_lanels is True:
                info = f"{sr.loc[sr == next_value].index[0]}\n" + info
            return info

        return my_autopct
    
    # Set different values of labels and autopct acoording pcts_labels parameter (special case 'outside')
    if pcts_labels is None:
        labels = None
        autopct_func = None
    elif pcts_labels == 'inside':
        labels = None
        autopct_func = _make_autopct(sr.values, inside_lanels=True)
    elif pcts_labels == 'invalues':
        labels = None
        autopct_func = _make_autopct(sr.values, inside_lanels=False)
    elif pcts_labels == 'mixed':
        labels = sr.index
        autopct_func = _make_autopct(sr.values, inside_lanels=False)
    elif pcts_labels == 'outside':
        # Special case where we build the chart here
        wedges, _ = ax.pie(sr, wedgeprops=wedgeprops, colors=color_list, startangle=startangle)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        # Build the labels. Annotations and legend in same label (External)
        labels = [f"{sr.loc[sr == value].index[0]}\n{value:,}\n({round(value / total * 100, pct_decimals)} %)"
                  for value in sr.values]

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.1*np.sign(x), 1.1*y),
                    horizontalalignment=horizontalalignment, fontsize=labels_size, **kw)
    else:
        raise ValueError(f"Invalid value for 'pcts_labels': {repr(pcts_labels)}. Expected one of: None, 'inside', 'invalues', 'mixed', 'outside'.")

    # Build the chart for the rest of pcts_labels options different from 'outside'. (None, 'inside', 'invalues', 'mixed')
    #   (It is more direct to run it with an if and not create another internal aux function because the parameterization is very heavy)
    if pcts_labels != 'outside':
        ax.pie(x = sr,
            wedgeprops = wedgeprops,
            labels = labels,
            rotatelabels = True,
            autopct = autopct_func,
            colors = color_list,
            startangle = startangle,
            textprops = dict(size = labels_size,
                             color = labels_color,
                             rotation = labels_rotatation,
                             weight = 'semibold'),)

    # Display legends based on the 'legends' parameter 
    if legends is not None:
        if legends == 'base':
            color_legends = sr.index
        elif legends == 'full':
            color_legends = [f"{sr.index[i]} \n {value:,} | {round(value / total * 100, pct_decimals)} %"
                           for i, value in enumerate(sr.values)]
        else:
            raise ValueError(f"Invalid value for 'legends': {repr(legends)}. Expected one of: None, 'base', 'full'.")

        ax.legend(color_legends,
                  loc='best',
                  prop={'size': labels_size},
                  title=legends_title,
                  title_fontproperties = {'size':labels_size, 'weight': 'bold'},
                  bbox_to_anchor=(1, 0.9),
        )
            
    # Buid and display title, subtitle, and footer
    if not (isinstance(title, str)):
        title = f"Pie/Donut Chart ({cat_name})"
    ax.set_title(title, fontsize=title_size, fontweight='bold', loc='left')

    if show_stats_subtitle:                              # Enhanced footer with statistics
        n_categories = len(sr)                  # len(categories)
        top_2_pct = (sr.head(2).sum() / total * 100) if n_categories >= 2 else 100

        stats = f"{total_label} {total:,} | Categories: {n_categories} | First 2: {top_2_pct:.1f}% | Nulls (nan): {n_nans}"
        ax.text(0, 1, stats, transform=ax.transAxes, fontsize=title_size * 0.65, ha='left', va='top', color='dimgray')

    if footer and isinstance(footer, str):
        ax.text(0, 0.01, footer, 
                transform=ax.transAxes,             # Coordinates relative to the axes (0-1)
                fontsize=title_size * 0.65,
                ha='left',                      
                va='bottom',                    
                style='italic',
                color='dimgray')

    return ax


def plt_pie(
    data: pd.Series | np.ndarray | dict | list | set | pd.DataFrame,
    value_counts: Optional[bool] = False,
    dropna: Optional[bool] = True,
    order: Optional[str] = 'desc',
    scale: Optional[int] = 3,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    kind: Optional[str] = 'pie',
    palette: Optional[list | None] = 'Blues_r',
    startangle: Optional[float] = 90,
    pcts_labels: Optional[str] = 'inside',
    pct_decimals: Optional[int] = 1,
    labels_rotatation: Optional[float] = 0,
    labels_color: Optional[str] = 'black',
    legends: Optional[str | None] = None,
    legends_title: Optional[str] = None,
    show_stats_subtitle: Optional[bool]= True,
    footer: Optional[str | None] = "Pie/Donut Chart from jm_utils.data.plt_charts",
    canvas_color: Optional[str | None] = 'whitesmoke'
) -> tuple[plt.Figure, plt.Axes]:
    """
    Creates a standalone pie or donut chart figure with extensive customization options.

    This high-level function generates a complete pie or donut chart figure. It is ideal for
    standalone visualizations and quick data exploration. Internally, it uses `ax_plt_pie6`
    to draw the chart on a newly created Axes.

    Parameters:
        data (pd.Series, np.ndarray, dict, list, set, pd.DataFrame): Input data.
            If not a Series, it will be converted.

        value_counts (bool, optional): If True, treats input data as raw categories and
            computes frequency counts. Default is False.

        dropna (bool, optional): If True, excludes NaN values from the chart.
            If False, includes NaN as a category. Default is True.

        order (str, optional): Sorting order for categories (see `ax_plt_pie6`).
            Default is 'desc'.

        scale (int, optional): Scaling factor (1 to 16) affecting figure and font sizes.
            Default is 1.

        figsize (tuple[float, float], optional): Width and height of the figure in inches.
            If not provided, size is calculated from `scale`.

        title (str, optional): Chart title. If not provided, a default title is generated.

        kind (str, optional): Type of chart: 'pie' or 'donut'. Default is 'pie'.

        palette (list or str or None, optional): Color palette to use (see `ax_plt_pie6`).
            Default is 'Blues_r'.

        startangle (float, optional): Starting angle for the first wedge. Default is 90.

        pcts_labels (str or None, optional): Label and percentage display style (see `ax_plt_pie6`).
            Default is 'inside'.

        pct_decimals (int, optional): Decimal places in percentage labels. Default is 1.

        labels_rotatation (float, optional): Rotation for internal labels. Default is 0.

        labels_color (str, optional): Color for internal text. Default is 'black'.

        legends (str or None, optional): Legend display option (see `ax_plt_pie6`).
            Default is None.

        legends_title (str, optional): Title for the legend. Default is None.

        show_stats_subtitle (bool, optional): If True, displays a subtitle with key statistics.
            Default is True.

        footer (str or None, optional): Text to display as a footer. If None, no footer.
            Default is a standard jm_utils footer.

        canvas_color (str or None, optional): Background color of the figure.
            If None, uses the default Matplotlib background. Default is 'whitesmoke'.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing:
            - fig: The matplotlib Figure object.
            - ax: The Axes object with the chart.

    Raises:
        ValueError: If `scale` is not between 1 and 16.
        ValueError: If `kind`, `pct_labels`, or `legends` are invalid (delegated to `ax_plt_pie6`).

    Notes:
        - This function is a wrapper around `ax_plt_pie6`, providing a simple interface
          for creating full figures.
        - Automatically calls `plt.show()` to display the chart.
        - Ideal for interactive use, reports, and exploratory data analysis.

    Example:
        >>> fig, ax = plt_pie6(data, kind='donut', title='Distribution of Categories')
        >>> plt.show()
    """
    # Build graphs size, and fonts size from scale, and validate scale.
    if scale < 1 or scale > 16:
        raise ValueError(f"Invalid value for 'scale': {repr(scale)}. Expected a value between 1 and 16 inclusive.")
    else:
        scale = round(scale)

    # Calculate figure dimensions
    if figsize is None:
        multiplier = scale + 5
        w_base, h_base = 1.25, 0.7
        width, height = w_base * multiplier, h_base * multiplier
        figsize = (width, height)
    else:
        width, height = figsize
        scale = (width + height) / 2.5

    # Base fig definitions
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"), facecolor=(canvas_color))

    ax_plt_pie(ax, data, value_counts, dropna, order, scale, title, kind, palette, startangle, pcts_labels,
                pct_decimals, labels_rotatation, labels_color, legends, legends_title, show_stats_subtitle, footer)    
                
    plt.show()
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
        color_palette = get_hexcolor_list_from_pltt(palette, fdt.shape[0])
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
    fig = plt_pie(df['Stock'], scale=4)



