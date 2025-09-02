"""
jm_pandas
"""

## TO-DO

## fdt!
# Considerar la opción de adicionar los nulls-nans-pd.NAs opcionalmente a la fdt
# EN REALIDAD el tema de los nans lo tengo que ver en el to_series_with_count()

## pie - paretto - en lo posible mismos parámetros

## OJO con la doble función de formateo de datos que tengo... OJO
# porque debería ajustar tanto esta que tengo acá com la de jm_rchprt o DEJAR SOLO UNA!!!???
# NO SE si conviene hacer dos porque en el caso de series tengo que considerar que NO es bueno mezclar n decimal con 0 decimals en una MISMA Series
# caso del cumulative relative frequency.

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

# Third-Party Libs
import numpy as np
import pandas as pd


## Custom types for non-included typing annotations
IndexElement: TypeAlias = Union[str, int, float, pd.Timestamp]
# IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', np.str_, np.int64, np.float64, np.datetime64, pd.Timestamp, ...]


# An auxiliar function to change num format - OJO se puede hacer más amplia como jm_utils.jm_rchprt.fmt...
def _fmt_value_for_pd(value, width=8, n_decimals=3, thousands_sep=',') -> str:
    """
    Format a value (numeric or string) into a right-aligned string of fixed width.

    Converts numeric values to formatted strings with thousands separators and
    specified decimal places. Strings are padded to the same width for consistent alignment.

    Parameters:
        value (int, float, str): The value to be formatted.
        width (int): Total width of the output string. Must be a positive integer.
        decimals (int): Number of decimal places for numeric values. Must be >= 0.
        miles (str or None): Thousands separator. Valid options: ',', '_', or None.

    Returns:
        str: The formatted string with right alignment.

    Raises:
        ValueError: If width <= 0, decimals < 0, or miles is invalid.

    Examples:
        >>> format_value(123456.789)
        '123,456.79'
        >>> format_value("text", width=10)
        '      text'
        >>> format_value(9876, miles=None)
        '    9876.00'
    """
    # Parameter Value validation <- vamos a tener que analizar este tema por si es un list , etc,,
    #   - En realidad acá tenemos que evaluar algo similar a jm_utils - fmt_values() FUTURE
    # if not isinstance(value, (int, float, np.integer, np.floating)) or pd.api.types.is_any_real_numeric_dtype(value)

    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"Width must be a positive integer. Not '{width}'")
    
    if not isinstance(n_decimals, int) or n_decimals < 0:
        raise ValueError(f"Decimals must be a non-negative integer. Not '{n_decimals}")
    
    if thousands_sep not in [',', '_', None]:
        raise ValueError(f"Miles must be either ',', '_', or None. Not '{thousands_sep}")
    
    try:
        num = float(value)                                          # Convert to float if possible
        if num % 1 == 0:                                            # it its a total integer number
            decimals = 0
        if thousands_sep:
            return f"{num:>{width}{thousands_sep}.{n_decimals}f}"   # Fixed width, 'x' decimal places, right aligned
        else:
            return f"{num:>{width}.{n_decimals}f}"
        
    except (ValueError, TypeError):
        return str(value).rjust(width)                              # Also align strings, to maintain the grid


def _validate_numeric_series(
        pd_data: Union[pd.Series, pd.DataFrame],
        positive: Optional[bool] = True
) -> Union[None, Exception]:

    # Validate data parameter a pandas object
    if not isinstance(pd_data, (pd.Series, pd.DataFrame)):     # pd.Series or pd.Datafram
        raise TypeError(
            f"Input data must be a pandas Series or DataFrame. Got {type(pd_data)} instead."
        )
              
    if positive:
        if not all(                                             # Only positve numeric values
            isinstance(val, (int, float, np.integer, np.floating)) and val > 0 for val in pd_data.values
        ):
            raise ValueError(f"All values in the data must be positive numeric..")
        pass
    else:                                                       # Just only numeric values
        if not all(isinstance(val, (int, float, np.integer, np.floating)) for val in pd_data.values):
            raise ValueError(f"All values in the data must be numeric values.")
        pass


def to_series(
    data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
    index: Optional[Union[pd.Index, Sequence[IndexElement]]] = None,
    name: Optional[str] = None
) -> pd.Series:
    """
    Converts input data into a pandas Series with optional custom index and name.

    This function standardizes various data types into a pandas Series. It supports
    arrays, dictionaries, lists, sets, DataFrames, and existing Series. Optionally,
    a custom index or series name can be assigned.

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame]):
            Input data to convert. Supported types:
            - pd.Series: returned as-is (can be overridden with new index/name).
            - np.ndarray: flattened and converted to a Series.
            - dict: keys become the index, values become the data.
            - list or set: converted to a Series with default integer index.
            - pd.DataFrame:
                - 1 column: converted directly to a Series.
                - 2 columns: first column becomes the index, second becomes the values.
        index (Union[pd.Index, Sequence], optional): Custom index to assign to the Series.
            If provided, overrides the original index. Default is None.
        name (str, optional): Name to assign to the Series. Default is None.

    Returns:
        pd.Series: A pandas Series constructed from the input data, with optional
            custom index and name.

    Raises:
        TypeError: If the input data type is not supported.
        ValueError: If the DataFrame has more than 2 columns.

    Examples:
        >>> import pandas as pd
        >>> to_series([1, 2, 3, 4])
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> to_series({'A': 10, 'B': 20, 'C': 30})
        A    10
        B    20
        C    30
        dtype: int64

        >>> df = pd.DataFrame({'Label': ['X', 'Y'], 'Value': [100, 200]})
        >>> to_series(df)
        Label
        X    100
        Y    200
        Name: Value, dtype: int64

        >>> to_series([10, 20, 30], index=['a', 'b', 'c'], name='Measurements')
        a    10
        b    20
        c    30
        Name: Measurements, dtype: int64
    """
    
    # Validate parameters - FUTURE
    
    if isinstance(data, pd.Series):                 # If series is already a Series no conversion needed
        series = data                                  
    elif isinstance(data, np.ndarray):              # If data is a NumPy array   
        series = pd.Series(data.flatten())
    elif isinstance(data, (dict, list)):
        series = pd.Series(data)
    elif isinstance(data, (set)):
        series = pd.Series(list(data))
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:                      # Also len(data.columns == 1)
            series = data.iloc[:, 0]
        elif data.shape[1] == 2:                    # Index: first col, Data: 2nd Col
            series = data.set_index(data.columns[0])[data.columns[1]]
        else:
            raise ValueError("DataFrame must have 1 oer 2 columns. Categories and values for 2 columns cases.")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. "
                    "Supported types: pd.Series, np.ndarray, pd.DataFrame, dict, list, set, and pd.DataFrame")

    if name:
        series.name = name

    if index:
        series.index = index

    return series

                      
def get_fdt(
        data: Union[pd.Series, np.ndarray, dict, list, set, pd.DataFrame],
        value_counts: Optional[bool] = False,
        dropna: Optional[bool] = True,
        na_position: Optional[str] = 'last',
        include_pcts: Optional[bool] = True,
        include_flat_values: Optional[bool] = True,
        fmt_values: Optional[bool] = False,
        order: Optional[str] = 'desc',
        na_aside_calc: Optional[bool] = True,
        index_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a Frequency Distribution Table (FDT) with absolute, relative, and cumulative frequencies.

    This function converts various input data types into a structured DataFrame containing:
    - Absolute frequencies
    - Cumulative frequencies
    - Relative frequencies (proportions and percentages)
    - Cumulative relative frequencies (percentages)

    Parameters:
        data (Union[pd.Series, np.ndarray, dict, list, pd.DataFrame]): Input data.
            If DataFrame, it will be converted to a Series using `to_series`.
        value_counts (bool, optional): Whether to count occurrences if input is raw data.
            Assumes data is not pre-counted. Default is False.
        dropna (bool, optional): Whether to exclude NaN values when counting frequencies.
            Default is True.
        na_position (str, optional): Position of NaN values in the output:
            - 'first': Place NaN at the top.
            - 'last': Place NaN at the bottom (default).
            - 'value': Keep NaN in its natural order.
            Default is 'last'.
        include_pcts (bool, optional): Whether to include percentage columns.
            If False, only absolute and cumulative frequencies are returned.
            Default is True.
        include_flat_relatives (bool, optional): Whether to return relative and cumulative relative values.
            If False, only frequency and percentage columns are included.
            Default is True.
        fmt_values (bool, optional): Whether to format numeric values using `_fmt_value_for_pd`.
            Useful for improving readability in reports. Default is False.
        order (str, optional): Sort order for the output:
            - 'asc': Sort values ascending.
            - 'desc': Sort values descending (default).
            - 'ix_asc': Sort by index ascending.
            - 'ix_desc': Sort by index descending.
            - None: No sorting.
            Default is 'desc'.
        na_aside_calc (bool, optional): Whether to separate NaN values from calculations but keep them in the output.
            If True, NaNs are added at the end and not included in cumulative or relative calculations.
            Default is True.
        index_name (str, optional): Set an specific index name for the fdt.

    Returns:
        pd.DataFrame: A DataFrame containing the frequency distribution table with the following columns
        (depending on parameters):
            - Frequency
            - Cumulative Frequency
            - Relative Frequency
            - Cumulative Relative Freq.
            - Relative Freq. [%]
            - Cumulative Freq. [%]

    Raises:
        ValueError: If `sort` or `na_position` receive invalid values.

    Notes:
        - This function uses `to_series` to convert input data into a pandas Series.
        - If `na_aside=True` and NaNs are present, they are placed separately and not included in relative calculations.
        - Useful for exploratory data analysis and generating clean statistical summaries.

    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'B', None])
        >>> fdt = get_fdt(data, sort='desc', fmt_values=True)
        >>> print(fdt)
              Frequency  Cumulative Frequency  Relative Freq. [%]  Cumulative Freq. [%]
        B           3                   3                42.86                  42.86
        A           2                   5                28.57                  71.43
        C           1                   6                14.29                  85.71
        Nulls       1                   7                14.29                 100.00
    """
    columns = [
        'Frequency',
        'Cumulative Frequency',
        'Relative Frequency',
        'Cumulative Relative Freq.',
        'Relative Freq. [%]',
        'Cumulative Freq. [%]'
    ]
    # def _calculate_fdt_relatives(series):     # Revisar, no me gusta el flujo actual
    
    sr = to_series(data)
    
    if dropna:
        sr = sr.dropna()                        # Drop all nulls values of the Series
        sr = sr.drop(np.nan, errors='ignore')   # For series with NaNs as a category with their count (errors='ignore': does not fail if it does not exist)

    if value_counts:
        sr = sr.value_counts(dropna=dropna, sort=False)

    # Validate that all the values are positive numbers
    _validate_numeric_series(sr)

    # Order de original Series to obtain the fdt in the same order as the original data
    match order:
        case 'asc':
            sr = sr.sort_values()
        case 'desc':
            sr = sr.sort_values(ascending=False)
        case 'ix_asc':
            sr = sr.sort_index()
        case 'ix_desc':
            sr = sr.sort_index(ascending=False)
        case None:
            pass
        case _:
            raise ValueError(f"Valid values for order: 'asc', 'desc', 'ix_asc', 'ix_desc', or None. Got '{order}'")
        
    # Handle NaNs values. Two cases: 1. na_aside: don't use for calcs and at the end; 2. use for calcs and locate according na_position
    #   - Determine the number of nans
    if pd.isna(sr.index).any():
        n_nans = sr[np.nan]
    else:
        n_nans = 0

    #   - Locale NaNs row in the Series 'sr'
    if na_aside_calc:
        sr = sr.drop(np.nan, errors='ignore')                   # Drop NaNs from the Series for calculations
        # Column that will then be concatenated to the end of the DF - Only 'Frequency' column, no calculated columns
        nan_row_df = pd.DataFrame(data = [n_nans], columns=[columns[0]], index=[np.nan])
    else:
        # As we use NaNs for calculations decide where locate these values
        sr_without_nan = sr.drop(np.nan, errors='ignore')       # Aux. sr wo/nans allow us to locate the NaNs
        match na_position:             
            case 'first':
                sr = pd.concat([pd.Series({np.nan: n_nans}), sr_without_nan])
            case 'last':
                sr = pd.concat([sr_without_nan, pd.Series({np.nan: n_nans})])
            case 'value' | None:
                pass                # Locates the Nulls row based on the value or index ordering
            case _:
                raise ValueError(f"Valid values for na_position: 'first', 'last', 'value' or None. Got '{na_position}'")

    # Central rutine: create the fdt, including relative and cumulative columns.
    fdt = pd.DataFrame(sr)
    fdt.columns = [columns[0]]
    fdt[columns[1]] = fdt['Frequency'].cumsum()
    fdt[columns[2]] = fdt['Frequency'] / fdt['Frequency'].sum()
    fdt[columns[3]] = fdt['Relative Frequency'].cumsum()
    fdt[columns[4]] = fdt['Relative Frequency'] * 100
    fdt[columns[5]] = fdt['Cumulative Relative Freq.'] * 100

    if na_aside_calc and not dropna:            # We add nan_columns at the end
        fdt = pd.concat([fdt, nan_row_df])

    # Logic to include: all values (all columns), only pcts, only flat values, of only frequencies (value counts), or only flat relatives
    if include_pcts is True and include_flat_values is True:
        fdt = fdt
    elif include_pcts is True and include_flat_values is False:
        fdt = fdt[[columns[0], columns[4], columns[5]]]
    elif include_pcts is False and include_flat_values is True:
        fdt = fdt[columns[0:4]]
    elif include_pcts is False and include_flat_values is False:
        fdt = fdt[[columns[0]]]
    else:
        raise ValueError('Paremeters include_pcts and include_flat_relatives must be bool, True or False')

    if fmt_values:
        fdt = fdt.map(_fmt_value_for_pd)

    # Set the index name
    fdt.index.name = index_name if index_name else sr.index.name
        
    return fdt


def describeplus(data, decimals=2, miles=',') -> pd.DataFrame:
    ''' Descriptive sats of data'''

    serie = to_series(data)          # Convert data to a pandas Series
    
    # Calc valid values for numerical and categorical series
    non_null_count = serie.count()
    null_count = serie.isnull().sum()
    num_uniques = serie.nunique()

    if len(serie) == non_null_count + null_count:
        total_count = len(serie)
    else:
        total_count = '[ERROR ¡?]'           # Error !?
    
    # Calc valid mode for any dtype
    modes = serie.mode()

    if len(modes) == 0:
        mode_str = "No mode"
    elif len(modes) == 1:
        mode_str = str(modes.iloc[0])
    else:
        mode_str = ", ".join(str(val) for val in modes)

    # Calc valid freq. (mode freq.) for any dtype
    if mode_str != "No mode":
        mode_freq = serie.value_counts().iloc[0] 
    else:
        mode_freq = mode_str

    # Avoid Object dtypes to calc stats
    serie = serie.convert_dtypes()

    # Calc. stats for numeric series
    try:
        stats = {
            'Non-null Count': non_null_count,
            'Null Count': null_count,
            'Total Count': total_count,
            'Unique Count': num_uniques,
            'Mean': serie.mean(),
            'Median (50%)': serie.median(),
            'Mode(s)': mode_str,
            'Mode_freq': mode_freq,
            'Skewness': serie.skew(),
            'Variance': serie.var(),
            'Standard Deviation': serie.std(),
            'Kurtosis': serie.kurt(),
            'Minimum': serie.min(),
            'Maximum': serie.max(),
            'Range': serie.max() - serie.min(),
            '25th Percentile': serie.quantile(0.25),
            '50th Percentile': serie.quantile(0.50),
            '75th Percentile': serie.quantile(0.75)
        }
    except:                                 # If the series is not numeric, or ? we will catch the exception and set categorical flag
        stats = {
            'Non-null Count': non_null_count,
            'Null Count': null_count,
            'Total Count': total_count,
            'Unique Count': num_uniques,
            'Top (mode)': mode_str,
            'Freq. mode': mode_freq
        }
    
    df = pd.DataFrame.from_dict(stats, orient='index', columns=[serie.name])
    
    if pd.api.types.is_numeric_dtype(serie):
        df['formatted'] = df[serie.name].apply(
            lambda x: _fmt_value_for_pd(x, width=8, decimals=decimals, miles=miles))      # Apply formatting to the stats values
    
    return df
    

def clean_df(df):
    ''' Delete duplicates and nulls'''
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.dropna(how='all', axis=1)
    return df_clean


def is_mostly_numeric(series, threshold):
    ''' Checks if at least 'threshold'% of the values ​​can be numeric'''
    converted = pd.to_numeric(series, errors='coerce')
    numeric_ratio = converted.notna().sum() / len(series)
    return numeric_ratio >= threshold


def petty_decimals_and_str(series):
    for ix, value in series.items():
        if isinstance(value, str):
            print(f"String -> {ix = } - {value = }")
        elif isinstance(value, float):
            if value % 1 > 0:
                print(f"float -> {ix = } - {value = }")






if __name__ == "__main__":

    df = pd.DataFrame({'A': [1, 2, pd.NA, pd.NA, 1],
                       'B': [4.0, pd.NA, pd.NA, 6.1, 4.0],
                       'C': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                       'D': ['x', 'y', pd.NA, 'z', 'x'],
                       'E': ['x', 'y', pd.NA, 'z', 'x']})
    
    print(df)

    df2 = clean_df(df)
    
    print(df2)
        


