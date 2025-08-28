"""
To try charts
"""

# Third-party Libs
import pandas as pd

# Loccal Libs
import jm_utils.data.plt_charts as pcharths


# df, from xls spreadsheet
try:
    spreadsheet = r"C:\Users\jm\Documents\__Dev\PortableGit\__localrepos\365DS_jm\3_statistics\2_13_Practical_Ex_Descriptive_Stats.xlsx"    # Casa
    with open(spreadsheet) as f:
        pass
except FileNotFoundError:
    spreadsheet = r"D:\git\PortableGit\__localrepos\365DS_jm\3_statistics\2_13_Practical_Ex_Descriptive_Stats.xlsx"                         # Office

df = pd.read_excel(spreadsheet, skiprows=4, usecols='B:J,L:AA', index_col='ID')

# Pie chart df['Source']
pcharths.plt_pie(df['Source'], value_counts=True)

# lst, final values
ventas = [175, 100, 50, 220, 75]

# Bar chart
pcharths.plt_pie(ventas, order='ix_asc', palette='sns')

# Pareto Chart
pcharths.plt_pareto(df['State'], value_counts=True)


