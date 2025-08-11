# show_palettes

# Local Libs
from jm_utils.data import plt_functions as plt_func

palette_group = input("Ingrese el grupo de paletas a mostrar ['Qualitatives', 'Sequential', 'Diverging', 'Cyclic', 'Sample']: ")
n_items = input("Ingrese el número de colores que quiere mostrar [> 1 and < 26]: ")

palette_group = 'Sequential' if palette_group == '' else palette_group

try:
    n_items = int(n_items)
except ValueError:
    n_items = 16

fig = plt_func.show_plt_palettes(palette_group, n_items)
# input("Presione Enter para continuar...")

# fig_2 = plt_func.show_sns_palettes(palette_group, n_items)
