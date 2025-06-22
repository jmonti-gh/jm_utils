'''
jm_img
'''

__version__ = "0.1.0"
__description__ = "Utilities I use frequently - Several modules"
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-15"


# Third-Party Libs
from PIL import Image

def png_to_ico(pngfile):
    ''' Save a nwe .ico file from a .png file
            pngfile: name of the png file without .png extensión'''
    img = Image.open(f"{pngfile}.png")
    img.save(f"{pngfile}.ico", format="ICO")

def img_to_excel(img):
    pass

def demo_png_to_ico():
    png_fielname = 'pngfile'
    png_to_ico(png_fielname)


if __name__ == '__main__':
    demo_png_to_ico()
