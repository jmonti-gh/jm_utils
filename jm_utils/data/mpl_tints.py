# jm_utils/data/mpl_tints
"""
Dictionaries and functions related to matplotlib colors and palettes

We use 'tints' instead of color in the name so as not to confuse it with the original matplotlib 'colors' module. 
-> import jm_utils.data.mpl_tints as mpl_tints
"""

## TO-DO
# Dicts
# plot_colors
# plot_palettes


__version__ = "0.1.0"
__description__ = "Diccionarios y funciones relacionadas con colores y paletas matplotlib."
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-08-20"


## Standard Libs
import random
import textwrap
from typing import Union, Optional, Any, Literal, Sequence, TypeAlias

# Third-Party Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors             # for get_color_list()
from matplotlib import colormaps                # for show_matplotlib_palettes()
from matplotlib.patches import Rectangle
# # from colorspacious import cspace_converter
# import matplotlib as mpl

# Local Libs
from jm_utils.data.pd_functions import to_series, get_fdt

## Custom types for non-included typing annotations
IndexElement: TypeAlias = Union[str, int, float, pd.Timestamp]
# IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', np.str_, np.int64, np.float64, np.datetime64, pd.Timestamp, ...]


## Dictionaries - Own colors dictionary added to matplotlib
BRAND_COLORS = {
    '365 DataScience': '#108A99', 'AT&T': '#00A8E0', 'Airbnb': '#FD5C63', 'Alibaba': '#FF6A00', 'Android': '#A4C639', 'Cadbury': '#472F92', 'DELL': '#0085C3', 'Django': '#092E20',
    'Docker(1)': '#0DB7ED', 'Docker(2)': '#384D54', 'E4': '#6D1D7C', 'ESPN': '#FF0033', 'Facebook': '#3B5998', 'Fairmont(1)': '#A6A685', 'Fairmont(2)': '#E32119', 'Ferrari': '#1C396D',
    'Ford': '#4078C0', 'GitHub(1)': '#FCA326', 'GitLab(1)': '#006699', 'IBM': '#F5CB39', 'IKEA(1)': '#FF6600', 'JBL': '#9CB443', 'Khan Academy(1)': '#242F3A', 'Khan Academy(2)': '#DDB321',
    'Lamborghini': '#0A66C2', 'LinkedIn(1)': '#0A3A6C', 'MTM(1)': '#113D76', 'MTM(2)': '#D6682D', 'MTM(3)': '#D8630E', 'MTM(4)': '#00A1F1', 'Microsoft': '#589636','MongoDB': '#5C92FA',
    'Motorola': '#00758F', 'MySQL(1)': '#F29111', 'MySQL(2)': '#FFCC00', 'National Geographic(1)': '#000000', 'National Geographic(2)': '#08107B', 'Olympus(1)': '#DFB226', 'Olympus(2)': '#777777', 'Olympus(3)': '#8892BE',
    'PHP(1)': '#4F5B93', 'PHP(2)': '#99CC99', 'PHP(3)': '#FFDE57', 'Python(1)': '#4584B6', 'Python(2)': '#646464', 'Python(3)': '#25D366', 'WhatsApp': '#9D0A0E', 'Western Digital(5)': '#003369',
    'Western Digital(6)': '#FF0000'
}

AUTOMOTIVE_COLORS = {
    "audi": "#ffffff", "audi-1": "#ffffff", "audi-2": "#000000", "audi-3": "#f50537",
    "continental-ag": "#ffa500", "continental-ag-1": "#ffa500", "continental-ag-2": "#00a5dc", "continental-ag-3": "#004eaf", "continental-ag-4": "#2db928", "continental-ag-5": "#057855", "continental-ag-6": "#ff2d37", "continental-ag-7": "#000000", "continental-ag-8": "#737373", "continental-ag-9": "#969696", "continental-ag-10": "#cdcdcd", "continental-ag-11": "#f0f0f0",
    "daimler": "#263f6a", "daimler-1": "#263f6a", "daimler-2": "#182b45", "daimler-3": "#6b0f24", "daimler-4": "#193725", "daimler-5": "#606061",
    "ferrari": "#e32119", "ferrari-1": "#e32119",
    "fiat": "#96172e", "fiat-1": "#96172e", "fiat-2": "#6d2d41",
    "ford": "#1c396d", "ford-1": "#1c396d",
    "kia": "#c21a30", "kia-1": "#c21a30",
    "lamborghini": "#ddb321", "lamborghini-1": "#ddb321",
    "nexar": "#fbb040", "nexar-1": "#fbb040", "nexar-2": "#44355b", "nexar-3": "#31263e", "nexar-4": "#ee5622", "nexar-5": "#221e22",
    "rolls-royce": "#680021", "rolls-royce-1": "#680021", "rolls-royce-2": "#fffaec", "rolls-royce-3": "#939598", "rolls-royce-4": "#000000",
    "skoda": "#00800d", "skoda-1": "#00800d",
    "tesla": "#cc0000", "tesla-1": "#cc0000",
    "toyota": "#eb0a1e", "toyota-1": "#eb0a1e", "toyota-2": "#ffffff", "toyota-3": "#000000", "toyota-4": "#58595b",
    "volvo": "#003057", "volvo-1": "#003057", "volvo-2": "#115740", "volvo-3": "#65665c", "volvo-4": "#425563", "volvo-5": "#517891", "volvo-6": "#212721",
    "webzunder": "#eea642", "webzunder-1": "#eea642"
}

EDUCATION_COLORS = {
    '365 DataScience': '#108999', '365 DataScience-1': '#108999', '365 DataScience-2': '#ee4f23', '365 DataScience-3': '#748190', '365 DataScience-4': '#1e9438', '365 DataScience-5': '#175f69', '365 DataScience-6': '#8f2f15', '365 DataScience-7': '#485361', '365 DataScience-8': '#0d6323', '365 DataScience-9': '#9fd0d6',
    "aiesec": "#037ef3", "aiesec-1": "#037ef3", "aiesec-2": "#f85a40", "aiesec-3": "#00c16e", "aiesec-4": "#7552cc", "aiesec-5": "#0cb9c1", "aiesec-6": "#f48924", "aiesec-7": "#ffc845", "aiesec-8": "#52565e", "aiesec-9": "#caccd1", "aiesec-10": "#f3f4f7",
    "boise-state-university": "#09347a", "boise-state-university-1": "#09347a", "boise-state-university-2": "#007dc3", "boise-state-university-3": "#0169a4", "boise-state-university-4": "#3399cc", "boise-state-university-5": "#f1632a", "boise-state-university-6": "#464646", "boise-state-university-7": "#b7b7b7", "boise-state-university-8": "#f6f6f5",
    "clemson-university": "#f66733", "clemson-university-1": "#f66733", "clemson-university-2": "#522d80", "clemson-university-3": "#d4c99e", "clemson-university-4": "#685c53", "clemson-university-5": "#a25016", "clemson-university-6": "#562e19", "clemson-university-7": "#86898c", "clemson-university-8": "#f9e498", "clemson-university-9": "#566127", "clemson-university-10": "#3a4958", "clemson-university-11": "#b5c327", "clemson-university-12": "#109dc0",
    "code-school": "#616f67", "code-school-1": "#616f67", "code-school-2": "#c68143",
    "codecademy": "#f65a5b", "codecademy-1": "#f65a5b", "codecademy-2": "#204056",
    "duke-university": "#001a57", "duke-university-1": "#001a57", "duke-university-2": "#003366",
    "duolingo": "#7ac70c", "duolingo-1": "#7ac70c", "duolingo-2": "#8ee000", "duolingo-3": "#faa918", "duolingo-4": "#ffc715", "duolingo-5": "#d33131", "duolingo-6": "#e53838", "duolingo-7": "#1cb0f6", "duolingo-8": "#14d4f4", "duolingo-9": "#8549ba", "duolingo-10": "#a560e8", "duolingo-11": "#4c4c4c", "duolingo-12": "#6f6f6f", "duolingo-13": "#cfcfcf", "duolingo-14": "#f0f0f0", "duolingo-15": "#bff199", "duolingo-16": "#f7c8c9",
    "freecodecamp": "#006400", "freecodecamp-1": "#006400", "freecodecamp-2": "#ff9c2a", "freecodecamp-3": "#ff4025", "freecodecamp-4": "#3949ab", "freecodecamp-5": "#efefef",
    "goethe": "#a5c500", "goethe-1": "#a5c500", "goethe-2": "#810061", "goethe-3": "#303600", "goethe-4": "#ec6400", "goethe-5": "#d4c78c", "goethe-6": "#5ec6f2", "goethe-7": "#003468", "goethe-8": "#4b1702", "goethe-9": "#717a83",
    "khan-academy": "#9cb443", "khan-academy-1": "#9cb443", "khan-academy-2": "#242f3a",
    "massachusetts-institute-of-technology": "#a31f34", "massachusetts-institute-of-technology-1": "#a31f34", "massachusetts-institute-of-technology-2": "#8a8b8c", "massachusetts-institute-of-technology-3": "#c2c0bf",
    "montclair-state-university": "#ce1141", "montclair-state-university-1": "#ce1141", "montclair-state-university-2": "#eeb111", "montclair-state-university-3": "#e87d1e", "montclair-state-university-4": "#94ce08", "montclair-state-university-5": "#00386b", "montclair-state-university-6": "#969491",
    "msu": "#18453b", "msu-1": "#18453b", "msu-2": "#000000", "msu-3": "#ffffff", "msu-4": "#008208", "msu-5": "#7bbd00", "msu-6": "#0b9a6d",
    "oxford-university-press": "#002147", "oxford-university-press-1": "#002147", "oxford-university-press-2": "#000000", "oxford-university-press-3": "#666666",
    "pearson": "#ed6b06", "pearson-1": "#ed6b06", "pearson-2": "#9d1348", "pearson-3": "#008b5d", "pearson-4": "#364395",
    "portfolium": "#0099ff", "portfolium-1": "#0099ff", "portfolium-2": "#fb0a2a", "portfolium-3": "#17ad49", "portfolium-4": "#333333",
    "quizlet": "#4257b2", "quizlet-1": "#4257b2", "quizlet-2": "#3ccfcf", "quizlet-3": "#f0f0f0",
    "rit": "#f76902", "rit-1": "#f76902", "rit-2": "#ffffff", "rit-3": "#000000",
    "rochester-institute-of-technology": "#f36e21", "rochester-institute-of-technology-1": "#f36e21", "rochester-institute-of-technology-2": "#513127",
    "rosetta-stone": "#0098db", "rosetta-stone-1": "#0098db", "rosetta-stone-2": "#ecc400",
    "rowan-university": "#3f1a0a", "rowan-university-1": "#3f1a0a", "rowan-university-2": "#edd51c",
    "rutgers-university": "#cc0033", "rutgers-university-1": "#cc0033",
    "seton-hall-university": "#004488", "seton-hall-university-1": "#004488",
    "skillshare": "#f26b21", "skillshare-1": "#f26b21", "skillshare-2": "#68b8be",
    "studyblue": "#00afe1", "studyblue-1": "#00afe1",
    "temple": "#a41e35", "temple-1": "#a41e35", "temple-2": "#222222", "temple-3": "#899197",
    "texas-am-university": "#500000", "texas-am-university-1": "#500000", "texas-am-university-2": "#003c71", "texas-am-university-3": "#5b6236", "texas-am-university-4": "#744f28", "texas-am-university-5": "#998542", "texas-am-university-6": "#332c2c", "texas-am-university-7": "#707373", "texas-am-university-8": "#d6d3c4",
    "texas-tech-university": "#cc0000", "texas-tech-university-1": "#cc0000", "texas-tech-university-2": "#000000",
    "the-college-of-new-jersey": "#293f6f", "the-college-of-new-jersey-1": "#293f6f", "the-college-of-new-jersey-2": "#a67a00",
    "treehouse": "#6fbc6d", "treehouse-1": "#6fbc6d", "treehouse-2": "#47535b",
    "uc-berkeley": "#003262", "uc-berkeley-1": "#003262", "uc-berkeley-2": "#3b7ea1", "uc-berkeley-3": "#fdb515", "uc-berkeley-4": "#c4820e",
    "ucsf": "#052049", "ucsf-1": "#052049", "ucsf-2": "#18a3ac", "ucsf-3": "#90bd31", "ucsf-4": "#178ccb", "ucsf-5": "#f48024",
    "universitat-hamburg": "#e2001a", "universitat-hamburg-1": "#e2001a", "universitat-hamburg-2": "#009cd1", "universitat-hamburg-3": "#3b515b",
    "university-of-cambridge": "#d6083b", "university-of-cambridge-1": "#d6083b", "university-of-cambridge-2": "#0072cf", "university-of-cambridge-3": "#ea7125", "university-of-cambridge-4": "#55a51c", "university-of-cambridge-5": "#8f2bbc", "university-of-cambridge-6": "#00b1c1",
    "xavier-university": "#0c2340", "xavier-university-1": "#0c2340", "xavier-university-2": "#9ea2a2", "xavier-university-3": "#0099cc"
}

FINANCIAL_COLORS = {
    "adyen": "#0abf53", "adyen-1": "#0abf53", "adyen-2": "#00112c",
    "american-express": "#002663", "american-express-1": "#002663", "american-express-2": "#4d4f53",
    "amp": "#1c79c0", "amp-1": "#1c79c0", "amp-2": "#0dd3ff", "amp-3": "#0389ff",
    "barclays": "#00aeef", "barclays-1": "#00aeef", "barclays-2": "#00395d",
    "blockchain": "#123962", "blockchain-1": "#123962", "blockchain-2": "#2754ba", "blockchain-3": "#00aee6", "blockchain-4": "#799eb2", "blockchain-5": "#b1d4e5",
    "capital-one": "#004977", "capital-one-1": "#004977", "capital-one-2": "#d03027",
    "diebold": "#007dc3", "diebold-1": "#007dc3", "diebold-2": "#003f69", "diebold-3": "#954010", "diebold-4": "#445c6e", "diebold-5": "#005238", "diebold-6": "#97824b",
    "dwolla": "#ff7404", "dwolla-1": "#ff7404",
    "etrade": "#6633cc", "etrade-1": "#6633cc", "etrade-2": "#99cc00",
    "flattr": "#f67c1a", "flattr-1": "#f67c1a", "flattr-2": "#338d11",
    "gittip": "#663300", "gittip-1": "#663300", "gittip-2": "#339966",
    "hellowallet": "#0093d0", "hellowallet-1": "#0093d0",
    "hsbc": "#db0011", "hsbc-1": "#db0011",
    "ideal": "#cc0066", "ideal-1": "#cc0066", "ideal-2": "#79afc1", "ideal-3": "#000000",
    "indiegogo": "#eb1478", "indiegogo-1": "#eb1478",
    "ing": "#ff6200", "ing-1": "#ff6200", "ing-2": "#000066",
    "intuit": "#365ebf", "intuit-1": "#365ebf",
    "kickstarter": "#2bde73", "kickstarter-1": "#2bde73", "kickstarter-2": "#0f2105",
    "kiwipay": "#00b0df", "kiwipay-1": "#00b0df",
    "lloyds": "#d81f2a", "lloyds-1": "#d81f2a", "lloyds-2": "#ff9900", "lloyds-3": "#e0d86e", "lloyds-4": "#9ea900", "lloyds-5": "#6ec9e0", "lloyds-6": "#007ea3", "lloyds-7": "#9e4770", "lloyds-8": "#631d76", "lloyds-9": "#1e1e1e",
    "localbitcoins-com": "#006fbf", "localbitcoins-com-1": "#006fbf", "localbitcoins-com-2": "#ff7b00",
    "mastercard": "#eb001b", "mastercard-1": "#eb001b", "mastercard-2": "#ff5f00", "mastercard-3": "#f79e1b",
    "mollie": "#c6d6df", "mollie-1": "#c6d6df", "mollie-2": "#ec4534",
    'MTM': '#113D76', 'MTM-1': '#113D76', 'MTM-2': '#D8630E', 'MTM-3': '#545559', 'MTM-4': '#161616', 'MTM-5': '#f4f4f4', 'MTM-6': '#FF7F22', 'MTM-7': '#1091EF', 'MTM-8': '#222222', 'MTM-9': '#b4b5bb', 'MTM-10': '#ffffff', 'MTM-11': '#00000012', 'MTM-12': '#0363AA', 'MTM-13': '#0A2540', 'MTM-14': '#0A254080',
    "n26": "#2b697a", "n26-1": "#2b697a", "n26-2": "#000000", "n26-3": "#ffffff",
    "paymill": "#f05000", "paymill-1": "#f05000",
    "paypal": "#003087", "paypal-1": "#003087", "paypal-2": "#009cde", "paypal-3": "#012169",
    "qonto": "#6b5aed", "qonto-1": "#6b5aed", "qonto-2": "#262a3e", "qonto-3": "#fafafc", "qonto-4": "#63ebe4",
    "realex-payments": "#f29023", "realex-payments-1": "#f29023", "realex-payments-2": "#4d5255",
    "square-cash": "#28c101", "square-cash-1": "#28c101",
    "stripe": "#00afe1", "stripe-1": "#00afe1",
    "suntrust": "#00447c", "suntrust-1": "#00447c", "suntrust-2": "#fdb913", "suntrust-3": "#e36f1e",
    "turbotax": "#355ebe", "turbotax-1": "#355ebe", "turbotax-2": "#d52b1d",
    "virgin-money": "#cc0000", "virgin-money-1": "#cc0000", "virgin-money-2": "#333333", "virgin-money-3": "#a93c3b", "virgin-money-4": "#a896a0", "virgin-money-5": "#7a3671", "virgin-money-6": "#bd1d65",
    "visa": "#1a1f71", "visa-1": "#1a1f71", "visa-2": "#f7b600",
    "wave-apps": "#1c2d37", "wave-apps-1": "#1c2d37", "wave-apps-2": "#4ec7c4", "wave-apps-3": "#00959f", "wave-apps-4": "#3b9bcc",
    "worldline": "#0066a1", "worldline-1": "#0066a1",
    "xero": "#06b3e8", "xero-1": "#06b3e8", "xero-2": "#000000", "xero-3": "#ffffff", "xero-4": "#7a7e85",
    "y-combinator": "#ff4000", "y-combinator-1": "#ff4000"
}

PROGRAMMING_COLORS = {
    "angularjs": "#b52e31", "angularjs-1": "#b52e31", "angularjs-2": "#000000",
    "django": "#092e20", "django-1": "#092e20",
    "docker": "#0db7ed", "docker-1": "#0db7ed", "docker-2": "#384d54",
    "ember": "#f23819", "ember-1": "#f23819",
    "grunt": "#fba919", "grunt-1": "#fba919", "grunt-2": "#463014",
    "html5": "#e34f26", "html5-1": "#e34f26",
    "javascript": "#f7df1e", "javascript-1": "#f7df1e",
    "jquery": "#0769ad", "jquery-1": "#0769ad", "jquery-2": "#7acef4",
    "laravel": "#f55247", "laravel-1": "#f55247",
    "node-js": "#215732", "node-js-1": "#215732", "node-js-2": "#6cc24a", "node-js-3": "#44883e", "node-js-4": "#333333",
    "npm": "#cb3837", "npm-1": "#cb3837",
    "php": "#8892be", "php-1": "#8892be", "php-2": "#4f5b93", "php-3": "#99cc99",
    "python": "#ffde57", "python-1": "#ffde57", "python-2": "#4584b6", "python-3": "#646464",
    "react": "#00d8ff", "react-1": "#00d8ff",
    "ruby": "#cc342d", "ruby-1": "#cc342d",
    "ruby-on-rails": "#cc0000", "ruby-on-rails-1": "#cc0000",
    "typescript": "#3178c6", "typescript-1": "#3178c6", "typescript-2": "#00273f",
    "yii-framework": "#d8582b", "yii-framework-1": "#d8582b", "yii-framework-2": "#16a314", "yii-framework-3": "#3b6fba",
    "vue-js": "#42b883", "vue-js-1": "#42b883", "vue-js-2": "#35495e"
}

# dict_list = [AUTOMOTIVE_COLORS, EDUCATION_COLORS, FINANCIAL_COLORS, PROGRAMMING_COLORS]

## Dictionaries
COLORS_NAMES_BY_GROUP = {
    "BASE_COLORS": (list(mcolors.BASE_COLORS.keys()),
                    """One letter color names: 'b'lue, 'g'reen, 'r'ed, 'c'yan, 'm'agenta, 'y'ellow, blac'k', 'w'hite
                    The colors g, c, m, and y do not coincide with X11/CSS4 colors. Their particular shades were chosen for better visibility of 
                    colored lines against typical backgrounds"""),
    'CSS4_COLORS': (list(mcolors.CSS4_COLORS.keys()),
                    "Case-insensitive X11/CSS4 color name with no spaces"),
    'TABLEAU_COLORS': (list(mcolors.TABLEAU_COLORS.keys()),
                    "Tableau Palette"),
    # 'BRAND_COLORS': ([
    #     '#108A99', '#00A8E0', '#FD5C63', '#FF6A00', '#A4C639', '#472F92', '#0085C3', '#092E20', '#0DB7ED', '#384D54',
    #     '#6D1D7C', '#FF0033', '#3B5998', '#A6A685', '#E32119', '#1C396D', '#4078C0', '#FCA326', '#006699', '#F5CB39',
    #     '#FF6600', '#9CB443', '#242F3A', '#DDB321', '#0A66C2', '#0A3A6C', '#113D76', '#D6682D', '#D8630E', '#00A1F1',
    #     '#589636', '#5C92FA', '#00758F', '#F29111', '#FFCC00', '#000000', '#08107B', '#DFB226', '#777777', '#8892BE',
    #     '#4F5B93', '#99CC99', '#FFDE57', '#4584B6', '#646464', '#25D366', '#9D0A0E', '#003369', '#FF0000'
    #     ],
    #     """365 DataScience, AT&T, Airbnb, Alibaba, Android, Cadbury, DELL, Django, Docker(1), Docker(2), E4, ESPN, Facebook, Fairmont(1),
    #     Fairmont(2), Ferrari, Ford, GitHub(1), GitLab(1), IBM, IKEA(1), JBL, Khan Academy(1), Khan Academy(2), Lamborghini, LinkedIn(1),
    #     MTM(1), MTM(2), MTM(3), MTM(4), Microsoft, MongoDB, Motorola, MySQL(1), MySQL(2), National Geographic(1), National Geographic(2),
    #     Olympus(1), Olympus(2), Olympus(3), PHP(1), PHP(2), PHP(3), Python(1), Python(2), Python(3), Wathsapp, Western Digital(5),
    #     Western Digital(6), Youtube(1)"""),
    'XKCD_COLORS': (list(mcolors.XKCD_COLORS.keys()),
            "The 954 most common RGB monitor colors, as defined by several hundred thousand participants in the xkcd color name survey"),
    'AUTOMOTIVE_COLORS': (list(AUTOMOTIVE_COLORS.keys()),
                          " Automotive brands colors")
}


JM_QUALITATIVE_CMAPS = {
    'colorblind': ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133',
                   '#56B4E9', '#5D8C3B', '#A93967', '#888888', '#FFC107', '#7C9680', '#E377C2', '#BCBD22', '#AEC7E8',
                   '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94', '#F7B6D2', '#DBDB8D', '#9EDAE5', '#D68E3A',
                   '#A65898', '#B2707D', '#8E6C87'],
    "neon": ["#04e762", "#f5b700", "#dc0073", "#008bf8", "#89fc00", "#3cfc8b", "#dcab18", "#dc1100", "#0060ac", 
             "#89fc00", "#2ee704", "#ffcf42"],
    "neat": ["#6494aa", "#a63d40", "#e9b872", "#90a959", "#904ca9", "#e56ba4", "#3d3d3d", "#95b6c5", "#8f5455",
             "#d3e972", "#66783e", "#9b33c2"],
    "serene": ["#e7ecef", "#274c77", "#6096ba", "#a3cef1", "#8b8c89", "#ffffff", "#374d67", "#6060ba", "#61abe7",
               "#93a372", "#e7efed", "#3a71b1"],
    "beach_day": ["#001524", "#15616d", "#ffecd1", "#ff7d00", "#78290f", "#004270", "#225860", "#f6ffd1", "#b25800",
                  "#852202", "#00241d", "#219aad"],
    "vibrant_summer": ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93", "#ffa6a8", "#ebc14e", "#28c926", "#105580",
                       "#6736a9", "#ff59c2", "#ffdf87"]
}


CMAP_NAMES_BY_CAT = {
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
            "cubehelix", "cubehelix_r", "gist_earth", "gist_earth_r", "gist_grey", "gist_grey_r", "gist_heat", "gist_heat_r",
            "gist_yarg", "gist_yarg_r", "gist_yerg", "gist_yerg_r", "hot", "hot_r", "managua", "managua_r",
            "ocean", "ocean_r", "pink", "pink_r", "terrain", "terrain_r", "vanimo", "vanimo_r",
            "Wistia", "Wistia_r"
        ], 'For representing information that has ordering'),

        "Qualitative": ([
            "Accent", "Accent_r", 'colorblind', 'colorblind_r', "Dark2", "Dark2_r", "neon", "neon_r",
            "Paired", "Paired_r", "Pastel1", "Pastel1_r", "Pastel2", "Pastel2_r", "neat", "neat_r",
            "serene", "serene_r", "Set1", "Set1_r", "Set2", "Set2_r", "Set3", "Set3_r",
            "beach_day", "beach_day_r", "tab10", "tab10_r", "tab20", "tab20_r", "tab20b", "tab20b_r",
            "tab20c", "tab20c_r", "vibrant_summer", "vibrant_summer_r"
        ], 'To represent information which does not have ordering or relationships. Also for categorical data')
}


## Functions
def get_named_colors_mapping():
    """ Function that returns a mapping of named colors to their hex values. """
    return {**AUTOMOTIVE_COLORS, **EDUCATION_COLORS, **FINANCIAL_COLORS, **PROGRAMMING_COLORS}

def get_hex_color(color_name: str):
    "Function that returns the hex_val of the color 'name' to be used in graphics."

    jm_colors = get_named_colors_mapping()
    mpl_hex_colors = {**mcolors.CSS4_COLORS, **mcolors.TABLEAU_COLORS, **mcolors.XKCD_COLORS}

    if color_name in jm_colors:
        return jm_colors[color_name]
    elif color_name in mpl_hex_colors:
        return mpl_hex_colors[color_name]
    elif color_name in mcolors.BASE_COLORS:
        return mcolors.rgb2hex(color_name)
    else:
        return None
    

def register_mpl_palette(cmap_name, cmap, n_bins=256):
    """ Function that register a custom palette (cmap) to Matplotlib """
    if cmap_name not in colormaps():
        cmap_custom = mcolors.LinearSegmentedColormap.from_list(cmap_name, cmap, N=n_bins)
        plt.colormaps.register(cmap_custom)
        return cmap_custom


def get_color_hex_list(palette: str, n_colors: Optional[int] = 10) -> list[str]:
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
    if palette in JM_QUALITATIVE_CMAPS.keys():
        return JM_QUALITATIVE_CMAPS[palette]
    else:
        cmap = plt.get_cmap(palette)                        # Get the colormap
        colors_normalized = np.linspace(0, 1, n_colors)     # Generate equidistant points between 0 and 1
        colors_rgba = cmap(colors_normalized)               # Get the colors from colormap
        return [mcolors.rgb2hex(color[:3]) for color in colors_rgba]
    

def ax_one_color(ax: plt.Axes, color_name: str) -> plt.Axes:
    """Function build an ax that is a rectangle of the color_name"""

    if color_name in get_named_colors_mapping():
        color = get_hex_color(color_name)  # Try to get the color from the custom dictionaries
    else:
        color = color_name
    
    ax.add_patch(Rectangle(xy=(0, 0), width=1, height=1, color=color))
    ax.set_axis_off()
    return ax


def plot_a_color(color_name: str): # -> tuple(plt.Figure, plt.Axes):
    """Function to plot a color by name."""

    fig, ax = plt.subplots(figsize=(2, 2))
    ax_one_color(ax, color_name)
    return fig, ax


def plot_colors(
        color_group: Optional[str | list[str]] = 'SAMPLE',
        alpha: Optional[float | None] = None,
        n_cols: Optional[int] = 6,
        sort_colors: Optional[bool] = True,
        hex_value: Optional[bool] = False
) -> tuple[plt.Figure, plt.Axes]:
    
    # Build the dictionary containing the color names sorted by category (dict key)
    colors_by_category_dic = COLORS_NAMES_BY_CAT

    # Add 'SAMPLE' key entry to colors_by_category_dic, value: (Sample of 8 colors of e/category, and a Description)
    list_of_pltt_lists = [value[0] for value in colors_by_category_dic.values()]              # Needed as source of data to get a random sample of 4 palettes of e/category
    colors_by_category_dic['SAMPLE'] = (                                                      # Added 'Sample' dict_key
        [color for c_g in list_of_pltt_lists for color in random.sample(c_g, k=8)],           # A random sample of 8 of e/category
        "8 colors of e/category: 8 BASE_COLORS, 8 CSS4_COLORS, 8 TABLEAU_COLORS, 8 XKCD_COLORS")

    # First internal aux. funct. _plot_dic(): generates a figure containing the names of the palettes according to their category
    def _show_dic(dic):
        all_text =""
        for group_name, (color_list, description) in dic.items():
            if group_name == 'SAMPLE':          # Sample list is not showed
                continue
            sp, ncols = 21, 10                  # sp: space between color names, n_cols: number of columns
            if group_name == 'XKCD_COLORS':
                sp, ncols = 27, 8               # Grater space between color names and less columns
                
            all_text += f"* {group_name} ({len(color_list)}).- {description}:\n"    # Category label and description

            # Dashed subline construction
            leng_ln = len(group_name) + len(description) + 12
            n_dashes = leng_ln if leng_ln < 151 else 150
            all_text += '-' * n_dashes + "\n"

            # Formeatado con f-strings
            for i, color in enumerate(color_list):
                all_text += f"{color:<{sp}}"
                if (i + 1) % ncols == 0:
                    all_text += '\n' if color != color_list[-1] else ''
            all_text += '\n\n\n'

        # Build de Figure showing all text
        fig, ax = plt.subplots(figsize=(20, len(all_text.splitlines()) * 0.2))
        ax.set_axis_off()                          # Hide x and y axis
        ax.set_title('Matplotlib colors names by category (color_group)', fontsize=12, fontweight='medium', family="monospace")
        ax.text(0.025, 0.46, all_text, fontsize=10, va="center", ha="left", family="monospace")
        plt.show()
        return fig, ax

    # 2nd internal aux. funct. _search_request(): build a color_grp_names_lst based on colors requested to find
    def _get_the_required_search(search):

        # Build a big list (most_colors) w/all colors, where we we'll search (except BASE_COLORS, those will be allocated en a dict)
        base_colors_lst = dic['BASE_COLORS'][0]
        most_colors_lst = dic['CSS4_COLORS'][0] + dic['TABLEAU_COLORS'][0] + dic['XKCD_COLORS'][0]
        
        base_colors_fullname_lst = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
        base_colors_dic = dict(zip(base_colors_fullname_lst, base_colors_lst))         # dict(zip(keys, values))

        # Beging the search in the most_color_lst and the base_colors_dic
        searched_colors_lst = search.split()[1:]        # only the colors (or strings to search for) separating it from the word 'FIND '
        # found_colors = [color for color in searched_colors_lst if color.lower() in most_colors_lst] # FUTURE
        colors_lst_tmp = []
        for target in searched_colors_lst:
            try:                                        # Cause key could not exist in base_colors_dic                                
                colors_lst_tmp.append(base_colors_dic[target.lower()])
            except KeyError:
                pass
            filtered_colors = list(filter(lambda color: target.lower() in color.lower(), most_colors_lst))
            colors_lst_tmp.append(filtered_colors.copy())

        found_colors = [color for sublist in colors_lst_tmp for color in sublist]              # color_grp_names_lst to plot. All colors found in a single list of strings
        if not found_colors:
            return ['none']
        else:
            return found_colors

    # Validate and preprocess palette_group parameter: get the palette_group_key, print lists/names if selected, or fill custom list
    if isinstance(color_group, str):          
        color_group_key = color_group.strip().upper()
        if color_group_key == 'NAMES':
            fig = _show_dic(colors_by_category_dic)
            return fig, ax
        elif color_group_key.startswith('FIND '):                    # Acá vamos a hacer un jorgitomonteada
            color_grp_names_lst = _get_the_required_search(color_group_key)   
            color_group_desc = color_group_key
            color_group_key = 'Search request'
        elif color_group_key not in colors_by_category_dic.keys():
            raise ValueError(f"Invalid value for 'palette_group': {repr(color_group)}. Expected one of:" 
                             "BASE_COLORS', 'CSS4_COLORS', 'TABLEAU_COLORS', 'XKCD_COLORS', 'SAMPLE' (default), 'NAMES'.")
        else:
            # Get the color_grp_names_lst and color_group_desc of the selected color category (group) - color_group_key
            color_grp_names_lst, color_group_desc = colors_by_category_dic[color_group_key]
                
    elif isinstance(color_group, list):   
        color_group_key = 'Custom'              # Only for title, no a new entry to de dictionary
        color_grp_names_lst = color_group      # The list of entered palettes to be shown
        color_group_desc = 'User selected colors'
    else:
        raise TypeError(f"Invalid type for 'palette_group': {repr(color_group)}. Expected one of: 'str' or 'list'.")
    
    # Sort colors by hue (tono), saturation, value and name (if sort_colors param id True). Reorder color_list 
    if sort_colors is True:                       
        color_grp_names_lst = sorted(color_grp_names_lst, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))

    # Buil the fig. with colors and colors names of the selected group: color_group_key: (color_grp_names_lst, color_group_desc)
    cell_width, cell_height, swatch_width, margin = 212, 22, 48, 12

    nrows = np.ceil(len(color_grp_names_lst) / n_cols)

    width = cell_width * n_cols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * n_cols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(f"* {color_group_key} ({len(color_grp_names_lst)}).- {color_group_desc}:\n",
                 fontsize=14, fontweight='bold')

    for i, color_name in enumerate(color_grp_names_lst):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        color_label = color_name if hex_value is False else mcolors.to_hex(color_name)
        ax.text(text_pos_x, y, color_label, fontsize=14, ha='left', va='center')

        ax.add_patch(
            # Rectangle(xy=(swatch_start_x, y-9), width=swatch_width, height=18, facecolor=(color_name, alpha), edgecolor='0.7')
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width, height=18, facecolor=color_name, edgecolor='0.7', alpha=alpha)
        )

    return fig, ax


def plot_mpl_palettes(
        palette_group: Union[str, list[str]] = 'Sample',
        n_colors: Optional[int] = 64,
        continous: Optional[bool] = False
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

        continous (bool, optional): If False, displays palettes as discrete color bars.
            If True, displays them as continuous color gradients.
            Default is False.

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
    palettes_by_category_dic = CMAP_NAMES_BY_CAT.copy()                                       # dict_keys(['Cyclic', 'Diverging', 'Miscellaneous', 'Perceptually Uniform Sequential', 'Single-Hue Sequential', 'Multi-Hue Sequential', 'Special Sequential', 'Qualitative'])
    
    list_of_pltt_lists = [value[0] for value in palettes_by_category_dic.values()]      # Nedeed as source of data to get a random sample of 4 palettes of e/category
    palettes_by_category_dic['Sample'] = (                                              # Added 'Sample' dict_key
        [pltt for p_g in list_of_pltt_lists for pltt in random.sample(p_g, k=4)],       # A random sample of four of e/category
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
                             "'Cyclic', 'Diverging', 'Miscellaneous', 'Perceptually Uniform Sequential',"
                             "'Single-Hue Sequential', 'Multi-Hue Sequential', 'Other Sequential', 'Qualitative','Sample'.")
        else:
            # Get the palette_group_list and palette_group_desc of the selected palette category (group)
            palette_group_list, palette_group_desc = palettes_by_category_dic[palette_group_key]
    elif isinstance(palette_group, list):   
        palette_group_key = 'Custom'            # Only for title, no a new entry to de dictionary
        palette_group_list = palette_group      # The list of entered palettes to be shown
        palette_group_desc = 'User selected palettes'
    else:
        raise TypeError(f"Invalid type for 'palette_group': {repr(palette_group)}. Expected one of: 'str' or 'list'.")

    if continous is False:                                                # Displays n_colors from the palette slightly separated by a thin white line
        # Build a Series of n_items elements to show colors
        sr = to_series({str(i): 1 for i in range(1, n_colors)})

        # Create a figure with two columns for the palettes - Bar charts showing palette colors
        rows = len(palette_group_list) // 2 if len(palette_group_list) % 2 == 0 else (len(palette_group_list) // 2) + 1
        width = 12                                              # Fixed width at 12 for now (can we look into making it proportional to n_colors?)
        height = rows / 1.75 if rows > 5 else rows / 1.375      # To avoid overlapping axes when there are few rows
        
        fig, axs = plt.subplots(rows, 2, figsize=(width, height), sharex=True, gridspec_kw=dict(wspace=0.1), constrained_layout=True)

        # Set the figure title and subtitle with the palette group key and description
        fig.suptitle(f"* Matplolib {palette_group_key} colormaps (palettes) - {len(palette_group_list)} - (n_colors = {n_colors - 1}) *\n{palette_group_desc}",
                    fontsize=12, fontweight='medium')

        # Iterate over the axes and palette group to plot each palette                                           
        for ax, pltt in zip(axs.flatten(), palette_group_list):
            try:
                color_list = get_color_hex_list(pltt, n_colors)
                ax.bar(sr.index, sr, color=color_list, width=1, edgecolor='white', linewidth=0.2)
                ax.set_xlim(-0.5, n_colors - 1.5)
                ax.set_ylim(0, 0.1)
                ax.set_title(pltt, loc='left', fontsize=10, fontweight='medium')
            except ValueError:
                err_msg = f"'{pltt}' is not currently a valid Matplotlib palette (cmap)"
                ax.set_title(err_msg, loc='left', fontsize=10, fontweight='medium', color='red')

            ax.set_axis_off()
    else:                                               # Displays a continuous strip of colors
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
            # ax.imshow(gradient, aspect='auto', cmap=colormaps[name])
            ax.imshow(gradient, aspect='auto', cmap=name)
            ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()

    plt.show()
    return fig                  # Return the current figure for further manipulation if needed


## Needed rutines that must be runed here
#   - Register direct and reversed JM_QUALITATIVE_CMAPS
for cmap_name, cmap in JM_QUALITATIVE_CMAPS.items():
    custom_cmap = register_mpl_palette(cmap_name, cmap, len(cmap))      # Get custom_cmap and register direct cmap
    custom_cmap_r = custom_cmap.reversed()                              # Build the reversed cmap (default name: ... + '_r')
    plt.colormaps.register(custom_cmap_r)                               # Register reversed cmap


    
if __name__ == "__main__":

    print(sorted(colormaps()))

    # # # Show palettes
    # # # plot_mpl_palettes('names')
    # for key in CMAP_NAMES_BY_CAT.keys():
    #     # print(key)
    #     plot_mpl_palettes(key)
    # # plot_mpl_palettes('Diverging')
    # # plot_mpl_palettes()
    # plot_mpl_palettes('Qualitative', n_colors=6)
    # # plot_mpl_palettes('Single-Hue Sequential', continous=True)
    # # plot_mpl_palettes('Cyclic')
    # #     # plot_mpl_palettes(key, continous=True)

    # # for pltt in ('qualitative','sunny_beach_day', 'Blues', 'colorblind', 'vibrant_summer', 'hot'):
    # #     print(pltt, ':', get_color_hex_list(pltt))
    






