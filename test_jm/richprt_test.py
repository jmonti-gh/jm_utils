# Standard Libs
from datetime import datetime

# Third-Party Libs
from rich.console import Console
from rich.text import Text


console = Console()


def prt_state(state='info', msg='Default message', color='white', mark='?'):
    """
    Prints a styled console message with a state indicator using customizable colors and symbols.

    This function displays a formatted message in the terminal using the Rich library. It supports
    predefined states ('info', 'ok', 'warning', 'error') that determine the symbol and color of the output.
    Custom styles can also be applied using the 'color' and 'mark' parameters.

    Parameters:
        state (str): The type of message to display. Predefined options are:
            - 'info'     : Informational message (default)
            - 'ok'       : Success message
            - 'warning'  : Warning message
            - 'error'    : Error message

        msg (str): The message text to print. Defaults to 'Default message'.

        color (str): Fallback color for custom states or when style customization is needed.
            Defaults to 'white'.

        mark (str): Fallback symbol to use if the state is not recognized or for custom states.
            Defaults to '?'.

    Raises:
        None: This function does not raise exceptions but relies on Rich for styling and output.

    Examples:
        >>> prt_state('info', 'Loading configuration...')
         i [INFO] > Loading configuration...

        >>> prt_state('ok', 'Operation completed successfully.')
         # [OK] > Operation completed successfully.

        >>> prt_state('warning', 'Disk space low.', mark='⚠️')
         ⚠️ [WARNING] > Disk space low.

        >>> prt_state('custom', 'Custom status.', color='magenta', mark='*')
         * [CUSTOM] > Custom status.
    """
    mark_colors = {
        'info': ['i', 'dodger_blue1', 'dodger_blue3', 'blue'],
        'ok': ['#', 'spring_green2', 'spring_green2', 'green'],
        'warning': ['!', 'yellow1', 'yellow3', 'yellow'],
        'error': ['X', 'red1', 'red2', 'red']
    }

    default = [mark, color, color, color]

    txt = Text(mark_colors.get(state, default)[0], style=mark_colors.get(state, default)[1])
    txt.append(f" [{state.upper()}] -", style=mark_colors.get(state, default)[2])
    txt.append(f" {msg} ", style=mark_colors.get(state, default)[3])

    console.print(txt)


def prt_prg_title0(prg, title, fmt_title=False, log='',
                   colors=['green', 'magenta', 'yellow', 'cyan']):
    ''' Prints a title with the program name, title, and log information.
        prg: Program name
        title: Title of the section
        log: Log information (default is 'No log')
    '''

    timestamp = datetime.now().strftime('%b %d %H:%M:%S')
   
    if fmt_title:
        title = title.title()

    txt = Text(f"{timestamp} > ", style=colors[0])
    txt.append('Iniciando la ejecución de: ', style='white')
    txt.append(f"{prg} \n", style=colors[1])
    txt.append(f">>>  {title}  <<<", style=colors[3])

    if log:
        txt.append('\n\t\t> ', style=colors[0])
        txt.append('Log: ', style='white')
        txt.append(f"{log} \n", colors[2])

    console.print(txt)

    
def prt_prg_title(prg, title, fmt_title=False, log='', version=1,
                  colors=['green', 'magenta', 'yellow', 'cyan']):
    """
    Prints a styled program title header with timestamp, program name, title, and optional log.

    This function prints a formatted header for a program or section using Rich text styling.
    It can display a timestamp, program name, title, and log information in different colors.
    The layout adapts based on the `version` and `fmt_title` flags.

    Parameters:
        prg (str): Name of the program or script. Displayed prominently in the header.

        title (str): Title of the current section or operation being executed.

        fmt_title (bool, optional): If True, capitalizes each word in the title using `.title()`.
            Defaults to False.

        log (str, optional): Optional log identifier or file path. If provided, it will be displayed.
            Defaults to empty string (no log shown).

        version (int, optional): Controls the format of the output:
            - 1 (default): Compact header with timestamp, program name, log info, and underline-style title.
            - 0: More verbose style with separated lines and emphasis on the title.

        colors (list of str, optional): List of color names/styles to use for different parts:
            - [0]: Timestamp color
            - [1]: Program name color
            - [2]: Log text color
            - [3]: Title text color
            Defaults to ['green', 'magenta', 'yellow', 'cyan'].

    Raises:
        None: This function does not raise exceptions.

    Examples:
        >>> prt_prg_title('my_script', 'Processing Data', log='run.log')
         Jan 01 12:00:00 - my_script - Log: run.log
        -------------------------------
         Processing Data
        -------------------------------

        >>> prt_prg_title('my_script', 'initial setup', fmt_title=True, version=0)
         Jan 01 12:00:00 > Iniciando la ejecución de: my_script
         >>>  Initial Setup  <<<
    """

    timestamp_str = str(datetime.now().strftime('%b %d %H:%M:%S'))

    if fmt_title:
        title = title.title()

    if version:
        len_data = len(timestamp_str) + len(prg) + len(log) + 6
        len_line = len(title) + 3
        if len_data > len_line:
            len_line = len_data

        txt = Text(f"{timestamp_str}", style=colors[0])
        txt.append(f" - ", style='white')
        txt.append(f"{prg}", style=colors[1])
        
        if log:
            txt.append(f" - Log: ", style='white')
            txt.append(log, colors[2])

        txt.append(f"\n{'-' * len_line}\n", style='white')
        txt.append(f" {title}", style=colors[3])
        txt.append(f"\n{'-' * len_line}\n", style='white')

    else:
        txt = Text(f"{timestamp_str} > ", style=colors[0])
        txt.append('Iniciando la ejecución de: ', style='white')
        txt.append(f"{prg} \n", style=colors[1])
        txt.append(f">>>  {title}  <<<", style=colors[3])

        if log:
            txt.append('\n\t\t> ', style=colors[0])
            txt.append('Log: ', style='white')
            txt.append(f"{log} \n", colors[2])

    console.print(txt)


if __name__ == '__main__':
    
    # # prt_state()
    # for state in ('info', 'ok', 'warning', 'error'):
    #     msg = f"Este es un mensaje de __{state}__\n"
    #     prt_state(state=state, msg=msg)

    # # prt_prg_title0()
    # prt_prg_title0('odbcinfo', 'Controladores ODBC instalados en su sistema', fmt_title=True, log='odbcinfo.log')
 
    # prt_prg_title()
    # prt_prg_title('odbcinfo', 'Controladores ODBC instalados en su sistema', fmt_title=True, log='odbcinfo.log')
    prt_prg_title('odbcinfo', 'Controladores ODBC instalados en su sistema', fmt_title=True)

    print('aca empieza los msgcvcv')
    print('## 2.- aca empieza los msgcvcv')
    input('.....')

