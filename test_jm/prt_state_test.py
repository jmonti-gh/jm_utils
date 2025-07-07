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
    txt.append(f" [{state.upper()}] >", style=mark_colors.get(state, default)[2])
    txt.append(f" {msg} ", style=mark_colors.get(state, default)[3])

    console.print(txt)


if __name__ == '__main__':
    
    for state in ('info', 'ok', 'warning', 'error'):
        msg = f"Este es un mensaje de __{state}__\n"
        prt_state(state=state, msg=msg)