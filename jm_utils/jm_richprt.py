'''
jm_richprt
'''
# copilot

__version__ = "0.1.0"
__description__ = "Utilities I use frequently - Several modules"
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-15"


# Standard Libs
from datetime import datetime

# Third-Party Libs
from rich.console import Console


console = Console()


def prtmsg(kind='inf', msg='Default message', mark=''):
    ''' Prints a message with a specific kind (ok, error, warning, info) and an optional mark.
        kind: 'ok', 'err', 'warn', 'inf'
    '''
    if kind not in ('ok', 'err', 'warn', 'inf'):
        raise ValueError(f"Invalid kind: {kind}. Must be one of 'ok', 'err', 'warn', 'info'.")
    
    if kind == "ok":
        if not mark:
            mark = '#'
        console.print(f"\n [bold green]{mark}[/bold green] [green]OK >[/green]  {msg}\n")
    elif kind == "err":
        if not mark:
            mark = 'X'
        console.print(f"\n [bold red]{mark}[/bold red] [red]ERROR >[/red]  {msg}\n")
    elif kind == "warn":
        if not mark:
            mark = '!'
        console.print(f"\n [bold yellow]{mark}[/bold yellow] [yellow]Warning >[/yellow]  {msg}\n")
    elif kind == "inf":
        if not mark:
            mark = 'i'
        console.print(f"\n [bold blue]{mark}[/bold blue] [blue]Info >[/blue]  {msg}\n")


def prt_title_log(prg, title, log='No log'):
    ''' Prints a title with the program name, title, and log information.
        prg: Program name
        title: Title of the section
        log: Log information (default is 'No log')
    '''
    spruce = f"[green]~~~~~[/green]"
    timestamp = datetime.now().strftime('%b %d %H:%M:%S')
    console.print(
        f'''\n[green]{timestamp} >[/green] Iniciando la ejecución de: [magenta]{prg}[/magenta]

{spruce} {title} {spruce}
                [green]>[/green] LOG: [magenta]{log}[/magenta]\n'''
     )
    

def prt_title(title):
    ''' Prints a title with a decorative border.
        title: Title of the section
    '''
    spruce = f"[green]~~~~~[/green]"
    console.print(f"\n{spruce} {title} {spruce}\n")


def enter_to_continue(msg='Press Enter to continue...'):
    ''' Custom input function to pause execution and wait for user input.
        msg: Message to display (default is 'Press Enter to continue...')
    '''
    console.input(f"\n[bold blue]{msg}[/bold blue]")




def demo_prtmsg():
    for knd in 'ok', 'err', 'warn', 'inf':
        print()
        prtmsg(knd)


if __name__ == '__main__':
    demo_prtmsg()
    prt_title_log('jm_richprt.py', 'Demo Title', 'Demo Log')
    prt_title('Demo Title')