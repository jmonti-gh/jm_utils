"""
To get environment values
==========================

- Hard
- SysOp
- Drivers (ODBC - ...)
- Software
- ...
"""

__author__ = "Jorge Monti"
__version__ = "0.1.0"
__date__ = "2025-05-28"
__status__ = "Development"             # Development, Beta, Production
__description__ = "Utilities I use frequently - Several modules"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-15"




### Libraries
import os
from datetime import datetime
import winreg
import ctypes
from ctypes import wintypes
from tabulate import tabulate
import sys
import pyodbc
import platform
import subprocess
import psutil
import wmi  # Solo en Windows


def bytes_to_gb(bytes_value):
    return round(bytes_value / (1024 ** 3), 2)  


def get_env_data():
    '''
    Obtiene datos del sistema en Windows, Linux y macOS.
    Returns:
        dict: Información del entorno incluyendo hardware y software instalado.
    '''
    now = datetime.now()
    now_str = now.strftime("%d/%m/%Y %H:%M:%S")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    system_name = platform.system()
    env_data = {'date_time': now_str, 'timestamp': timestamp}

    if system_name == "Windows":
        c = wmi.WMI()
        cpu = c.Win32_Processor()[0]
        system = c.Win32_ComputerSystem()[0]
        opsys = c.Win32_OperatingSystem()[0]
        memory = c.Win32_PhysicalMemory()
        gpu = c.Win32_VideoController()[0]
        disk = c.Win32_DiskDrive()[0]
        network = c.Win32_NetworkAdapterConfiguration(IPEnabled=True)
        logical_disks = c.Win32_LogicalDisk()

        env_data.update({
            'pc_name': system.DNSHostName,
            'user': system.UserName,
            'domain': system.Domain,
            'winver': f"{opsys.Caption.replace('Microsoft ', '')} ({opsys.Version}) {opsys.OSArchitecture}",
            'ram_phys_total': bytes_to_gb(sum(int(mem.Capacity) for mem in memory)),
            'processor': cpu.Name,
            'gpu': gpu.Name,
            'gpu_vram': bytes_to_gb(int(gpu.AdapterRAM)) if gpu.AdapterRAM else None,
            'disk_model': disk.Model,
            'disk_size_gb': bytes_to_gb(int(disk.Size)) if disk.Size else None,
            'free_disk_space_gb': sum(bytes_to_gb(int(disk.FreeSpace)) for disk in logical_disks if disk.FreeSpace),
            'network_ip': network[0].IPAddress[0] if network and network[0].IPAddress else None,
            'boot_time': opsys.LastBootUpTime,
            'installed_software': [sw.Name for sw in c.Win32_Product()]
        })

    elif system_name in ["Linux", "Darwin"]:  # Darwin = macOS
        env_data.update({
            'pc_name': platform.node(),
            'user': os.getenv("USER"),
            'os_version': platform.platform(),
            'ram_phys_total': bytes_to_gb(psutil.virtual_memory().total),
            'processor': platform.processor(),
            'gpu': subprocess.run(["lspci"], capture_output=True, text=True).stdout if system_name == "Linux" else "N/A",
            'disk_size_gb': bytes_to_gb(psutil.disk_usage('/').total),
            'free_disk_space_gb': bytes_to_gb(psutil.disk_usage('/').free),
            'network_ip': subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0] if system_name == "Linux" else subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True).stdout.strip(),
            'boot_time': datetime.datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S"),
            'installed_software': subprocess.run(["dpkg", "--list"], capture_output=True, text=True).stdout.split("\n") if system_name == "Linux" else subprocess.run(["system_profiler", "SPApplicationsDataType"], capture_output=True, text=True).stdout.split("\n"),
        })

    return env_data






if __name__ == '__main__':
    env_data = get_env_data()
    for key, value in env_data.items():
        print(key, '->', value)


