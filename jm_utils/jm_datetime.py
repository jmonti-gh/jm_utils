"""
To get useful values of datetime
===============================

- last_day_of_month(yyyymm: str, format: str):
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
import datetime as dtm
import re

def datetime_YMD_type_conv(datetime: dtm.datetime, output_type: str='string'):
    '''
    format:
        'string':
        'date':
        'datetime':
    '''
    if output_type not in ['string', 'date', 'datetime']:
        raise ValueError(f"output_type solemente puede tomar los valores 'string', 'date', 'datetime', no {output_type}")

    if not isinstance(datetime, dtm.datetime):
        raise TypeError(f"{datetime} debe ser del tipo 'datetime.date' y no {type(datetime)}")
    
    if output_type == 'string':
        return datetime.strftime('%Y%m%d')
    elif output_type == 'date':
        return datetime.date()
    elif output_type == 'datetime':
        return datetime


def last_day_of_month(yyyymm: str, output_type: str='string'):
    '''
    Get last day of the month in yyyymm

    yyyymm: str         -> Valid every year
    output_type: str    -> 'string' (default), 'date', 'datetime'
    '''
    try:
        yyyymm_as_dtm = dtm.datetime.strptime(yyyymm, '%Y%m')
    except ValueError:
        raise ValueError(f"El valor '{yyyymm}' no tiene el formato 'YYYYMM' válido.")
                      
    # day 25 exists in every month. 9 days later, it's always next month
    nxt_mnth = yyyymm_as_dtm.replace(day=25) + dtm.timedelta(days=9)
    # subtracting the number of days of nxt_mnth we'll get last day of original month
    last_day_mnth_dtm = nxt_mnth - dtm.timedelta(days=nxt_mnth.day)
    return datetime_YMD_type_conv(last_day_mnth_dtm, output_type)


def last_day_month_past_year(yyyymm: str, output_type: str='string'):
    yyyymm_as_date = dtm.datetime.strptime(yyyymm, '%Y%m').date()
    yyyymm_past_year = yyyymm_as_date.replace(year= yyyymm_as_date.year - 1)
    ym_past_year_str = dtm.datetime.strftime(yyyymm_past_year, '%Y%m')
    return last_day_of_month(ym_past_year_str, output_type)


def date_str_to_dt(date_str: str) -> dtm.datetime:
    """ Convierte una fecha en formato string a datetime, manejando múltiples formatos y limpiando la entrada. """
    
    # Limpiar la cadena para eliminar caracteres no numéricos salvo separadores permitidos
    date_str = re.sub(r"[^\d\-/.:\s]", "", date_str.strip())

    # Definir formatos compatibles
    formatos = [
        "%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y",
        "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y"
    ]
    
    # Intentar convertir con cada formato
    for formato in formatos:
        try:
            date_dt = dtm.datetime.strptime(date_str, formato)
            return date_dt
        except ValueError:
            continue
    
    # Si ningún formato funcionó, lanzar un error detallado
    raise ValueError(f"Formato de fecha no reconocido o inválido: {date_str}")

def arg_now() -> str:
    """Argentinian date-time usage"""
    return dtm.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def is_business_day(date: str) -> bool:
    """Verifica si una fecha es día hábil (lunes a viernes)"""
    if isinstance(date, str):
        date = date_str_to_dt(date)
    return date.weekday() < 5

def add_business_days(initial_date, days):
    """Agrega días hábiles a una fecha"""
    if isinstance(initial_date, str):
        initial_date = date_str_to_dt(initial_date)
    
    actual_date = initial_date
    added_days = 0
    
    while added_days < days:
        actual_date += dtm.timedelta(days=1)
        if is_business_day(actual_date):
            added_days += 1
    
    return actual_date



if __name__ == '__main__':

    # date = '202502'
    
    # # last_day_of_month() AND last_day_month_past_year()
    # for format in 'string', 'date', 'datetime':
    #     output = last_day_of_month(date, format)
    #     print('Last day of Month:', output, type(output))
    #     o2 = last_day_month_past_year(date, format)
    #     print('Last day of Month past Year:', o2, type(o2))
    #     print()

    # Diff tries date_str_to_dt(fecha_str)
    # for ds in '24/1/2025', '2025-10-24', '24/10/2025':
    #     out = date_str_to_dt(ds)
    #     print(out, type(out))



    # Ejemplo de uso
    fecha_str = "13/06/2025"
    fecha_dt = date_str_to_dt(fecha_str)

    # Aplicar timedelta (sumar 3 días)
    nueva_fecha = fecha_dt + dtm.timedelta(days=3)

    # Obtener el día de la semana
    dia_semana = nueva_fecha.weekday()  # 0=Lunes, 6=Domingo

    print(f"Fecha original: {fecha_dt}")
    print(f"Fecha modificada: {nueva_fecha}")
    print(f"Día de la semana (0=Lunes, 6=Domingo): {dia_semana}")

    print()
    ahora = arg_now()
    print(ahora, type(ahora))
    print(is_business_day('20251024'))
    print(add_business_days('20251024', 10))







