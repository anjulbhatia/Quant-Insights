from typing import Union, Literal

class Plotter:
    def __init__(self, ChartType: Literal['plotly','trading_view'] = 'trading_view'):
        if ChartType == 'trading_view':
            pass # Already Installed
        else:
            try:
                import plotly
            except:
                raise ImportError("Cannot import plotly")