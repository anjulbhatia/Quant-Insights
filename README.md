## Folder Structure

RT2 *(root/master)* (
    ./.streamlit *(dir)*
        ./config.toml
    ./styles *(dir)*
        ./style.css
        ./sidebar.css
    ./dashboard_components *(dir)*
        ./sidebar.py
        ./dashboard.py
    ./data *(dir)*
        ./data.csv
    ./notebooks *(dir)*
        ./data_cleaning.ipynb
        ./data_visualization.ipynb
    ./RT2 *(dir)*
        ./__utils__.py
        ./TradeModel.py
        ./TradeModel.ipynb
        ./MLmodels.py
        ./MLmodels.ipynb
        ./Backtest.py
        ./Backtest.ipynb
        ./TradeStrategies.py
        ./TradeStrategies.ipynb
)

## Tasks for the day - 23/07/2024
- [ ] Build the RT2 Lib using yfinance/pandas_datadeader
- [ ] Create a basic CLI to init the library
- [ ] Create a basic Streamlit dashboard for it

