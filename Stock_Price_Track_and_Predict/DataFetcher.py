#%%
import yfinance as yf
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from yahoo_fin import stock_info as si


class DataFetcher:
    """
    Serves as a comprehensive data retrieval and preprocessing utility aimed at gathering financial information about 
    a specified company, along with data on similar companies based on their sector and beta value. 
    """
    def __init__(self, ticker):
        self.ticker = ticker
        self.df, self.actual_ticker = self.get_historical_data(self.ticker)
        #self.tickers_df = self.get_companies_tickers()
        #self.filtered_df = self.get_filtered_companies_by_beta()
    
    def get_ticker(self, company_name):
        """
        This function returns the ticker of a company, given the name.
        Input:
            -company_name: string of lenght > 4
        """
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like"
            " Gecko) Chrome/108.0.0.0 Safari/537.36"
        )
        params = {"q": company_name, "quotes_count": 1, "country": "United States"}
        res = requests.get(
            url=yfinance, params=params, headers={"User-Agent": user_agent}
        )
        data = res.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            company_code = data["quotes"][0]["symbol"]
            return company_code
        else:
            raise AssertionError("Company may not be listed")

    def get_historical_data(self, symbol):
        """
        This function returns a PANDAS dataframe about the price of the company and the stock symbol of the company input.
        The company must be listed in the NYSE.
        
        Input:
            -symbol: string with the name (or ticker) of the company in question; capitalization is not regarded.
        """
        for attempt in range(2):
            try:
                stock_symbol = symbol
                if attempt == 1:
                    stock_symbol = self.get_ticker(stock_symbol)
                    if stock_symbol is None:
                        return None
                stock = yf.Ticker(stock_symbol)
                historical_data = stock.history(period="max")
                historical_data["Volatility"] = historical_data["Close"].rolling(window=30).std()
                historical_data = historical_data.dropna()
                return historical_data, stock_symbol
            except Exception as e:
                if attempt == 0:
                    print(f"Trying fetching for name")
                else:
                    print(
                        f"An error occurred while fetching the historical data: {type(e)}, {e}"
                    )
        return None, None

    def fetch_beta_value(self, symbol, retries=3, backoff_factor=0.5):
        """
        This function returns the 5-year monthly Beta value of a company, given the symbol. 
        
        Notes:
            - Beta is a measure of a stock's volatility in relation to the overall market.

            - Retries with exponential backoff are added to handle intermittent issues.
        """
        site = f"https://finance.yahoo.com/quote/{symbol}/key-statistics?p={symbol}"
        headers = {'User-agent': 'Mozilla/5.0'}
        for attempt in range(retries):
            response = requests.get(url=site, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                beta_table = soup.find('table', {'class': 'W(100%)'})
                if beta_table:
                    rows = beta_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) > 1 and cells[0].get_text() == 'Beta (5Y Monthly)':
                            beta_value = cells[1].get_text()
                            try:
                                return float(beta_value)
                            except ValueError:
                                return 1.0
                return 1.0
            else:
                if response.status_code == 404:
                    return f"Failed to retrieve the page. Status code: 404 for ticker {symbol}"
                time_to_sleep = backoff_factor * (2 ** attempt)
                print(f"Attempt {attempt+1} failed. Retrying in {time_to_sleep} seconds...")
                time.sleep(time_to_sleep)
        return 1.0
    
    def get_company_sector(self, symbol, retries=3, backoff_factor=0.5):
        """
        This function returns the sector of a company, given the name. 
        
        Notes: 
            -sector is NOT industry

            -it was created by amalgamating two functions that don't work anymore 
                from yahoo_fin, and it was PAINFUL

        Retries with exponential backoff are added to handle intermittent issues.
        """
        site = f"https://finance.yahoo.com/quote/{symbol}/profile?p={symbol}"
        headers = {'User-agent': 'Mozilla/5.0'}
        for attempt in range(retries):
            response = requests.get(url=site, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                sector_span = soup.find('span', string='Sector(s)')
                sector = sector_span.find_next_sibling('span', {'class': 'Fw(600)'}).text if sector_span else 'N/A'
                return sector
            else:
                if response.status_code == 404:
                    return f"Failed to retrieve the page. Status code: 404 for ticker {symbol}"
                time_to_sleep = backoff_factor * (2 ** attempt)
                print(f"Attempt {attempt+1} failed. Retrying in {time_to_sleep} seconds...")
                time.sleep(time_to_sleep)
        return f"Failed to retrieve the page after {retries} attempts."
    

    def get_similar_company_data(self):
        """
        This function takes as input the output of get_filtered_companies_by_beta and returns a dictionary
        of PANDAS dataframes with the keys being the ticker of each company in the dictionary returned by get_filtered_companies_by_beta
        """
        try:
            similar_companies = self.filtered_df
            similar_company_data = {}
            for symbol in similar_companies:
                similar_data, _ = self.get_historical_data(symbol)
                similar_company_data[symbol] = similar_data
            return similar_company_data
        except Exception as e:
            print(f"An error occurred: {type(e)}, {e}")

    def fetch_pe_ratio(self, symbol):
        """
        This function returns the PE Ratio of a company, given the name.
        
        Input:
            -symbol: string with the ticker of the company in question
        """
        try:
            ticker = yf.Ticker(symbol)
            stock_info = ticker.info
            pe_ratio = stock_info.get("trailingPE", 0.0)
            return pe_ratio
        except Exception as e:
            print(f"An error occurred while fetching the P/E ratio for {symbol}: {e}")
            return 0.0

    def get_companies_tickers(self, workers = 5):
        """
        This function uses the yahoo_fin package to scrape all the ticker symbols
        currently listed on the NASDAQ and organizes them by sector.
        """
        sector_ticker_dict = {}
        try:
            nasdaq_tickers = si.tickers_nasdaq()
            # Create a thread pool executor
            with ThreadPoolExecutor(max_workers = workers) as executor:
                # Submit all the get_company_sector tasks to the executor
                future_to_ticker = {executor.submit(self.get_company_sector, ticker): ticker for ticker in nasdaq_tickers}
                
                # Process the results as they become available
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        sector = future.result()
                        if sector:
                            if sector not in sector_ticker_dict:
                                sector_ticker_dict[sector] = []
                            sector_ticker_dict[sector].append(ticker)
                            #print(f"{ticker} has been added successfully")
                        else:
                            print(f"Could not fetch sector for {ticker}")
                    except Exception as e:
                        print(f"An error occurred while fetching info for {ticker}: {type(e)}, {e}")
            return sector_ticker_dict
        except Exception as e:
            print(f"An error occurred while fetching NASDAQ tickers: {type(e)}, {e}")
            return {}
       

    def get_filtered_companies_by_beta(self):
        """
        This function takes the output of get_companies_ticker, selects the companies within the same sector as the input's sector and removes the ones with difference in Beta value > 0.1
        """
        sector_ticker_dict = self.tickers_df
        target_beta = self.fetch_beta_value(self.actual_ticker)
        target_sector = self.get_company_sector(self.actual_ticker)
        filtered_companies = {}
        for ticker in sector_ticker_dict:
            beta = self.fetch_beta_value(ticker)
            
            if not isinstance(beta, float):
                print(f"Failed to fetch beta value for ticker {ticker}")
                continue  # Skip this ticker

            if abs(beta - target_beta) <= 0.1:  # Change this to increase tolerance
                filtered_companies[ticker] = beta
        
        return filtered_companies

    def get_pe_multiplier(self):
        """
        This function returns a multiplier given by dividing the average pe of the companies within the sector of the company given by the beta value of the company
        It's only used by Linear Regression, but I kept it here for cleanliness (or lack thereof)        
        """
        try:
            target_pe = self.fetch_pe_ratio(self.actual_ticker)
            portfolio = self.filtered_df
            portfolio_pe = {
                symbol: self.fetch_pe_ratio(symbol) for symbol in portfolio.keys()
            }
            avg_pe = sum(portfolio_pe.values()) / len(portfolio_pe)
            multiplier = avg_pe / target_pe
            return multiplier
        except Exception as e:
            print(f"An error has occurred: {type(e)}, {e}")

# %%
