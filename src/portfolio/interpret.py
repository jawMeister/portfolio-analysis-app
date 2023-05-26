import streamlit as st
import openai

import src.session as session

# TODO: move this to streamlit config
from config import OPENAI_API_KEY, FRED_API_KEY

def openai_interpret_portfolio_summary(portfolio_summary):
    
    #print(f"openai api key: {openai_api_key}")
    
    if session.check_for_openai_api_key():
        
        tickers = []
        for ticker in portfolio_summary["stock_data"].columns:
            if portfolio_summary["weights"][ticker] > 0:
                tickers.append(ticker)

        sharpe_q = f"What is a Sharpe Ratio and what does a Sharpe Ratio of {portfolio_summary['sharpe_ratio']:.2f} imply about a portfolio? Is it good or bad and what are your recommendations for adjustments?"
        sortino_q = f"What is a Sortino Ratio and what does a Sortino Ratio of {portfolio_summary['sortino_ratio']:.2f} imply about a portfolio? Is it good or bad and what are your recommendations for adjustments?"
        cvar_q = f"What is a CVaR and what does a CVaR of {portfolio_summary['cvar']:.2f} for this portfolio imply? Is it good or bad and what are your recommendations for adjustments?"
        treynor_q = "" #f"What is the Treynor Ratio and what does a Treynor Ratio of {portfolio_summary['treynor_ratio']:.2f} for this portfolio imply? Is it good or bad and what are your recommendations for adjustments?"
        data_to_share = f"The portfolio is composed of {tickers} weighted as {portfolio_summary['weights']}. \
                        The portfolio has a total return of {portfolio_summary['portfolio_return']} and a volatility of {portfolio_summary['volatility']}."
        all_q = f"Taken together, what is your analysis of a portfolio with these statistics and what are your suggestions for optimizing potential returns?"

        question = sharpe_q + sortino_q + cvar_q + treynor_q + data_to_share + all_q
        
        print(f"question: {question}")
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}])
        
        return chat_completion.choices[0].message.content