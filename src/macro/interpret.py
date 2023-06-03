import streamlit as st
import openai

import config as config

def openai_ask_about_macro_economic_factors(portfolio_summary):
    
    if config.check_for_api_key('openai'):
        openai.api_key = config.get_api_key('openai')
        
        tickers = []
        for ticker in portfolio_summary["stock_data"].columns:
            if portfolio_summary["weights"][ticker] > 0:
                tickers.append(ticker)
                
        portfolio_stats = f"Given a portfolio with a Sharpe Ratio of {portfolio_summary['sharpe_ratio']:.2f}, Sortino Ratio of {portfolio_summary['sortino_ratio']:.2f}, " + \
                            f"CVaR of {portfolio_summary['cvar']:.2f} composed of {tickers} weighted as {portfolio_summary['weights']}," + \
                                "I executed a Monte Carlo Simulation over {portfolio_summary['years']} years " + \
                                    "to simulate future returns by leveraging historical volatily per asset to establish random returns. "
        
        question = portfolio_stats + \
                    "As a financial advisor, I am now interested in understanding the impact of macro-economic factors on the portfolio. " + \
                    "What are the top 3 macro-economic factors that may impact the particular assets in this portfolio and how should I explore them?"
                    
        print(f"question: {question}")
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}])
        
        return chat_completion.choices[0].message.content