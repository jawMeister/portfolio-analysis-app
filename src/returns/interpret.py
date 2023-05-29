import streamlit as st
import openai

import src.session as session

def openai_interpret_montecarlo_simulation(portfolio_summary, number_of_simulations, volatility_distribution):
    
    if session.check_for_openai_api_key():
        openai.api_key = session.get_openai_api_key()
        
        tickers = []
        for ticker in portfolio_summary["stock_data"].columns:
            if portfolio_summary["weights"][ticker] > 0:
                tickers.append(ticker)
                
        portfolio_stats = f"Given a portfolio with a Sharpe Ratio of {portfolio_summary['sharpe_ratio']:.2f}, Sortino Ratio of {portfolio_summary['sortino_ratio']:.2f}, " + \
                            f"CVaR of {portfolio_summary['cvar']:.2f} composed of {tickers} weighted as {portfolio_summary['weights']}," + \
                                "I executed a Monte Carlo Simulation with {number_of_simulations} simulations over {portfolio_summary['years']} years " + \
                                    "to simulate future returns by leveraging a {volatility_distribution} for volatily per asset to establish random returns. "
        
        question = portfolio_stats + \
                    "What is your assessment of this portfolio, " + \
                    "how valid are the assumptions made in the Monte Carlo Simulation and " + \
                    "how should I interpret the simulation results which plot probability densities by year? " + \
                    "What are your recommendations for optimizing potential returns and what other analyses do you recommend for the portfolio? "
                    
        print(f"question: {question}")
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}])
        
        return chat_completion.choices[0].message.content