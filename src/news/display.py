import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain

# tried install lib per thread below, couldn't find python-magic-bin, so installed python-magic instead
# https://stackoverflow.com/questions/76247540/loading-data-using-unstructuredurlloader-of-langchain-halts-with-tp-num-c-bufpip 

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import yfinance as yf
import requests
from stqdm import stqdm

import time

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import config as config

GPT_3_5_TOKEN_LIMIT = 4096

def display_news_analysis(portfolio_summary):
    input_container = st.container()
    output_container = st.container()
    st.caption("*Search & Summarize: Uses Langchaain, OpenAI APIs and Serper or Alpha Vantage, to search the web for news and summarize search results.*")
    
    with input_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.form(key='search_form'):
                if not config.check_for_api_key('openai'):
                    label = "Enter [OpenAI API Key](https://platform.openai.com/account/api-keys) to interpret news"
                    temp_key = st.text_input(label, value=config.get_api_key('openai'), type="password")
                    if temp_key:
                        config.set_api_key('openai', temp_key)
                        
                # TODO: add Bing, Google to this list - and perhaps a multi-select?
                st.write("Alpha Vantage is free, targeted financial news, Serper is paid, open internet search")
                st.radio("News Source", ['Alpha Vantage', 'Serper'], index=0, key='news_source')
                
                if st.session_state.news_source == 'Alpha Vantage (free)':
                    if not config.check_for_api_key('alpha_vantage'):
                        label = "Enter [Alpha Vantage API Key](https://www.alphavantage.co/) to retrieve news and sentiment from the internet (free service)"
                        temp_key = st.text_input(label, value=config.get_api_key('serper'), type="password")
                        if temp_key:
                            config.set_api_key('alpha_vantage', temp_key)
                else:
                    if not config.check_for_api_key('serper'):
                        label = "Enter [Serper API Key](https://serper.dev/api-key) to retrieve news from the internet (paid service)"
                        temp_key = st.text_input(label, value=config.get_api_key('serper'), type="password")
                        if temp_key:
                            config.set_api_key('serper', temp_key)
                        
                        
                st.write("Leverage OpenAI's GPT-3 to interpret news and summarize it, currently experiencing some hangs, so be aware")   
                search_and_summarize_w_openai = st.form_submit_button("Search & Summarize with OpenAI")
        with col2:
            stop_if_hung = st.button("Stop if Hung")
        
    # TODO: find a way to paint the news as it's being retrieved as seems to hang sometimes - maybe a placeholder per ticker?
    # TODO: also retrieve news that we don't have yet first, then update the ones we do have
    with output_container:
        col1, col2, col3 = st.columns(3)
        if config.check_for_api_key('openai') and (config.check_for_api_key('serper') or config.check_for_api_key('alpha_vantage')):
            if search_and_summarize_w_openai:
                with st.spinner("Searching & Summarizing..."):
                    n_cores = multiprocessing.cpu_count()
                    n_tickers = len(portfolio_summary['tickers'])
                    with stqdm(total=n_tickers) as progress_bar:
                        with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_cores, n_tickers)) as executor:
                            futures = []
                            for ticker in portfolio_summary['tickers']:
                                # Get news for ticker and store it in session state
                                future = executor.submit(get_news_for_ticker_and_analyze, ticker, st.session_state.news_source)
                                futures.append(future)
                                
                            for future in concurrent.futures.as_completed(futures):
                                news_for_ticker, news_results = future.result()
                                logger.debug(f"Process returned news for: {news_for_ticker}")
                                st.session_state[f'{news_for_ticker}_news_and_analysis'] = news_results
                                progress_bar.update(1)
                                
        half_index = int(len(portfolio_summary['tickers']) / 2)
        with col1:
            # If session state contains news data, display it
            for ticker in portfolio_summary['tickers'][half_index:]:
                display_news_for_ticker(ticker)
                                
        with col2:
            for ticker in portfolio_summary['tickers'][:half_index]:
                display_news_for_ticker(ticker)


# result limit based on rate limits of the LLM model - if run locally, can increase this
def get_news_for_ticker_and_analyze(ticker, news_source, n_search_results=3):
    news_results = []
    try:
        logger.debug(f"Getting news for {ticker}")
        company = yf.Ticker(ticker).info['longName']
        
        # prioritize news from alpha vantage as it includes sentiment - TODO: add sentiment score processing & a toggle for news source?
        if news_source == 'Alpha Vantage (free)':
            result_dict = get_news_and_sentiment_from_alpha_vantage(ticker, n_search_results)
        else:
            result_dict = get_news_from_serper(ticker, company, n_search_results)

        if not result_dict or not result_dict['news']:
            logger.error(f"No search results for: {ticker}.")
        else:
            # Load URL data from the news search
            for i, item in zip(range(n_search_results), result_dict['news']):
                try:
                    logger.debug(f'{ticker}: processing news item {i} for company {company} from link {item["link"]}')
                    # TODO: appears to hang sometimes...
                    loader = UnstructuredURLLoader(urls=[item['link']], continue_on_failure=False)
                    data = loader.load()
                    logger.debug(f'{ticker}: done processing news item {i} for company {company} from link {item["link"]}')
                                   
                    summary = "No summary available"     
                    # Truncate the data to 4096 characters
                    if isinstance(data, list):
                        for i, element in enumerate(data):
                            # If the element is a Document object, extract and truncate the text
                            #logger.debug(f"Element {i} is type: {type(element)}")
                            if isinstance(element, Document):
                                #logger.debug(f"Element {i} is a Document object\n{element}")
                                element.page_content = element.page_content[:GPT_3_5_TOKEN_LIMIT]
                                #logger.debug(f"Truncated data: {data}")
                                break
                            else:
                                logger.debug(f"Element {i} is not a Document object\n{element}")
            
                        # to help with rate limiting
                        time.sleep(1)
                        
                        # Initialize the ChatOpenAI module, load and run the summarize chain
                        llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=config.get_api_key('openai'))
                        chain = load_summarize_chain(llm, chain_type="map_reduce")
                        summary = chain.run(data)

                    news_results.append({'title': item['title'], 'link': item['link'], 'summary': summary})
                except Exception as e:
                    news_results.append({'title': item['title'], 'link': item['link'], 'summary': 'Error while summarizing'})
                    logger.error(f"Exception summarizing news about {company} w/ticker {ticker}: {e}")
                    
    except Exception as e:
        logger.error(f"Exception searching for news about {company} w/ticker {ticker}: {e}")

    logger.debug(f"Completed getting news for {ticker}")
        
    return (ticker, news_results)

def get_news_from_serper(ticker, company, n_search_restuls=3):
    result_dict = {}
    try:
        search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=config.get_api_key('serper'))
        search_query = f"financial news about {company} or {ticker}"
        logger.debug(f"Search query: {search_query}")
        
        # search hangs sometimes... trying sleep
        result_dict = search.results(search_query)
        logger.debug(f"Search results returned for {search_query}, {result_dict.keys()}")
        
    except Exception as e:
        logger.error(f"Exception searching for news about {company} w/ticker {ticker} with Serper: {e}")
        
    return result_dict

def get_news_and_sentiment_from_alpha_vantage(ticker, n_search_results=3):
    articles = {}
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": config.get_api_key('alpha_vantage'),
            "limit": n_search_results
        }

        response = requests.get(url, params=params)
        data = response.json()

        articles['news'] = []
        for item in data['feed'][:n_search_results]:
            article = {
                "title": item['title'],
                "link": item['url'],
                "summary": item['summary'],
                "sentiment_score": item['overall_sentiment_score'],
                "sentiment_label": item['overall_sentiment_label'],
            }
            articles['news'].append(article)
    
    except Exception as e:
        logger.error(f"Exception searching for news about ticker {ticker} with Alpha Vantage: {e}")
        
    return articles

def display_news_for_ticker(ticker):
    if f'{ticker}_news_and_analysis' in st.session_state:
        with st.expander(f"News about {ticker}"):
            for news_item in st.session_state[f'{ticker}_news_and_analysis']:
                st.markdown(f"## {news_item['title']}")
                
                if news_item['summary'] != 'Error while summarizing' and news_item['summary'] != 'No summary available':
                    st.markdown(f"### {news_item['summary']}")
                
                if news_item['sentiment_label'] and news_item['sentiment_score']:
                    st.markdown(f"#### Sentiment: {news_item['sentiment_label']} ({news_item['sentiment_score']})")
                st.markdown(f"#### [Read more]({news_item['link']})")
    else:
        st.markdown(f"## No news found for {ticker}")
        
