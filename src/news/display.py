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
    st.markdown('<p style="color:red;">WIP: Langchain Unstructured URL loader can hang, pressing a button will clear browser although a process can still be hung and hold the PORT</p>',unsafe_allow_html=True)
    input_container = st.container()
    output_container = st.container()
    st.caption("*Search & Summarize: Uses Langchaain, Serper & OpenAI APIs, to search the web for news and summarizes each search result.*")
    
    with input_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            if not config.check_for_api_key('openai'):
                label = "Enter [OpenAI API Key](https://platform.openai.com/account/api-keys) to interpret news"
                temp_key = st.text_input(label, value=config.get_api_key('openai'), type="password")
                if temp_key:
                    config.set_api_key('openai', temp_key)
                    
            if not config.check_for_api_key('serper'):
                label = "Enter [Serper API Key](https://serper.dev/api-key) to retrieve news from the internet"
                temp_key = st.text_input(label, value=config.get_api_key('serper'), type="password")
                if temp_key:
                    config.set_api_key('serper', temp_key)
                    
            search_and_summarize = st.button("Search & Summarize")
        
    # TODO: find a way to paint the news as it's being retrieved as seems to hang sometimes - maybe a placeholder per ticker?
    # TODO: also retrieve news that we don't have yet first, then update the ones we do have
    with output_container:
        if config.check_for_api_key('openai') and config.check_for_api_key('serper'):
            if search_and_summarize:
                with st.spinner("Searching & Summarizing..."):
                    n_cores = multiprocessing.cpu_count()
                    n_tickers = len(portfolio_summary['tickers'])
                    with stqdm(total=n_tickers) as progress_bar:
                        with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_cores, n_tickers)) as executor:
                            futures = []
                            for ticker in portfolio_summary['tickers']:
                                # Get news for ticker and store it in session state
                                future = executor.submit(get_news_for_ticker, ticker)
                                futures.append(future)
                                
                            for future in concurrent.futures.as_completed(futures):
                                news_for_ticker, news_results = future.result()
                                logger.debug(f"Process returned news for: {news_for_ticker}")
                                st.session_state[f'{news_for_ticker}_news'] = news_results
                                progress_bar.update(1)

            # If session state contains news data, display it
            for ticker in portfolio_summary['tickers']:
                if f'{ticker}_news' in st.session_state:
                    with st.expander(f"News about {ticker}"):
                        for news_item in st.session_state[f'{ticker}_news']:
                            st.write(f"Title: {news_item['title']}\n\nLink: {news_item['link']}\n\n")
                            if news_item['summary'] != 'Error while summarizing' and news_item['summary'] != 'No summary available':
                                st.success(f"Summary: {news_item['summary']}")

def get_news_for_ticker(ticker, n_search_restuls=3):
    news_results = []
    try:
        logger.debug(f"Getting news for {ticker}")
        company = yf.Ticker(ticker).info['longName']
        
        # TODO: try bing search or vanilla google - plus might be free vs Serper?
        # Show the top X relevant news articles from the previous week using Google Serper API
        search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=config.get_api_key('serper'))
        search_query = f"financial news about {company} or {ticker}"
        logger.debug(f"Search query: {search_query}")
        
        # search hangs sometimes... trying sleep
        result_dict = search.results(search_query)
        logger.debug(f"Search results returned for {search_query}, {result_dict.keys()}")

        if not result_dict['news']:
            logger.error(f"No search results for: {search_query}.")
        else:
            # Load URL data from the top X news search results
            for i, item in zip(range(n_search_restuls), result_dict['news']):
                try:
                    logger.debug(f'processing news item {i} for company {company} and ticker {ticker} from link {item["link"]}')
                    # TODO: appears to hang sometimes...
                    loader = UnstructuredURLLoader(urls=[item['link']], continue_on_failure=False)
                    data = loader.load()
                    logger.debug(f'done processing news item {i} for company {company} and ticker {ticker} from link {item["link"]}')
                                   
                    summary = "No summary available"     
                    # Truncate the data to 4096 characters
                    if isinstance(data, list):
                        for i, element in enumerate(data):
                            # If the element is a Document object, extract and truncate the text
                            logger.debug(f"Element {i} is type: {type(element)}")
                            if isinstance(element, Document):
                                #logger.debug(f"Element {i} is a Document object\n{element}")
                                element.page_content = element.page_content[:GPT_3_5_TOKEN_LIMIT]
                                #logger.debug(f"Truncated data: {data}")
                                break
                            else:
                                logger.debug(f"Element {i} is not a Document object\n{element}")
            
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
