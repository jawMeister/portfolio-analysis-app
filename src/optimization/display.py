import streamlit as st

def display_portfolio_optimization(portfolio_summary):
    st.container()
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.write("TODO: portfolio optimization")
        st.write("The concepts of robust optimization, Bayesian methods, and resampling methods can be applied in the context of \
                portfolio optimization to address the limitations of the traditional mean-variance optimization approach which can be overly sensitive to \
                input estimation errors.")
        st.write("It's essential to remember that no single method will always outperform the others in all scenarios. Each method has its own assumptions and trade-offs, " \
                "and the best approach can depend on various factors, including the number and diversity of assets in the portfolio, the investor's risk tolerance, and the reliability of the input estimates.")
        
        with st.expander("Robust Optimization: This method is designed to find solutions that perform well across a range of scenarios, not just a single 'expected' scenario."):
            st.write("It builds in a degree of 'immunity' against estimation errors. It might involve, for example, minimizing the portfolio's worst-case scenario rather than its expected risk.")
            st.write("Robust optimization often results in more diversified portfolios.")
        
        with st.expander("Bayesian Methods: Bayesian methods incorporate prior beliefs about parameters and then update these beliefs based on observed data."):
            st.write("This can help mitigate the impact of estimation error in inputs such as expected returns, variances, and covariances.")
            st.write("It involves developing a prior distribution for the inputs and then updating this distribution given the observed data to get a posterior distribution.")
            st.write("The portfolio optimization can then be performed using these posterior distributions.")
        
        with st.expander("Resampling Methods: Resampling involves generating many possible sets of inputs (like returns, variances, and covariances), optimizing the portfolio for each set, and then averaging over these portfolios to get the final portfolio."):
            st.write("This can help to mitigate the impact of input estimation errors because it averages over many different scenarios.")
            st.write("The most common approach for resampling in portfolio optimization is the Bootstrap method, where multiple subsamples of the historical returns data are created (with replacement), and each subsample is used to estimate the inputs for optimization.")

