import boto3
import os
import json
from botocore.exceptions import NoCredentialsError, BotoCoreError, ClientError

# TODO: refactor down the road to try and us lambda for simulations if available
def run_portfolio_simulations(portfolio_summary, n_simulations, distribution="T-Distribution"):
    start_time = time.time()
    
    if is_lambda_available():
        print("AWS Lambda available, using Lambda for simulations.")
        results = run_simulations_lambda(portfolio_summary, n_simulations, distribution)
    else:
        print("AWS Lambda not available, using local multiprocessing for simulations.")
        n_cores = multiprocessing.cpu_count()
        print(f"Number of cores: {n_cores}")
        
        results = []
        with stqdm(total=n_simulations) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = []
                for _ in range(n_simulations):
                    future = executor.submit(simulate_portfolio, portfolio_summary, distribution)
                    futures.append(future)
                    
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
    end_time = time.time()
    print("The main sim loop took", end_time - start_time, "seconds to run")
    
    return results


def is_lambda_available():
    """
    This function attempts to retrieve AWS credentials. If it succeeds, it assumes
    that Lambda is available. In practice, you might want to replace this with a
    more sophisticated check.
    """
    try:
        boto3.Session().get_credentials().get_frozen_credentials()
        return True
    except (NoCredentialsError, BotoCoreError):
        return False

def run_simulations_lambda(portfolio_summary, n_simulations, distribution):
    lambda_client = boto3.client('lambda')

    results = []
    with stqdm(total=n_simulations) as pbar:
        for _ in range(n_simulations):
            response = lambda_client.invoke(
                FunctionName='YourLambdaFunctionName',  # replace with your Lambda function name
                InvocationType='RequestResponse',
                Payload=json.dumps({
                    "portfolio_summary": portfolio_summary,
                    "distribution": distribution
                })
            )
            result = json.loads(response['Payload'].read().decode())
            results.append(result)
            pbar.update(1)
    return results