# had a weird DNS issue downloading libs from debian & installing some pip packages, setting network to host resolved the issues
#docker build --network=host --no-cache -t portfolio-analysis-app .
docker build --network=host -t portfolio-analysis-app .

docker scout quickview portfolio-analysis-app:latest
docker scout recommendations portfolio-analysis-app:latest