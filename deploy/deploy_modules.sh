# !/bin/bash

# Build and publish the pipeline packages

# Go to the data_ingest directory
cd ../data-ingest
poetry build
poetry publish -r my-pypi

cd ../model-training
poetry build
poetry publish -r my-pypi

cd ../model-inference
poetry build
poetry publish -r my-pypi