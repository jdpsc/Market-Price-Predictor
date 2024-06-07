from datetime import datetime

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.dummy_operator import DummyOperator


@dag(
    dag_id="ml_pipeline",
    schedule="0 15 * * *", # Run daily at 15:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data-ingest", "model-training", "model-inference"],
    max_active_runs=1,
)
def ml_pipeline():

    @task.virtualenv(
        task_id="run_data_ingest",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "data_ingest",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=True,
    )
    def run_data_ingest(
        logical_date: str,
        symbol: str,
        days_delay: int,
        interval: str,
        feature_group_version: int,
    ) -> dict:
        """Run the data ingestion pipeline."""

        from datetime import datetime, timedelta

        from data_ingest import utils, pipeline

        logger = utils.get_logger(__name__)

        end_date = logical_date

        logger.info(f"Raw end_date: {end_date}")

        end_date = (
            datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S%z")
            if "." not in end_date
            else datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S.%f%z")
        )
        end_date = end_date.date()
        end_date = end_date + timedelta(days=1)

        # Calculate start_date
        start_date = end_date - timedelta(days=days_delay)

        logger.info(
            f"Running data ingest pipeline from {start_date} to {end_date} at {interval} interval, feature_group_version={feature_group_version}"
        )

        return pipeline.run_pipeline(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            feature_group_version=feature_group_version,
        )

    @task.virtualenv(
        task_id="create_feature_view",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "data_ingest",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def create_feature_view(
        logical_date: str,
        days_delay: int,
        feature_group_version: int,
    ) -> dict:
        """Create a feature view from the feature group."""

        from data_ingest import utils, feature_views
        from datetime import datetime, timedelta

        logger = utils.get_logger(__name__)

        end_date = logical_date

        logger.info(f"Raw end_date: {end_date}")

        # TODO: Use parser to avoid this

        end_date = (
            datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S%z")
            if "." not in end_date
            else datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S.%f%z")
        )
        end_date = end_date.date()
        end_date = end_date + timedelta(days=1)

        # Calculate start_date
        start_date = end_date - timedelta(days=days_delay)

        logger.info(
            f"Creating feature view from {start_date} to {end_date} with feature_group_version={feature_group_version}"
        )

        return feature_views.create_feature_view(
            start_date=start_date,
            end_date=end_date,
            feature_group_version=feature_group_version,
        )

    @task.virtualenv(
        task_id="batch_predict",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "model_inference",
            "model_training",
        ],
        python_version="3.9",
        system_site_packages=False,
    )
    def batch_predict(
        logical_date: str,
        symbol: str,
        feature_view_version: int,
    ):
        """Run the batch prediction pipeline."""

        from model_inference import predict, utils
        from datetime import datetime, timedelta

        logger = utils.get_logger(__name__)

        inference_datetime = (
            datetime.strptime(logical_date, "%Y-%m-%d %H:%M:%S%z")
            if "." not in logical_date
            else datetime.strptime(logical_date, "%Y-%m-%d %H:%M:%S.%f%z")
        )
        inference_datetime = inference_datetime.date()
        inference_datetime = inference_datetime + timedelta(days=1)

        logger.info(
            f"Running batch prediction pipeline for symbol {symbol} at {inference_datetime} with feature_view_version={feature_view_version}"
        )

        predict.run_inference(
            symbol=symbol,
            inference_datetime=inference_datetime,
            feature_view_version=feature_view_version,
        )

    @task.virtualenv(
        task_id="monitoring",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "model_inference",
        ],
        python_version="3.9",
        system_site_packages=False,
    )
    def compute_monitoring(
        logical_date: str,
        symbol: str,
        feature_view_version: int,
    ):
        """Compute monitoring metrics for newly observed data (previous day)."""

        from model_inference import monitoring, utils
        from datetime import datetime

        logger = utils.get_logger(__name__)

        inference_datetime = (
            datetime.strptime(logical_date, "%Y-%m-%d %H:%M:%S%z")
            if "." not in logical_date
            else datetime.strptime(logical_date, "%Y-%m-%d %H:%M:%S.%f%z")
        )
        inference_datetime = inference_datetime.date()

        monitoring.calculate_metrics(
            symbol=symbol,
            inference_datetime=inference_datetime,
            feature_view_version=feature_view_version,
        )

    # Define Airflow variables.
    symbols = list(Variable.get("ml_pipeline_symbols", deserialize_json=True))
    interval = str(Variable.get("ml_pipeline_interval", default_var="1d"))
    days_delay = int(Variable.get("ml_pipeline_days_delay", default_var=14))
    feature_group_version = int(
        Variable.get("ml_pipeline_feature_group_version", default_var=1)
    )
    feature_view_version = int(
        Variable.get("ml_pipeline_feature_view_version", default_var=1)
    )

    logical_date = "{{ dag_run.logical_date }}"

    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    # Run data ingest pipeline for each symbol.
    data_ingest_runs = []
    for symbol in symbols:
        data_ingest_runs.append(
            run_data_ingest(
                logical_date=logical_date,
                symbol=symbol,
                days_delay=days_delay,
                interval=interval,
                feature_group_version=feature_group_version,
            )
        )

    # Create feature view
    feature_view = create_feature_view(
        logical_date=logical_date,
        days_delay=days_delay,
        feature_group_version=feature_group_version,
    )

    # Run batch prediction pipeline for each symbol.
    batch_predict_runs = []
    for symbol in symbols:
        batch_predict_runs.append(
            batch_predict(
                logical_date=logical_date,
                symbol=symbol,
                feature_view_version=feature_view_version,
            )
        )

    # Compute monitoring metrics for each symbol.
    monitoring_runs = []
    for symbol in symbols:
        monitoring_runs.append(
            compute_monitoring(
                logical_date=logical_date,
                symbol=symbol,
                feature_view_version=feature_view_version,
            )
        )

    # TODO: Use a different feature view version than for training but careful to avoid costs

    # Set up task dependencies, first do all the data ingest runs, then create the feature view,
    # then run all the batch prediction runs and compute the monitoring metrics.
    # Don't do the ingest runs in parallel to avoid overloading the feature store.
    start >> data_ingest_runs[0]
    for i in range(len(data_ingest_runs) - 1):
        data_ingest_runs[i] >> data_ingest_runs[i + 1]
    data_ingest_runs[-1] >> feature_view
    feature_view >> batch_predict_runs
    feature_view >> monitoring_runs
    (batch_predict_runs + monitoring_runs) >> end


ml_pipeline()
