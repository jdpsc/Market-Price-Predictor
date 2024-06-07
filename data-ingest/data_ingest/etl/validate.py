from great_expectations.core import ExpectationSuite, ExpectationConfiguration


def build_expectation_suite() -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    """

    expectation_suite_stock = ExpectationSuite(expectation_suite_name="stock_suite")

    # Columns
    expectation_suite_stock.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": ["open", "high", "low", "close", "volume", "s", "date"]
            },
        )
    )
    expectation_suite_stock.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_column_count_to_equal", kwargs={"value": 7}
        )
    )

    # Nulls
    columns_to_check = ["close", "date", "s"]
    for column in columns_to_check:
        expectation_suite_stock.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column},
            )
        )

    # Positive values
    columns_to_check = ["open", "high", "low", "close", "volume"]
    for column in columns_to_check:
        expectation_suite_stock.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": column, "type_": "FLOAT"},
            )
        )
        expectation_suite_stock.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={"column": column, "min_value": 0},
            )
        )

    return expectation_suite_stock
