import pytest
from unittest.mock import patch, MagicMock
from Logging.log_function import PySimFin

API_KEY = "test-api-key"

@pytest.fixture
def simfin():
    return PySimFin(API_KEY)

@patch("log_function.requests.get")
def test_get_share_prices_success(mock_get, simfin):
    # Simulate successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = [{"Date": "2022-01-01", "Close": 150}]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    df = simfin.get_share_prices("AAPL", "2022-01-01", "2024-12-31")

    assert not df.empty
    assert "Close" in df.columns
    mock_get.assert_called_once()

@patch("log_function.requests.get")
def test_get_financial_statement_success(mock_get, simfin):
    # Simulate successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = [{"Revenue": 1000000}]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    df = simfin.get_financial_statement("AAPL", "2022-01-01", "2024-12-31")

    assert not df.empty
    assert "Revenue" in df.columns
    mock_get.assert_called_once()

@patch("log_function.requests.get")
def test_get_share_prices_failure(mock_get, simfin):
    # Simulate an error response
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API Error")
    mock_get.return_value = mock_response

    with pytest.raises(Exception, match="API Error"):
        simfin.get_share_prices("INVALID", "2022-01-01", "2024-12-31")

@patch("log_function.requests.get")
def test_get_financial_statement_failure(mock_get, simfin):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API Error")
    mock_get.return_value = mock_response

    with pytest.raises(Exception, match="API Error"):
        simfin.get_financial_statement("INVALID", "2022-01-01", "2024-12-31")

@patch("log_function.requests.get")
def test_get_share_prices_empty(mock_get, simfin):
    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    df = simfin.get_share_prices("AAPL", "2022-01-01", "2024-12-31")

    assert df.empty
    mock_get.assert_called_once()

@patch("log_function.requests.get")
def test_get_financial_statement_partial_data(mock_get, simfin):
    mock_response = MagicMock()
    mock_response.json.return_value = [{"Revenue": None, "Cost": 123}]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    df = simfin.get_financial_statement("MSFT", "2022-01-01", "2024-12-31")

    assert "Revenue" in df.columns
    assert "Cost" in df.columns
    assert df.iloc[0]["Cost"] == 123

@patch("log_function.logger")
@patch("log_function.requests.get")
def test_logging_called_on_success(mock_get, mock_logger, simfin):
    mock_response = MagicMock()
    mock_response.json.return_value = [{"Date": "2022-01-01", "Close": 100}]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    simfin.get_share_prices("AAPL", "2022-01-01", "2024-12-31")

    assert mock_logger.info.called
