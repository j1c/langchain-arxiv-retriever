import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    return {
        "decode_compressed_response": True,
    }
