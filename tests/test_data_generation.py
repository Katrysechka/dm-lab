import pytest
from src.data_generation import generate_sample


def test_generate_sample_t3_length():
    sample = generate_sample("t3", 100, seed=42)
    assert len(sample) == 100


def test_generate_sample_normal_length():
    sample = generate_sample("normal", 50, seed=1)
    assert len(sample) == 50


def test_generate_sample_invalid_distribution():
    with pytest.raises(ValueError):
        generate_sample("invalid_dist", 10)
