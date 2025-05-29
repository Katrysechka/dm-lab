import sys

sys.path.append("..")
import numpy as np
import pytest
from ..src.data_generation import generate_sample


def test_t3_sample_length_and_type():
    """Генерация t(3): правильный размер и тип."""
    x = generate_sample("t3", alpha=3, size=100, seed=42)
    assert isinstance(x, np.ndarray), "Ожидался numpy.ndarray"
    assert x.shape == (100,), "Ожидалась форма (100,)"


def test_normal_sample_length_and_type():
    """Генерация N(0,1): правильный размер и тип."""
    x = generate_sample("normal", alpha=1, size=50, seed=123)
    assert isinstance(x, np.ndarray), "Ожидался numpy.ndarray"
    assert x.shape == (50,), "Ожидалась форма (50,)"


def test_invalid_distribution_raises():
    """Неверное имя распределения вызывает ValueError."""
    with pytest.raises(ValueError, match="Unknown distribution"):
        generate_sample("unknown", alpha=1, size=10)


def test_negative_size_raises():
    """Отрицательный размер вызывает ValueError."""
    with pytest.raises(ValueError):
        generate_sample("normal", alpha=1, size=-5)


def test_t3_sample_statistics():
    """t(3) распределение: среднее ≈ 0, стандартное отклонение > 1."""
    x = generate_sample("t3", alpha=3, size=100_000, seed=0)
    mean = np.mean(x)
    std = np.std(x)
    assert abs(mean) < 0.05, f"Среднее t3 слишком далеко от 0: {mean}"
    assert std > 1.0, f"Ст. отклонение t3 слишком маленькое: {std}"


def test_normal_sample_statistics():
    """Normal(0,1): среднее ≈ 0, стандартное отклонение ≈ 1."""
    x = generate_sample("normal", alpha=1, size=100_000, seed=1)
    mean = np.mean(x)
    std = np.std(x)
    assert abs(mean) < 0.01, f"Среднее нормальное не близко к 0: {mean}"
    assert abs(std - 1.0) < 0.01, f"Ст. отклонение нормального не ≈ 1: {std}"


def test_gamma_sample_length_and_type():
    """Генерация gamma: правильный размер и тип."""
    x = generate_sample("gamma", alpha=2.0, size=100, seed=42)
    assert isinstance(x, np.ndarray), "Ожидался numpy.ndarray"
    assert x.shape == (100,), "Ожидалась форма (100,)"


def test_gamma_sample_statistics():
    """Gamma: среднее и стандартное отклонение разумны."""
    x = generate_sample("gamma", alpha=2.0, size=100_000, seed=0)
    mean = np.mean(x)
    std = np.std(x)
    assert mean > 0, f"Среднее gamma должно быть положительным: {mean}"
    assert std > 0, f"Ст. отклонение gamma должно быть положительным: {std}"


def test_exp_sample_length_and_type():
    """Генерация exponential: правильный размер и тип."""
    x = generate_sample("exp", alpha=1.0, size=100, seed=123)
    assert isinstance(x, np.ndarray), "Ожидался numpy.ndarray"
    assert x.shape == (100,), "Ожидалась форма (100,)"


def test_exp_sample_statistics():
    """Exponential: среднее ≈ alpha, стандартное отклонение ≈ alpha."""
    alpha = 2.0
    x = generate_sample("exp", alpha=alpha, size=100_000, seed=1)
    mean = np.mean(x)
    std = np.std(x)
    assert abs(mean - alpha) < 0.05 * alpha, f"Среднее эксп. не ≈ {alpha}: {mean}"
    assert abs(std - alpha) < 0.05 * alpha, f"Ст. отклонение эксп. не ≈ {alpha}: {std}"
