"""Tests the slicing functions."""
from dla import standard_slices
import pytest


def test_standard_slices_no_overlap():
    """Test standard slices without overlap."""
    problem_size = 10
    num_agents = 5
    ground_truth = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        ]
    idxs = standard_slices(problem_size, num_agents)
    assert idxs == ground_truth


def test_standard_slices_no_overlap_2():
    """Test standard slices without overlap, another variant."""
    problem_size = 14
    num_agents = 2
    ground_truth = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        ]
    idxs = standard_slices(problem_size, num_agents)
    assert idxs == ground_truth


def test_standard_slices_with_overlap():
    """Test standard slices with overlap."""
    problem_size = 10
    num_agents = 5
    overlap = 3
    ground_truth = [
        [0, 1, 2, 3, 4],
        [2, 3, 4, 5, 6],
        [4, 5, 6, 7, 8],
        [6, 7, 8, 9, 0],
        [8, 9, 0, 1, 2],
        ]
    idxs = standard_slices(problem_size, num_agents, overlap=overlap)
    assert idxs == ground_truth


def test_standard_slices_too_few_agents():
    """Test if an exception is raised if too few agents."""
    with pytest.raises(ValueError) as excinfo:
        standard_slices(10, 0)
    assert "Number of agents must be greater or equal one" \
        in str(excinfo.value)


def test_standard_slices_negative_overlap():
    """Test if an exception is raised if the overlap is negative."""
    with pytest.raises(ValueError) as excinfo:
        standard_slices(2, 2, overlap=-1)
    assert "Overlap must be greater or equal zero" in str(excinfo.value)


def test_standard_slices_too_large_overlap():
    """Test if an exception is raised if the overlap is too large."""
    with pytest.raises(ValueError) as excinfo:
        standard_slices(2, 2, overlap=2)
    assert "Overlap is too large" in str(excinfo.value)


def test_standard_slices_undivisible_problem_size():
    """Test if an exc is raised if prob size is indivisible by no of agents."""
    with pytest.raises(ValueError) as excinfo:
        standard_slices(2, 3)
    assert "Problem size must be an exact multiple" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        standard_slices(7, 3)
    assert "Problem size must be an exact multiple" in str(excinfo.value)
