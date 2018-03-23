"""Integration test of the DARK algorithm."""
from dla.algos import DaleModel
from dla.slicing import standard_slices
from dla.topology import StaticAgentsNetwork
import networkx as nx
import numpy as np
import numpy.testing as npt


def test_dale_small():
    """Test DALE on a small matrix."""
    size = 3
    a = np.array([
        [7.0, 1.0, 0.0],
        [2.0, 3.0, 3.0],
        [0.0, 1.0, 8.0],
        ])
    b = np.array([1.0, 0.0, 3.0])
    x = np.linalg.solve(a, b)

    slices = standard_slices(size, 3, overlap=0)
    g = nx.path_graph(len(slices))
    model = DaleModel(a, b, slices)
    network = StaticAgentsNetwork(model, g)
    model.attach_network(network)
    for _ in range(500):
        model.step()

    for agent in model.schedule.agents:
        npt.assert_almost_equal(agent.sol, x)


def test_dale_large():
    """Test DALE on a larger matrix."""
    size = 30
    np.random.seed(777)
    a = np.random.randn(size, size) + np.eye(size, size)*10.0
    b = np.random.randn(size)
    x = np.linalg.solve(a, b)

    slices = standard_slices(size, 15, overlap=2)
    g = nx.wheel_graph(len(slices))
    model = DaleModel(a, b, slices)
    network = StaticAgentsNetwork(model, g)
    model.attach_network(network)
    for _ in range(1500):
        model.step()

    for agent in model.schedule.agents:
        npt.assert_almost_equal(agent.sol, x)
