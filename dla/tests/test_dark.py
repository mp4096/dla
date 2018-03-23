"""Integration test of the DARK algorithm."""
from dla.algos import DarkModel
from dla.linalg import arnoldi, modified_gram_schmidt
from dla.slicing import standard_slices
from dla.topology import StaticAgentsNetwork
import networkx as nx
import numpy as np
import numpy.testing as npt


def test_dark_without_mgs():
    """Test DARK without reorthonormalisation."""
    size = 3
    num_directions = 2
    a = np.array([
        [0.0, 1.0, 2.0],
        [2.0, 3.0, 3.0],
        [2.0, 1.0, 1.0],
        ])
    b = np.array([
        [1.0],
        [0.0],
        [3.0],
        ])

    x = np.empty((size, num_directions))
    x[:, 0] = b.squeeze()
    x[:, 1] = np.linalg.solve(a, x[:, 0])

    slices = standard_slices(size, 3, overlap=0)
    g = nx.path_graph(len(slices))
    model = DarkModel(a, b, slices, num_directions)
    network = StaticAgentsNetwork(model, g)
    model.attach_network(network)
    for _ in range(2000):
        model.step()

    for agent in model.schedule.agents:
        npt.assert_almost_equal(agent.rks, x)


def test_dark_without_mgs_zero_elements():
    """Test DARK without reorthonormalisation if b contains zeros."""
    # We need more than one iteration for this
    num_directions = 3
    size = 4
    a = np.array([
        [0.0, 1.0, 2.0, 0.0],
        [2.0, 3.0, 3.0, 0.0],
        [2.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 3.0],
        ])
    b = np.array([1.0, 0.0, 0.0, 0.0])
    x = np.empty((size, num_directions))
    x[:, 0] = b.squeeze()
    x[:, 1] = np.linalg.solve(a, x[:, 0])
    x[:, 2] = np.linalg.solve(a, x[:, 1])

    slices = standard_slices(size, 4, overlap=1)
    g = nx.path_graph(len(slices))
    model = DarkModel(a, b, slices, num_directions)
    network = StaticAgentsNetwork(model, g)
    model.attach_network(network)
    for _ in range(2000):
        model.step()

    for agent in model.schedule.agents:
        npt.assert_almost_equal(agent.rks, x)
