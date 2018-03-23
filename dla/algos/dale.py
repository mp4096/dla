"""DALE algorithm implementation."""
import itertools
from mesa import Model, Agent
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import numpy as np
from ..linalg import online_variance, projector_onto_kernel


class DaleAgent(Agent):
    """Agent for the DALE algorithm."""

    def __init__(self, unique_id, model, slice_idx):
        """Initialise an agent for the DALE algorithm.

        Parameters
        ----------
        unique_id : str
            unique ID for the agent.

        model : inherited from Mesa's Model class
            reference to the corresponding ABM model.

        slice_idx : list of ints
            list containing indices that describe the agent's problem subset.
            Basically a compressed representation of the selector matrix,
            see paper.

        """
        super().__init__(unique_id, model)

        # Check if there any repeated indices
        if not len(set(slice_idx)) == len(slice_idx):
            raise ValueError("Slice indices must be unique")
        self.slice_idx = slice_idx
        self.subproblem_size = len(slice_idx)

        self.a_sub = self.model.a[slice_idx, :]
        self.b_sub = self.model.b[slice_idx]

        # Precompute the projector onto kernel of A_i
        self.proj = projector_onto_kernel(self.a_sub)

        # Initialise the solution matrix
        self.sol = np.linalg.pinv(self.a_sub) @ self.b_sub
        self._new_sol = self.sol.copy()

    def step(self):
        """Perform a simulation step.

        Only precomputes the new solution; the update of the current estimate
        is performed in the advance step.

        """
        neighbours_avg = np.zeros(self.sol.shape)
        num_neighbours = 0
        for neighbour_agent in self.model.network.get_neighbours(self):
            neighbours_avg += neighbour_agent.sol
            num_neighbours += 1
        neighbours_avg /= num_neighbours

        # Agent update rule
        self._new_sol = self.sol - self.proj @ (self.sol - neighbours_avg)

    def advance(self):
        """Advance and update the current solution estimate."""
        self.sol[:] = self._new_sol[:]


class DaleModel(Model):
    """Agent-based model for the DALE algorithm."""

    def __init__(self, a, b, slices, log_each=1, thr=1.0e-9):
        """Initialise the DALE model.

        Parameters
        ----------
        a : (n, n) array_like
            coefficients matrix.

        b : (n, 1) or (n,) array_like
            right hand side vector.

        slices : list of lists of ints
            definition of subproblem distribution onto agents.

        log_each : int, optional
            if greater then one, data will be collected not at each step.
            Helps to keep the dataframe sizes low if simulation time is long.

        thr : float
            convergence threshold.

        """
        if a.shape[0] != b.shape[0]:
            raise ValueError("Matrix dimensions must agree")

        self.log_each = log_each
        self.num_steps = 0
        self.problem_size = b.shape[0]

        self.a = a
        self.b = b.squeeze()

        # Get a set of indices specified in slices
        idx_in_slices = set(itertools.chain.from_iterable(slices))
        if len(idx_in_slices) != self.problem_size:
            msg = "Not all indices are contained in slices: "
            raise ValueError(msg + str(idx_in_slices))

        self.num_agents = len(slices)
        self.network = None
        self.schedule = SimultaneousActivation(self)

        for i in range(self.num_agents):
            agent = DaleAgent(str(i + 1), self, slices[i])
            self.schedule.add(agent)

        self.running = True
        self.thr = thr

        # TODO: Bandwidth monitor
        self.datacollector = DataCollector(
            model_reporters={"step": lambda m: m.num_steps},
            agent_reporters={"solution": lambda a: a.sol.copy()},
            )

    def attach_network(self, network):
        """Attach a network to the model.

        Parameters
        ----------
        network : an object from the topology submodule
            agents network definition

        """
        if len(list(network.graph)) != self.num_agents:
            raise ValueError("No. of agents must be equal to no. of nodes")
        self.network = network
        for agent in self.schedule.agents:
            self.network.add_agent(agent)

    def agents_variance(self):
        """Compute element-wise variance across agents."""
        return online_variance(
            map(lambda agent: agent.sol, self.schedule.agents),
            (self.problem_size,),
            )

    def step(self):
        """Perform one model step."""
        if self.log_each > 0 and self.num_steps % self.log_each == 0:
            self.datacollector.collect(self)

        self.num_steps += 1

        self.schedule.step()

        # Check for convergence
        if np.linalg.norm(self.agents_variance()) < self.thr:
            self.running = False
