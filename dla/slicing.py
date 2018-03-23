"""Routines for slicing, i.e. distributing the problem Ax = b onto agents."""


def standard_slices(problem_size, num_agents, overlap=0):
    """Create standard slices for a problem.

    We assume that the problem size is exactly divisible by the number
    of agents; hence all agents have exactly the same subproblem size.

    Parameters
    ----------
    problem_size : int
        problem size

    num_agents : int
        number of agents

    overlap : int (optional)
        how many rows overlap should the agents have. The overlap is cyclic,
        i.e. the last agent shares some rows with the first agent.

    Returns
    -------
    list of lists of ints
        a list of row indices corresponding to each agent

    """
    if num_agents < 1:
        raise ValueError("Number of agents must be greater or equal one")
    if overlap < 0:
        raise ValueError("Overlap must be greater or equal zero")
    if problem_size % num_agents != 0:
        raise ValueError(
            "Problem size must be an exact multiple of number of agents"
            )
    stride = problem_size // num_agents
    if stride + overlap > problem_size:
        raise ValueError("Overlap is too large, repeating rows")
    return [
        [(i + j) % problem_size for j in range(stride + overlap)]
        for i in range(0, problem_size, stride)
        ]
