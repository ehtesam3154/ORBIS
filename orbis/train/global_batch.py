"""Split a fixed number G of views per optimiser step over any number of GPUs.

Slot j of a step goes to rank j % world; every rank runs the same number K = ceil(G / world) of micro-steps, padding
micro-steps carry loss weight 0, and each real view's loss is divided by G / world, so the averaged gradient is the mean
over exactly G views on any GPU count."""


def micro_plan(G, world, rank):
    """(K, real): micro-steps per optimiser step, and whether this rank's micro-step m holds a view."""
    assert G >= 1 and world >= 1 and 0 <= rank < world
    K = -(-G // world)
    n_real = len(range(rank, G, world))
    return K, [m < n_real for m in range(K)]


def loss_divisor(G, world):
    """Divisor of each real micro-step's loss."""
    return G / world
