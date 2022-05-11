import itertools
import math


def get_distance(pos: tuple) -> float:
    """Calculate distance from given grid position

    Args:
        pos (tuple): Three dimensional grid position

    Returns:
        float: Distance in meters
    """
    assert len(pos) == 3, "Grid positions must be in three dimensions"
    assert all(isinstance(p, int)
               for p in pos), "Grid positions must be integer values"

    X, Y, Z = pos
    x = (3 - X) * 0.5
    y = (3 - Y) * 0.5
    z = (2 - Z) * 0.3

    return math.sqrt(sum(p**2 for p in (x, y, z)))
    
def generate_positions() -> list:
    """Generate all possible grid positions on SPARG AIR Dataset

    Returns:
        list: List of all possible grid positions
    """
    x = list(range(7))
    y = list(range(7))
    z = list(range(5))

    grid = list(itertools.product(x, y, z))

    # Remove the position of the microphone array
    grid.remove((3, 3, 2)) 
    # Remove the positions right above and right beyond the mic array
    for p in [(3,3,0), (3,3,1), (3,3,3), (3,3,4)]: 
        grid.remove(p)

    return grid
