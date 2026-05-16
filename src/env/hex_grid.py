"""
Hex grid using even-r offset coordinates.

Cell IDs are assigned row-by-row, left-to-right:
  cell_id = row * width + col

Direction encoding (0-5):
  0 = NW  (top-left)
  1 = NE  (top-right)
  2 = E   (right)
  3 = SE  (bottom-right)
  4 = SW  (bottom-left)
  5 = W   (left)

Neighbor offsets differ for even/odd rows (even-r offset system).
"""
from typing import Dict, List, Optional, Tuple


# (delta_row, delta_col) for each direction, indexed by [row_parity][direction]
_OFFSETS: List[List[Tuple[int, int]]] = [
    # even rows
    [(-1, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (0, -1)],
    # odd rows
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1)],
]

OPPOSITE_DIR = [3, 4, 5, 0, 1, 2]   # opposite of direction d


class HexGrid:
    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self._adj:   List[List[Tuple[int, int]]] = self._build_adjacency()
        self._dist:  Dict[Tuple[int, int], int]  = {}   # cached distances

    # ------------------------------------------------------------------ #
    # Coordinate helpers                                                   #
    # ------------------------------------------------------------------ #

    def cell_to_rc(self, cell_id: int) -> Tuple[int, int]:
        return divmod(cell_id, self.width)

    def rc_to_cell(self, row: int, col: int) -> int:
        return row * self.width + col

    def is_valid_rc(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_valid(self, cell_id: int) -> bool:
        return 0 <= cell_id < self.width * self.height

    def n_cells(self) -> int:
        return self.width * self.height

    # ------------------------------------------------------------------ #
    # Adjacency                                                            #
    # ------------------------------------------------------------------ #

    def _build_adjacency(self) -> List[List[Tuple[int, int]]]:
        """For each cell, list of (direction, neighbor_cell_id)."""
        adj = []
        for cid in range(self.width * self.height):
            row, col  = self.cell_to_rc(cid)
            parity    = row & 1
            nbrs: List[Tuple[int, int]] = []
            for d, (dr, dc) in enumerate(_OFFSETS[parity]):
                nr, nc = row + dr, col + dc
                if self.is_valid_rc(nr, nc):
                    nbrs.append((d, self.rc_to_cell(nr, nc)))
            adj.append(nbrs)
        return adj

    def neighbors(self, cell_id: int) -> List[Tuple[int, int]]:
        """Return list of (direction, neighbor_cell_id)."""
        return self._adj[cell_id]

    def neighbor_in_dir(self, cell_id: int, direction: int) -> Optional[int]:
        for d, nid in self._adj[cell_id]:
            if d == direction:
                return nid
        return None   # edge of map

    def direction_to(self, src: int, dst: int) -> Optional[int]:
        """Return direction index from src to dst if they are adjacent, else None."""
        for d, nid in self._adj[src]:
            if nid == dst:
                return d
        return None

    # ------------------------------------------------------------------ #
    # Distance (cube coordinate hex distance)                             #
    # ------------------------------------------------------------------ #

    def hex_distance(self, a: int, b: int) -> int:
        """Exact hex grid distance (unweighted, ignoring terrain)."""
        r1, c1 = self.cell_to_rc(a)
        r2, c2 = self.cell_to_rc(b)
        # Convert even-r offset → cube
        q1 = c1 - (r1 - (r1 & 1)) // 2
        q2 = c2 - (r2 - (r2 & 1)) // 2
        dq = q2 - q1
        dr = r2 - r1
        return max(abs(dq), abs(dr), abs(dq + dr))
