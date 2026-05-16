"""Unit tests for HexGrid: coordinates, neighbors, distance."""
import pytest
from env.hex_grid import HexGrid, OPPOSITE_DIR


def grid(w=4, h=4):
    return HexGrid(w, h)


# ------------------------------------------------------------------ #
# Coordinate helpers                                                   #
# ------------------------------------------------------------------ #

class TestCoordinates:
    def test_cell_to_rc(self):
        g = grid(4, 4)
        assert g.cell_to_rc(0)  == (0, 0)
        assert g.cell_to_rc(3)  == (0, 3)
        assert g.cell_to_rc(4)  == (1, 0)
        assert g.cell_to_rc(15) == (3, 3)

    def test_rc_to_cell(self):
        g = grid(4, 4)
        assert g.rc_to_cell(0, 0) == 0
        assert g.rc_to_cell(0, 3) == 3
        assert g.rc_to_cell(1, 0) == 4
        assert g.rc_to_cell(3, 3) == 15

    def test_roundtrip_all_cells(self):
        g = grid(5, 6)
        for cid in range(g.n_cells()):
            r, c = g.cell_to_rc(cid)
            assert g.rc_to_cell(r, c) == cid

    def test_is_valid(self):
        g = grid(4, 4)
        assert g.is_valid(0)
        assert g.is_valid(15)
        assert not g.is_valid(-1)
        assert not g.is_valid(16)


# ------------------------------------------------------------------ #
# Neighbors                                                            #
# ------------------------------------------------------------------ #

class TestNeighbors:
    def test_interior_cell_has_6_neighbors(self):
        g = grid(5, 5)
        # cell 12 = (row=2, col=2) is fully interior
        assert len(g.neighbors(12)) == 6

    def test_top_left_corner_has_2_neighbors(self):
        g = grid(4, 4)
        # cell 0 (row=0, col=0, even row): only E and SE are valid
        nbrs = g.neighbors(0)
        assert len(nbrs) == 2
        directions = {d for d, _ in nbrs}
        assert 2 in directions   # E
        assert 3 in directions   # SE

    def test_all_neighbor_cells_are_valid(self):
        g = grid(6, 6)
        for cid in range(g.n_cells()):
            for d, nbr in g.neighbors(cid):
                assert g.is_valid(nbr), (
                    f"cell {cid} dir {d} → invalid neighbor {nbr}"
                )

    def test_neighbor_symmetry(self):
        """If B is a neighbor of A in direction d, then A must be B's neighbor in OPPOSITE_DIR[d]."""
        g = grid(6, 6)
        for cid in range(g.n_cells()):
            for d, nbr in g.neighbors(cid):
                back = g.neighbor_in_dir(nbr, OPPOSITE_DIR[d])
                assert back == cid, (
                    f"symmetry broken: cell {cid} dir {d} → {nbr}, "
                    f"reverse (dir {OPPOSITE_DIR[d]}) gives {back}"
                )

    def test_neighbor_in_dir_off_edge_is_none(self):
        g = grid(4, 4)
        # cell 0 (top-left, even row): NW=0 and NE=1 go off the map
        assert g.neighbor_in_dir(0, 0) is None   # NW
        assert g.neighbor_in_dir(0, 1) is None   # NE
        assert g.neighbor_in_dir(0, 4) is None   # SW
        assert g.neighbor_in_dir(0, 5) is None   # W

    def test_neighbor_in_dir_known_values(self):
        g = grid(4, 4)
        # cell 0 (row=0, col=0, even): E=2 → (0,1)=cell 1; SE=3 → (1,0)=cell 4
        assert g.neighbor_in_dir(0, 2) == 1
        assert g.neighbor_in_dir(0, 3) == 4
        # cell 4 (row=1, col=0, odd): E=2 → (1,1)=cell 5; NE=1 → (0,1)=cell 1
        assert g.neighbor_in_dir(4, 2) == 5
        assert g.neighbor_in_dir(4, 1) == 1

    def test_direction_to_adjacent(self):
        g = grid(4, 4)
        # cell 0 → cell 1 is E (direction 2)
        assert g.direction_to(0, 1) == 2
        # cell 0 → cell 4 is SE (direction 3)
        assert g.direction_to(0, 4) == 3

    def test_direction_to_non_adjacent_is_none(self):
        g = grid(4, 4)
        assert g.direction_to(0, 2)  is None   # two cells away
        assert g.direction_to(0, 15) is None   # far corner


# ------------------------------------------------------------------ #
# Distance                                                             #
# ------------------------------------------------------------------ #

class TestHexDistance:
    def test_same_cell_distance_is_zero(self):
        g = grid(8, 8)
        for cid in [0, 7, 15, 32, 63]:
            assert g.hex_distance(cid, cid) == 0

    def test_adjacent_cells_distance_is_one(self):
        g = grid(8, 8)
        for cid in range(g.n_cells()):
            for _, nbr in g.neighbors(cid):
                assert g.hex_distance(cid, nbr) == 1, (
                    f"adjacent pair ({cid}, {nbr}) has distance != 1"
                )

    def test_distance_is_symmetric(self):
        g = grid(6, 6)
        # Sample every 5th cell to keep it fast
        cells = list(range(0, g.n_cells(), 5))
        for a in cells:
            for b in cells:
                assert g.hex_distance(a, b) == g.hex_distance(b, a)

    def test_triangle_inequality(self):
        g = grid(6, 6)
        cells = list(range(0, g.n_cells(), 7))
        for a in cells:
            for b in cells:
                for c in cells:
                    assert g.hex_distance(a, c) <= g.hex_distance(a, b) + g.hex_distance(b, c)

    def test_known_distance_same_row(self):
        # 4x4 grid: cell 0 (0,0) to cell 3 (0,3)
        # cube: q0=0,r0=0; q3=3,r3=0; dq=3,dr=0 → max(3,0,3)=3
        g = grid(4, 4)
        assert g.hex_distance(0, 3) == 3

    def test_known_distance_cross_rows(self):
        # cell 0 (0,0) to cell 5 (1,1): verified manually → 2
        g = grid(4, 4)
        assert g.hex_distance(0, 5) == 2
