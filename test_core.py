import unittest
from p7.sudoku import make_demo, find_conflicts, solve_sudoku
from p7.search import correct_grid


def make_probabilities(grid, solution, changed):
    changed_map = {cell: (wrong, correct) for cell, wrong, correct in changed}
    values = []
    for cell, digit in enumerate(grid):
        if digit == 0:
            values.append({d: 1 / 9 for d in range(1, 10)})
        elif cell in changed_map:
            wrong, correct = changed_map[cell]
            row = {d: 0.01 for d in range(1, 10)}
            row[wrong], row[correct] = 0.60, 0.32
            values.append(row)
        else:
            row = {d: 0.0125 for d in range(1, 10)}
            row[digit] = 0.90
            values.append(row)
    return values


class SudokuCoreTests(unittest.TestCase):
    def test_demo_contains_conflicts(self):
        puzzle, solution, changed = make_demo(21, 36, 2)
        self.assertEqual(len(puzzle), 81)
        self.assertEqual(len(solution), 81)
        self.assertEqual(len(changed), 2)
        self.assertGreater(len(find_conflicts(puzzle)), 0)

    def test_ucs_repairs_demo(self):
        puzzle, solution, changed = make_demo(21, 36, 2)
        result = correct_grid(puzzle, make_probabilities(puzzle, solution, changed), "ucs", 3)
        self.assertTrue(result["success"])
        self.assertEqual(result["changes"], 2)
        self.assertEqual(find_conflicts(result["corrected_grid"]), [])
        self.assertIsNotNone(solve_sudoku(result["corrected_grid"]))

    def test_solver_rejects_duplicate_grid(self):
        puzzle = [0] * 81
        puzzle[0], puzzle[1] = 5, 5
        self.assertIsNone(solve_sudoku(puzzle))


if __name__ == "__main__":
    unittest.main()
