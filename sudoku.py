"""Small, readable Sudoku helpers: conflicts and backtracking solver."""
from typing import List, Optional, Tuple


def find_conflicts(grid: List[int]) -> List[Tuple[int, int]]:
    conflicts = set()
    groups = []
    for r in range(9):
        groups.append([r * 9 + c for c in range(9)])
    for c in range(9):
        groups.append([r * 9 + c for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            groups.append([(br + dr) * 9 + bc + dc for dr in range(3) for dc in range(3)])
    for group in groups:
        seen = {}
        for cell in group:
            value = grid[cell]
            if value:
                if value in seen:
                    conflicts.add(tuple(sorted((seen[value], cell))))
                else:
                    seen[value] = cell
    return sorted(conflicts)


def valid_grid(grid: List[int]) -> bool:
    return len(grid) == 81 and all(0 <= x <= 9 for x in grid) and not find_conflicts(grid)


def solve_sudoku(grid: List[int]) -> Optional[List[int]]:
    if not valid_grid(grid):
        return None
    board = grid[:]

    def candidates(cell):
        r, c = divmod(cell, 9)
        used = set(board[r * 9:(r + 1) * 9])
        used.update(board[rr * 9 + c] for rr in range(9))
        br, bc = (r // 3) * 3, (c // 3) * 3
        used.update(board[(br + dr) * 9 + bc + dc] for dr in range(3) for dc in range(3))
        return [d for d in range(1, 10) if d not in used]

    def backtrack():
        empty = [i for i, value in enumerate(board) if value == 0]
        if not empty:
            return True
        best_cell, best_values = min(((i, candidates(i)) for i in empty), key=lambda item: len(item[1]))
        if not best_values:
            return False
        for value in best_values:
            board[best_cell] = value
            if backtrack():
                return True
            board[best_cell] = 0
        return False

    return board if backtrack() else None


def count_solutions(grid: List[int], limit: int = 2) -> int:
    if not valid_grid(grid):
        return 0
    board = grid[:]
    count = 0

    def candidates(cell):
        r, c = divmod(cell, 9)
        used = set(board[r * 9:(r + 1) * 9])
        used.update(board[rr * 9 + c] for rr in range(9))
        br, bc = (r // 3) * 3, (c // 3) * 3
        used.update(board[(br + dr) * 9 + bc + dc] for dr in range(3) for dc in range(3))
        return [d for d in range(1, 10) if d not in used]

    def visit():
        nonlocal count
        if count >= limit:
            return
        empty = [i for i, v in enumerate(board) if v == 0]
        if not empty:
            count += 1
            return
        cell, values = min(((i, candidates(i)) for i in empty), key=lambda item: len(item[1]))
        for value in values:
            board[cell] = value
            visit()
            board[cell] = 0
            if count >= limit:
                return
    visit()
    return count


def make_demo(seed: int = 7, givens: int = 36, mistakes: int = 2):
    """Create a reproducible Sudoku and inject visible wrong readings for method comparison."""
    import random
    rng = random.Random(seed)
    base = [((r * 3 + r // 3 + c) % 9) + 1 for r in range(9) for c in range(9)]
    # Shuffle symbols and rows/columns within bands/stacks to vary the board while preserving validity.
    symbols = list(range(1, 10))
    rng.shuffle(symbols)
    solution = [symbols[v - 1] for v in base]
    cells = list(range(81))
    rng.shuffle(cells)
    kept = set(cells[:givens])
    puzzle = [solution[i] if i in kept else 0 for i in range(81)]
    filled = [i for i, v in enumerate(puzzle) if v]
    rng.shuffle(filled)
    changed = []
    for cell in filled:
        if len(changed) >= mistakes:
            break
        correct = puzzle[cell]
        # Pick a different digit that creates a duplicate in at least one peer group.
        r, c = divmod(cell, 9)
        peers = set(puzzle[r * 9:(r + 1) * 9])
        peers.update(puzzle[rr * 9 + c] for rr in range(9))
        br, bc = (r // 3) * 3, (c // 3) * 3
        peers.update(puzzle[(br + dr) * 9 + bc + dc] for dr in range(3) for dc in range(3))
        options = [d for d in range(1, 10) if d != correct and d in peers]
        if options:
            wrong = rng.choice(options)
            puzzle[cell] = wrong
            changed.append((cell, wrong, correct))
    return puzzle, solution, changed
