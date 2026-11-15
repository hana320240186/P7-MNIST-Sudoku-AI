"""Beginner-friendly UCS, Greedy and A* correction search."""
import heapq
import math
from p7.sudoku import find_conflicts, solve_sudoku


def heuristic(grid):
    # Greedy disjoint-conflict count: a lower-bound estimate of cells that must change.
    conflicts = find_conflicts(grid)
    used = set()
    count = 0
    for a, b in conflicts:
        if a not in used and b not in used:
            used.update((a, b))
            count += 1
    return count


def correct_grid(start_grid, probabilities, method="ucs", max_changes=3, top_k=4):
    """probabilities[cell] is {digit: probability}; returns result + simple search statistics."""
    start = tuple(start_grid)
    # heap item: (priority, cost_so_far, number_of_changes, grid, correction_log)
    queue = [(0.0, 0.0, 0, start, [])]
    best_cost = {start: 0.0}
    expanded = 0
    generated = 1

    while queue:
        _, cost, depth, state, log = heapq.heappop(queue)
        if cost != best_cost.get(state):
            continue
        expanded += 1
        solution = solve_sudoku(list(state))
        if solution is not None:
            return {"success": True, "corrected_grid": list(state), "solution": solution,
                    "log": log, "expanded": expanded, "generated": generated,
                    "total_cost": cost, "changes": depth}
        if depth >= max_changes:
            continue

        conflicts = find_conflicts(list(state))
        conflict_cells = sorted({cell for pair in conflicts for cell in pair})
        for cell in conflict_cells:
            old = state[cell]
            ranked = sorted(probabilities[cell].items(), key=lambda item: item[1], reverse=True)[:top_k]
            for new, probability in ranked:
                if new == old:
                    continue
                new_state = list(state)
                new_state[cell] = int(new)
                new_state = tuple(new_state)
                # Non-negative likelihood-drop cost, protected against zero probabilities.
                old_p = max(probabilities[cell].get(old, 1e-8), 1e-8)
                new_p = max(float(probability), 1e-8)
                step_cost = max(0.0, math.log(old_p / new_p))
                new_cost = cost + step_cost
                if new_cost >= best_cost.get(new_state, float("inf")):
                    continue
                best_cost[new_state] = new_cost
                new_log = log + [(cell, old, int(new), step_cost)]
                h = heuristic(new_state)
                if method == "greedy":
                    priority = float(h)
                elif method == "astar":
                    priority = new_cost + h
                else:
                    priority = new_cost
                heapq.heappush(queue, (priority, new_cost, depth + 1, new_state, new_log))
                generated += 1
    return {"success": False, "corrected_grid": list(start), "solution": None,
            "log": [], "expanded": expanded, "generated": generated,
            "total_cost": None, "changes": 0}
