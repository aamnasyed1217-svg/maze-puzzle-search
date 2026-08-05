# submitted.py
# Search algorithms for the maze assignment.
# Only Python standard-library modules are used.

from collections import deque
import heapq
import math


def _goal(maze):
    """Return the single waypoint used as the goal."""
    return maze.waypoints[0]


def _reconstruct(parent, start, goal):
    """Reconstruct a path from start to goal using a parent dictionary."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = parent.get(current)
    path.reverse()
    return path


def manhattan(a, b):
    """Manhattan distance heuristic for grid movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """Euclidean distance heuristic for grid movement."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def bfs(maze):
    """Breadth-first search for a single waypoint.

    BFS uses a FIFO queue and returns a shortest path in number of moves
    when all moves have equal cost.
    """
    start = maze.start
    goal = _goal(maze)

    if start == goal:
        return [start]

    frontier = deque([start])
    visited = {start}
    parent = {start: None}

    while frontier:
        state = frontier.popleft()
        for nbr in maze.neighbors(*state):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = state
                if nbr == goal:
                    return _reconstruct(parent, start, goal)
                frontier.append(nbr)

    return []


def dfs(maze):
    """Depth-first search for a single waypoint.

    DFS uses a stack. Neighbors are pushed in reverse order so the first
    neighbor returned by maze.neighbors is explored first.
    """
    start = maze.start
    goal = _goal(maze)

    if start == goal:
        return [start]

    stack = [start]
    visited = set()
    parent = {start: None}

    while stack:
        state = stack.pop()
        if state in visited:
            continue
        visited.add(state)

        if state == goal:
            return _reconstruct(parent, start, goal)

        neighbors = list(maze.neighbors(*state))
        for nbr in reversed(neighbors):
            if nbr not in visited and nbr not in parent:
                parent[nbr] = state
                stack.append(nbr)

    return []


def dijkstra(maze):
    """Dijkstra's algorithm for a single waypoint.

    Each move has cost 1, so Dijkstra returns an optimal path. It is included
    for the report/autograder even if main.py only lists bfs, dfs, and A*.
    """
    start = maze.start
    goal = _goal(maze)

    frontier = [(0, start)]
    parent = {start: None}
    best_cost = {start: 0}
    explored = set()

    while frontier:
        cost, state = heapq.heappop(frontier)
        if state in explored:
            continue
        explored.add(state)

        if state == goal:
            return _reconstruct(parent, start, goal)

        for nbr in maze.neighbors(*state):
            new_cost = cost + 1
            if nbr not in best_cost or new_cost < best_cost[nbr]:
                best_cost[nbr] = new_cost
                parent[nbr] = state
                heapq.heappush(frontier, (new_cost, nbr))

    return []


def astar_single(maze):
    """A* search for a single waypoint using Manhattan distance.

    Manhattan distance is admissible for 4-direction grid movement because
    every move changes the row or column by exactly one.
    """
    start = maze.start
    goal = _goal(maze)

    frontier = [(manhattan(start, goal), 0, start)]  # (f, g, state)
    parent = {start: None}
    best_g = {start: 0}
    explored = set()

    while frontier:
        f, g, state = heapq.heappop(frontier)
        if state in explored:
            continue
        explored.add(state)

        if state == goal:
            return _reconstruct(parent, start, goal)

        for nbr in maze.neighbors(*state):
            new_g = g + 1
            if nbr not in best_g or new_g < best_g[nbr]:
                best_g[nbr] = new_g
                parent[nbr] = state
                new_f = new_g + manhattan(nbr, goal)
                heapq.heappush(frontier, (new_f, new_g, nbr))

    return []
