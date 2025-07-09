import streamlit as st
import plotly.graph_objects as go
from queue import PriorityQueue
import random
import itertools
import numpy as np
import time

# ---------------------------- Grid3D Class ----------------------------
class Grid3D:
    def __init__(self, grid_size, obstacle_count):
        self.grid_size = grid_size
        self.grid = [[[0 for _ in range(grid_size)] for _ in range(grid_size)] for _ in range(grid_size)]
        self.obstacles = self.generate_obstacles(obstacle_count)

    def generate_obstacles(self, obstacle_count):
        max_obstacles = self.grid_size ** 3 - 2
        obstacles = set()
        while len(obstacles) < min(obstacle_count, max_obstacles):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            z = random.randint(0, self.grid_size - 1)
            obstacles.add((x, y, z))
        return obstacles

    def is_valid(self, point):
        x, y, z = point
        return (0 <= x < self.grid_size and
                0 <= y < self.grid_size and
                0 <= z < self.grid_size and
                point not in self.obstacles)

    def get_neighbors(self, cell):
        directions = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if (dx, dy, dz) != (0, 0, 0)]
        neighbors = []
        for dx, dy, dz in directions:
            neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if self.is_valid(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def a_star_search(self, start, end):
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        visited = set()

        while not open_set.empty():
            _, current = open_set.get()
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            visited.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                step_cost = np.linalg.norm(np.array(neighbor) - np.array(current))
                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    open_set.put((f_score[neighbor], neighbor))

        return []

# ---------------------------- TSP Solver ----------------------------
def solve_tsp_branch_and_bound(distance_matrix):
    n = len(distance_matrix)
    min_cost = float('inf')
    best_path = []

    for perm in itertools.permutations(range(1, n)):
        path = [0] + list(perm)
        cost = sum(distance_matrix[path[i]][path[i+1]] for i in range(n-1))
        cost += distance_matrix[path[-1]][0]
        if cost < min_cost:
            min_cost = cost
            best_path = path

    return best_path, min_cost

# ---------------------------- Knapsack Solver ----------------------------
def knapsack_01(values, weights, capacity):
    n = len(values)
    dp = [[0]*(capacity+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for w in range(capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    w = capacity
    selected = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= weights[i-1]

    return selected[::-1]

# ---------------------------- Visualization ----------------------------
def plot_path(grid3d, points, paths, opacity):
    fig = go.Figure()

    # Obstacles
    if grid3d.obstacles:
        obs_x, obs_y, obs_z = zip(*grid3d.obstacles)
        fig.add_trace(go.Scatter3d(
            x=obs_x, y=obs_y, z=obs_z,
            mode='markers',
            marker=dict(size=3, color='gray', opacity=opacity),
            name='Obstacles'
        ))

    # Points
    colors = ['green'] + ['orange']*(len(points)-2) + ['red']
    labels = ['Start'] + [f"Goal {i}" for i in range(1, len(points)-1)] + ['End']
    for i, (point, color, label) in enumerate(zip(points, colors, labels)):
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode='markers+text',
            marker=dict(size=6, color=color),
            text=[f"{i}: {point}"],
            textposition='top center',
            name=label
        ))

    # Paths with Distance and Hop Count Labels
    hop_counter = 0
    for path in paths:
        if len(path) > 1:
            x, y, z = zip(*path)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='blue', width=4),
                name='Path'
            ))
            for i in range(len(path)-1):
                hop_counter += 1
                mid = tuple((np.array(path[i]) + np.array(path[i+1])) / 2)
                dist = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                # Distance Label
                fig.add_trace(go.Scatter3d(
                    x=[mid[0]], y=[mid[1]], z=[mid[2]],
                    mode='text',
                    text=[f"{dist:.1f}"],
                    textposition='middle center',
                    showlegend=False
                ))
                # Hop Count Label (slightly offset in Z)
                fig.add_trace(go.Scatter3d(
                    x=[mid[0]], y=[mid[1]], z=[mid[2] + 0.3],
                    mode='text',
                    text=[f"{hop_counter}"],
                    textposition='middle center',
                    showlegend=False
                ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1, y=1),
        title="3D Route Visualization (Interactive)"
    )

    return fig

# ---------------------------- Streamlit App ----------------------------
def main():
    st.title("🚁 3D Pathfinding with TSP and Knapsack (Euclidean)")

    grid_size = st.slider("Grid Size", 5, 20, 10)
    obstacle_count = st.slider("Obstacle Count", 0, grid_size**2, 30)
    opacity = st.slider("Obstacle Opacity", 0.1, 1.0, 0.5)
    num_goals = st.slider("Number of Goal Points", 2, 6, 3)

    seed = st.number_input("Random Seed (0 for random)", min_value=0, value=0)
    if seed == 0:
        random.seed(time.time_ns())
    else:
        random.seed(seed)

    st.markdown("### Define Goal Points")
    goal_points, values, weights = [], [], []
    for i in range(num_goals):
        col1, col2, col3 = st.columns(3)
        with col1:
            pt = st.text_input(f"Point {i+1} (x y z)", value=f"{i} {i} {i}")
        with col2:
            val = st.number_input(f"Value {i+1}", min_value=1, value=10)
        with col3:
            wt = st.number_input(f"Weight {i+1}", min_value=1, value=5)
        try:
            goal_points.append(tuple(map(int, pt.split())))
            values.append(val)
            weights.append(wt)
        except:
            st.error(f"Invalid input for Point {i+1}")
            return

    capacity = st.number_input("Drone Capacity", min_value=1, value=15)
    algo = st.radio("Select Algorithm Mode", ["A* Only", "Multi-Goal TSP", "TSP + Knapsack"])

    if st.button("Run Pathfinding"):
        grid3d = Grid3D(grid_size, obstacle_count)
        valid_goals = [pt for pt in goal_points if grid3d.is_valid(pt)]
        if len(valid_goals) < len(goal_points): st.warning("Some points are in obstacles or invalid!")

        start_point = valid_goals[0]
        middle_points = valid_goals[1:-1]
        end_point = valid_goals[-1]

        if algo == "A* Only":
            if len(valid_goals) != 2: st.warning("A* only works with exactly 2 valid points.") ; return
            path = grid3d.a_star_search(valid_goals[0], valid_goals[1])
            fig = plot_path(grid3d, valid_goals, [path], opacity)
            total_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
            hop_count = len(path) - 1
            st.markdown("### 📏 Path Details")
            st.success(f"Path Length: {total_length:.2f} units")
            st.info(f"Hop Count: {hop_count} steps")

        elif algo == "Multi-Goal TSP":
            tsp_points = [start_point] + middle_points + [end_point]
            n = len(tsp_points)
            dist_matrix = [[float('inf')]*n for _ in range(n)]
            paths_matrix = [[[]]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        p = grid3d.a_star_search(tsp_points[i], tsp_points[j])
                        if p:
                            dist_matrix[i][j] = sum(np.linalg.norm(np.array(p[k+1]) - np.array(p[k])) for k in range(len(p)-1))
                            paths_matrix[i][j] = p
            order, _ = solve_tsp_branch_and_bound(dist_matrix)
            if order[0] != 0 or order[-1] != n - 1:
                order = [0] + [i for i in order if i not in [0, n-1]] + [n-1]
            full_path = [paths_matrix[order[i]][order[i+1]] for i in range(len(order)-1)]
            fig = plot_path(grid3d, tsp_points, full_path, opacity)
            total_length, total_hops = 0, 0
            for p in full_path:
                total_length += sum(np.linalg.norm(np.array(p[i+1]) - np.array(p[i])) for i in range(len(p)-1))
                total_hops += len(p) - 1
            st.markdown("### 📏 TSP Path Summary")
            st.success(f"Total Path Length: {total_length:.2f} units")
            st.info(f"Total Hop Count: {total_hops} steps")

        elif algo == "TSP + Knapsack":
            idx = knapsack_01(values, weights, capacity)
            selected_goals = [valid_goals[i] for i in idx]
            st.info(f"Selected {len(idx)} goals based on value and capacity.")
            n = len(selected_goals)
            dist_matrix = [[float('inf')]*n for _ in range(n)]
            paths_matrix = [[[]]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        p = grid3d.a_star_search(selected_goals[i], selected_goals[j])
                        if p:
                            dist_matrix[i][j] = sum(np.linalg.norm(np.array(p[k+1]) - np.array(p[k])) for k in range(len(p)-1))
                            paths_matrix[i][j] = p
            order, _ = solve_tsp_branch_and_bound(dist_matrix)
            full_path = [paths_matrix[order[i]][order[i+1]] for i in range(len(order)-1)]
            fig = plot_path(grid3d, selected_goals, full_path, opacity)
            total_length, total_hops = 0, 0
            for p in full_path:
                total_length += sum(np.linalg.norm(np.array(p[i+1]) - np.array(p[i])) for i in range(len(p)-1))
                total_hops += len(p) - 1
            st.markdown("### 📏 TSP + Knapsack Path Summary")
            st.success(f"Total Path Length: {total_length:.2f} units")
            st.info(f"Total Hop Count: {total_hops} steps")
            total_value = sum(values[i] for i in idx)
            total_weight = sum(weights[i] for i in idx)
            st.markdown("### 🎒 Knapsack Summary")
            st.write(f"Total Value Collected: {total_value}")
            st.write(f"Total Weight Carried: {total_weight}")

        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()