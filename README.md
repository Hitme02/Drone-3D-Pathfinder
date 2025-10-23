# 🚁 Drone 3D Pathfinder

A Streamlit-based 3D simulation system for intelligent drone navigation in complex environments. This project integrates A* pathfinding, Traveling Salesman Problem (TSP), and 0/1 Knapsack algorithms to simulate multi-goal delivery with obstacle avoidance and capacity optimization — visualized in real-time using Plotly.

---

## 📌 Features

- 🔍 A* Algorithm for shortest path computation in a 3D obstacle-filled grid  
- 🧭 Traveling Salesman Problem (Branch & Bound) for optimal goal visiting order  
- 🎒 0/1 Knapsack Optimization to select deliveries under weight constraints  
- 📊 Interactive 3D visualization using Plotly (obstacles, goals, routes)  
- ✅ Displays path length, hop count, total value, and total weight carried  
- 🖱️ Streamlit interface for user inputs, randomization, and simulation control  

---

## 🛠 Tech Stack

| Category         | Tools / Libraries              |
|------------------|--------------------------------|
| Language         | Python 3.11                    |
| GUI Framework    | Streamlit                      |
| 3D Visualization | Plotly (Scatter3D, Line3D)     |
| Algorithms       | A*, TSP (Branch & Bound), Knapsack (DP) |
| Others           | NumPy, itertools, PriorityQueue |

---

## 🧪 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/Drone3D-Pathfinder.git
cd Drone3D-Pathfinder
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📋 Algorithms Used

- A* Pathfinding: Uses f(n) = g(n) + h(n) with Euclidean heuristic  
- TSP (Branch and Bound): Generates shortest cycle visiting all selected goals  
- 0/1 Knapsack (DP): Selects highest-value deliveries under max weight

---

## 📈 Output Metrics

- Path Length (Euclidean)
- Hop Count (number of segments)
- Total Value (knapsack)
- Total Weight (knapsack)
- Route Visualization in 3D

---

## 🔮 Future Enhancements

- Dynamic obstacles and re-routing
- Real-time drone simulation with GPS integration
- Battery-aware delivery planning
- Multi-drone coordination and collision avoidance

---

## 🙌 Acknowledgements

- Plotly for powerful 3D visualization  
- Streamlit for interactive Python UI  
- Classic CS algorithms: A*, TSP, Knapsack

svjnsvjlnsv
janvljanv
kjnvljanv
fjknbvlakkb
jnvlanba
ljfnblkanb
ajnbklkba
