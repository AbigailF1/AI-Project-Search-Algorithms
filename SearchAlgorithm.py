import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button, TextBox, RadioButtons
from collections import deque
from heapq import heappop, heappush


class GraphVisualizer:
    def __init__(self):
        self.graph = {}
        self.heuristics = {}
        self.start = None
        self.goal = None
        self.algorithm = None
        self.node_colors = {}
        self.edge_colors = {}
        self.visited = set()
        self.queue = deque()
        self.pq = []
        self.dfs_stack = []
        self.depth = 0
        self.path_display = []
        self.goal_found = False

        # Setup the main figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.3, bottom=0.3)

        # Add widgets
        self.ax_menu = plt.axes([0.05, 0.7, 0.2, 0.25])
        self.menu = RadioButtons(self.ax_menu, ["BFS", "DFS", "DLS", "IDDFS", "UCS", "Bidirectional", "Best-First", "A*"])

        self.ax_start = plt.axes([0.05, 0.55, 0.2, 0.05])
        self.start_box = TextBox(self.ax_start, "Start Node:")

        self.ax_goal = plt.axes([0.05, 0.45, 0.2, 0.05])
        self.goal_box = TextBox(self.ax_goal, "Goal Node:")

        self.ax_edge = plt.axes([0.05, 0.35, 0.2, 0.05])
        self.edge_box = TextBox(self.ax_edge, "Edge (u,v,w):")

        self.ax_depth = plt.axes([0.05, 0.25, 0.2, 0.05])
        self.depth_box = TextBox(self.ax_depth, "Depth Limit (DLS):")

        self.ax_apply_depth = plt.axes([0.3, 0.25, 0.1, 0.05])
        self.btn_apply_depth = Button(self.ax_apply_depth, "Apply Depth")
        self.btn_apply_depth.on_clicked(self.apply_depth_limit)

        self.ax_add_edge = plt.axes([0.05, 0.15, 0.1, 0.05])
        self.btn_add_edge = Button(self.ax_add_edge, "Add Edge")

        self.ax_clear_edges = plt.axes([0.15, 0.15, 0.1, 0.05])
        self.btn_clear_edges = Button(self.ax_clear_edges, "Clear Edges")

        self.ax_heuristics = plt.axes([0.05, 0.05, 0.2, 0.05])
        self.heuristic_box = TextBox(self.ax_heuristics, "Node, Heuristic:")

        self.ax_add_heuristic = plt.axes([0.15, 0.05, 0.1, 0.05])
        self.btn_add_heuristic = Button(self.ax_add_heuristic, "Add Heuristic")

        self.ax_start_algo = plt.axes([0.25, 0.05, 0.2, 0.05])
        self.btn_start_algo = Button(self.ax_start_algo, "Start Algorithm")

        self.ax_next = plt.axes([0.45, 0.05, 0.1, 0.05])
        self.btn_next = Button(self.ax_next, "Next Step")

        self.ax_quit = plt.axes([0.55, 0.05, 0.1, 0.05])
        self.btn_quit = Button(self.ax_quit, "Quit")

        # Attach callbacks
        self.btn_add_edge.on_clicked(self.add_edge)
        self.btn_clear_edges.on_clicked(self.clear_edges)
        self.btn_add_heuristic.on_clicked(self.add_heuristic)
        self.btn_start_algo.on_clicked(self.start_algorithm)
        self.btn_next.on_clicked(self.next_step)
        self.btn_quit.on_clicked(self.quit)
        self.menu.on_clicked(self.select_algorithm)

        self.ax_path_display = plt.axes([0.65, 0.05, 0.3, 0.05])
        self.path_textbox = TextBox(self.ax_path_display, "Path:")
        self.path_textbox.set_val("Traversal path will appear here.")
        self.draw_graph()

    def draw_graph(self):
        """Draw the current graph."""
        self.ax.clear()
        G = nx.DiGraph()
        for node, edges in self.graph.items():
            for neighbor, weight in edges:
                G.add_edge(node, neighbor, weight=weight)

        pos = nx.spring_layout(G)

        # Update node and edge colors
        self.node_colors = {node: self.node_colors.get(node, "skyblue") for node in G.nodes}
        self.edge_colors = {(u, v): self.edge_colors.get((u, v), "black") for u in self.graph for v, _ in self.graph[u]}

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color=list(self.node_colors.values()), node_size=500)
        nx.draw_networkx_edges(G, pos, ax=self.ax, edge_color=list(self.edge_colors.values()))
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        nx.draw_networkx_edge_labels(G, pos, ax=self.ax, edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)})

        self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def add_edge(self, event):
        """Add an edge to the graph."""
        edge = self.edge_box.text
        try:
            u, v, w = edge.split(",")
            u, v, w = u.strip(), v.strip(), float(w)
            if u not in self.graph:
                self.graph[u] = []
            self.graph[u].append((v, w))
            if v not in self.graph:
                self.graph[v] = []  # Ensure all nodes are represented in the graph
            self.draw_graph()
        except ValueError:
            print("Invalid edge format. Use 'u,v,w'.")

    def add_heuristic(self, event):
        """Add heuristic values for nodes."""
        heuristic_input = self.heuristic_box.text
        try:
            node, heuristic = heuristic_input.split(",")
            node = node.strip()
            heuristic = float(heuristic.strip())
            self.heuristics[node] = heuristic
            print(f"Added heuristic: {node} -> {heuristic}")
        except ValueError:
            print("Invalid heuristic format. Use 'node, heuristic'.")
    
    def apply_depth_limit(self, event):
        """Apply the depth limit for DLS."""
        try:
            self.depth = int(self.depth_box.text.strip())
            print(f"DLS Depth Limit Applied: {self.depth}")
        except ValueError:
            print("Invalid depth limit. Using default depth of 1.")
            self.depth = 1

    def clear_edges(self, event):
        """Clear all edges and heuristics from the graph."""
        self.graph.clear()
        self.heuristics.clear()
        self.node_colors.clear()
        self.edge_colors.clear()
        self.path_display.clear()
        self.path_textbox.set_val("Traversal path will appear here.")
        self.draw_graph()

    def select_algorithm(self, label):
        """Select the algorithm."""
        self.algorithm = label
        print(f"Selected algorithm: {self.algorithm}")

    def start_algorithm(self, event):
        """Initialize the selected algorithm."""
        self.start = self.start_box.text.strip()
        self.goal = self.goal_box.text.strip()
        self.goal_found = False

        if not self.start or not self.goal:
            print("Start and Goal nodes are required.")
            return

        if not self.graph:
            print("Graph is empty. Please add edges before starting the algorithm.")
            return

        print("Graph structure before starting:", self.graph)

        self.visited.clear()
        self.path_display.clear()

        if self.algorithm == "DLS" and self.depth == 0:
            print("Please set a depth limit using the 'Apply Depth' button.")
            return

        self.node_colors = {node: "skyblue" for node in self.graph}
        self.edge_colors = {(u, v): "black" for u in self.graph for v, _ in self.graph[u]}
        self.draw_graph()

        if self.algorithm == "BFS":
            self.queue.append(self.start)
        elif self.algorithm == "DFS":
            self.dfs_stack.append(self.start)
        elif self.algorithm == "UCS":
            heappush(self.pq, (0, self.start))
        elif self.algorithm in ["Best-First", "A*"]:
            if self.start in self.heuristics:
                heappush(self.pq, (self.heuristics[self.start], self.start))
            else:
                print("Heuristic value for start node is missing.")
        elif self.algorithm == "Bidirectional":
            self.bidirectional_queue = {
                "start": deque([self.start]),
                "goal": deque([self.goal]),
            }

    def next_step(self, event):
        """Perform the next step of the selected algorithm."""
        if self.goal_found:
            return

        if self.algorithm == "BFS":
            self.bfs_step()
        elif self.algorithm == "DFS":
            self.dfs_step()
        elif self.algorithm == "DLS":
            self.dls_step()
        elif self.algorithm == "IDDFS":
            self.iddfs_step()
        elif self.algorithm == "UCS":
            self.ucs_step()
        elif self.algorithm == "Bidirectional":
            self.bidirectional_step()
        elif self.algorithm == "Best-First":
            self.best_first_step()
        elif self.algorithm == "A*":
            self.a_star_step()

        self.draw_graph()

    def bfs_step(self):
        """Perform one step of BFS."""
        if self.queue:
            self.current_node = self.queue.popleft()
            self.node_colors[self.current_node] = "green"
            self.path_display.append(self.current_node)

            if self.current_node == self.goal:
                print("Goal node found!")
                self.node_colors[self.current_node] = "red"
                self.goal_found = True
                self.path_textbox.set_val(f"Goal Node '{self.goal}' Found: " + " -> ".join(self.path_display))
                return

            for neighbor, _ in self.graph[self.current_node]:
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.queue.append(neighbor)
                    self.edge_colors[(self.current_node, neighbor)] = "blue"
        else:
            print("BFS traversal complete.")

    def dfs_step(self):
        """Perform one step of DFS."""
        if self.dfs_stack:
            self.current_node = self.dfs_stack.pop()
            self.node_colors[self.current_node] = "green"
            self.path_display.append(self.current_node)

            if self.current_node == self.goal:
                print("Goal node found!")
                self.node_colors[self.current_node] = "red"
                self.goal_found = True
                self.path_textbox.set_val(f"Goal Node '{self.goal}' Found: " + " -> ".join(self.path_display))
                return

            for neighbor, _ in self.graph[self.current_node]:
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.dfs_stack.append(neighbor)
        else:
            print("DFS traversal complete.")

    def depth_limited_search(self, node, depth):
        """Recursive helper for Depth-Limited Search."""
        if node == self.goal:
            self.path_display.append(node)
            self.node_colors[node] = "green"
            return True
        if depth == 0:
            return False

        self.node_colors[node] = "yellow"  # Node is being processed
        self.draw_graph()

        for neighbor, _ in self.graph[node]:
            if neighbor not in self.visited:
                self.visited.add(neighbor)
                self.edge_colors[(node, neighbor)] = "blue"
                if self.depth_limited_search(neighbor, depth - 1):
                    self.path_display.append(node)
                    return True

        self.node_colors[node] = "skyblue"  # Backtrack
        self.draw_graph()
        return False

    def dls_step(self):
        """Perform Depth-Limited Search."""
        print(f"Running DLS with depth limit: {self.depth}")
        self.visited.clear()
        self.path_display.clear()
        if self.depth_limited_search(self.start, self.depth):
            self.path_display.reverse()  # Reverse path to show from start to goal
            self.goal_found = True
            print("Goal node found!")
            self.path_textbox.set_val(f"Goal Node '{self.goal}' Found: " + " -> ".join(self.path_display))
        else:
            print("Goal node not found.")
            self.path_textbox.set_val(f"Goal Node '{self.goal}' Not Found")


    def iddfs_step(self):
        """Perform Iterative Deepening Depth-First Search."""
        print(f"Running IDDFS with depth limit: {self.depth}")
        self.visited.clear()
        self.path_display.clear()
        if self.depth_limited_search(self.start, self.depth):
            self.path_display.reverse()
            self.goal_found = True
            print("Goal node found!")
            self.path_textbox.set_val(f"Goal Node '{self.goal}' Found: " + " -> ".join(self.path_display))
        else:
            self.depth += 1  # Increment depth limit for the next iteration
            print(f"Increasing depth to {self.depth}")


    def ucs_step(self):
        """Perform one step of UCS."""
        if self.pq:
            cost, self.current_node = heappop(self.pq)
            self.node_colors[self.current_node] = "green"
            self.path_display.append(self.current_node)

            if self.current_node == self.goal:
                print("Goal node found!")
                self.node_colors[self.current_node] = "red"
                self.goal_found = True
                self.path_textbox.set_val(f"Goal Node '{self.goal}' Found: " + " -> ".join(self.path_display))
                return

            for neighbor, weight in self.graph[self.current_node]:
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    heappush(self.pq, (cost + weight, neighbor))
                    self.edge_colors[(self.current_node, neighbor)] = "blue"
        else:
            print("UCS traversal complete.")
    
    def bidirectional_step(self):
        """Perform one step of Bidirectional Search."""
        if not self.bidirectional_queue["start"] and not self.bidirectional_queue["goal"]:
            print("Bidirectional traversal complete.")
            self.path_textbox.set_val(f"Goal Node '{self.goal}' Not Found")
            return

        for direction in ["start", "goal"]:
            if self.bidirectional_queue[direction]:
                current = self.bidirectional_queue[direction].popleft()
                self.node_colors[current] = "green"
                self.path_display.append(current)

                # Check if paths meet
                if current in self.visited:
                    print("Paths meet! Goal node found.")
                    self.node_colors[current] = "red"
                    self.goal_found = True
                    self.path_textbox.set_val(f"Paths Met: " + " -> ".join(self.path_display))
                    return

                self.visited.add(current)
                for neighbor, _ in self.graph[current]:
                    if neighbor not in self.visited:
                        self.bidirectional_queue[direction].append(neighbor)
                        self.edge_colors[(current, neighbor)] = "blue"
    def best_first_step(self):
        """Perform one step of Best-First Search."""
        if self.pq:
            heuristic, self.current_node = heappop(self.pq)
            self.node_colors[self.current_node] = "green"
            self.path_display.append(self.current_node)

            if self.current_node == self.goal:
                print("Goal node found!")
                self.node_colors[self.current_node] = "red"
                self.goal_found = True
                self.path_textbox.set_val(f"Goal Node '{self.goal}' Found: " + " -> ".join(self.path_display))
                return

            for neighbor, _ in self.graph[self.current_node]:
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    heuristic = self.heuristics.get(neighbor, float('inf'))
                    heappush(self.pq, (heuristic, neighbor))
                    self.edge_colors[(self.current_node, neighbor)] = "blue"
        else:
            print("Best-First traversal complete.")

    def a_star_step(self):
        """Perform one step of A* Search."""
        if self.pq:
            cost, self.current_node = heappop(self.pq)
            self.node_colors[self.current_node] = "green"
            self.path_display.append(self.current_node)

            if self.current_node == self.goal:
                print("Goal node found!")
                self.node_colors[self.current_node] = "red"
                self.goal_found = True
                self.path_textbox.set_val(f"Goal Node '{self.goal}' Found: " + " -> ".join(self.path_display))
                return

            for neighbor, weight in self.graph[self.current_node]:
                if neighbor not in self.visited:
                    heuristic = self.heuristics.get(neighbor, float('inf'))
                    total_cost = cost + weight + heuristic
                    heappush(self.pq, (total_cost, neighbor))
                    self.edge_colors[(self.current_node, neighbor)] = "blue"
        else:
            print("A* traversal complete.")


            

    
    def quit(self, event):
        """Exit the program."""
        plt.close(self.fig)


# Start the visualization tool
visualizer = GraphVisualizer()
plt.show()
