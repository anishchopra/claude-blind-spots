"""Generator for flowchart path following tasks."""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class FlowchartOutput(BaseModel):
    next_step: str  # The letter label of the next step


class FlowchartGenerator(BaseGenerator):
    """Generate flowchart images to test path following ability."""

    task_name = "flowchart"
    output_model = FlowchartOutput

    BOX_WIDTH = 50
    BOX_HEIGHT = 30
    BOX_COLOR = "#E8F4FD"
    BOX_BORDER_COLOR = "#2980B9"
    ARROW_COLOR = "#2C3E50"
    TEXT_COLOR = "#2C3E50"

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_nodes",
                param_type=int,
                default=7,
                help="Number of nodes in the flowchart",
                min_value=3,
                max_value=20,
            ),
            ParamSpec(
                name="crossing_probability",
                param_type=float,
                default=0.4,
                help="Probability of edge crossings (0.0-1.0)",
                min_value=0.0,
                max_value=1.0,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a flowchart image with a path following question."""
        num_nodes = params.get("num_nodes", 7)
        crossing_prob = params.get("crossing_probability", 0.4)

        # Generate DAG structure
        nodes, edges = self._generate_dag(num_nodes, crossing_prob)

        # Layout nodes in layers
        positions = self._layout_nodes(nodes, edges, crossing_prob)

        # Find a node with exactly one outgoing edge for the question
        query_node, answer_node = self._select_query(nodes, edges)

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw edges first (so boxes are on top)
        self._draw_edges(draw, edges, positions)

        # Draw nodes
        self._draw_nodes(draw, nodes, positions)

        # Count actual crossings for metadata
        num_crossings = self._count_crossings(edges, positions)

        # Create prompt and ground truth
        prompt = f"In this flowchart, which step comes immediately after step {query_node}?"
        ground_truth = FlowchartOutput(next_step=answer_node)

        generation_params = {
            "num_nodes": num_nodes,
            "crossing_probability": crossing_prob,
            "num_edges": len(edges),
            "query_node": query_node,
            "answer_node": answer_node,
            "num_crossings": num_crossings,
            "edges": [(a, b) for a, b in edges],
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _generate_dag(
        self, num_nodes: int, crossing_prob: float
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Generate a DAG structure with letter-labeled nodes.

        Returns:
            Tuple of (list of node labels, list of directed edges)
        """
        # Create node labels (A, B, C, ...)
        nodes = [chr(ord("A") + i) for i in range(num_nodes)]

        # Assign nodes to layers (ensures DAG property)
        # More layers with fewer nodes each = more potential for crossings
        num_layers = max(3, num_nodes // 2)
        layers: list[list[str]] = [[] for _ in range(num_layers)]

        # First node always in first layer, last node always in last layer
        layers[0].append(nodes[0])
        layers[-1].append(nodes[-1])

        # Distribute remaining nodes across middle layers
        remaining = nodes[1:-1]
        random.shuffle(remaining)

        for node in remaining:
            # Prefer middle layers for more crossing potential
            if crossing_prob > 0.5:
                # For high crossing prob, concentrate in fewer layers
                layer_idx = random.randint(1, num_layers - 2)
            else:
                # For low crossing prob, spread out more
                layer_idx = random.randint(0, num_layers - 1)
                if layer_idx == 0 and len(layers[0]) >= 2:
                    layer_idx = 1
                if layer_idx == num_layers - 1 and len(layers[-1]) >= 2:
                    layer_idx = num_layers - 2
            layers[layer_idx].append(node)

        # Remove empty layers
        layers = [layer for layer in layers if layer]

        # Store layer info for each node
        node_layer = {}
        for layer_idx, layer in enumerate(layers):
            for node in layer:
                node_layer[node] = layer_idx

        # Generate edges (only forward edges to maintain DAG)
        edges = []
        nodes_with_outgoing = set()
        nodes_with_incoming = set()

        # Ensure connectivity: each node (except last layer) has at least one outgoing edge
        for layer_idx in range(len(layers) - 1):
            for node in layers[layer_idx]:
                # Connect to at least one node in a later layer
                later_nodes = []
                for future_layer_idx in range(layer_idx + 1, len(layers)):
                    later_nodes.extend(layers[future_layer_idx])

                if later_nodes:
                    # Always add at least one edge
                    target = random.choice(later_nodes)
                    edges.append((node, target))
                    nodes_with_outgoing.add(node)
                    nodes_with_incoming.add(target)

                    # Maybe add more edges for crossing potential
                    if crossing_prob > 0.3 and len(later_nodes) > 1 and random.random() < crossing_prob:
                        other_targets = [n for n in later_nodes if n != target]
                        if other_targets:
                            extra_target = random.choice(other_targets)
                            edges.append((node, extra_target))
                            nodes_with_incoming.add(extra_target)

        # Ensure all nodes (except first layer) have incoming edges
        for layer_idx in range(1, len(layers)):
            for node in layers[layer_idx]:
                if node not in nodes_with_incoming:
                    # Find a node from an earlier layer
                    earlier_nodes = []
                    for prev_layer_idx in range(layer_idx):
                        earlier_nodes.extend(layers[prev_layer_idx])
                    if earlier_nodes:
                        source = random.choice(earlier_nodes)
                        edges.append((source, node))
                        nodes_with_outgoing.add(source)
                        nodes_with_incoming.add(node)

        return nodes, edges

    def _layout_nodes(
        self,
        nodes: list[str],
        edges: list[tuple[str, str]],
        crossing_prob: float,
    ) -> dict[str, tuple[int, int]]:
        """Layout nodes in positions that may create edge crossings.

        Returns:
            Dictionary mapping node labels to (x, y) positions
        """
        width, height = self.image_size
        margin = 60

        # Determine layers from edges
        node_layer = {node: 0 for node in nodes}

        # Topological ordering to assign layers
        changed = True
        while changed:
            changed = False
            for source, target in edges:
                if node_layer[target] <= node_layer[source]:
                    node_layer[target] = node_layer[source] + 1
                    changed = True

        # Group by layer
        layers: dict[int, list[str]] = {}
        for node, layer in node_layer.items():
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        num_layers = max(layers.keys()) + 1

        # Calculate positions
        positions = {}
        layer_x_spacing = (width - 2 * margin) / max(num_layers - 1, 1)

        for layer_idx, layer_nodes in layers.items():
            x = margin + layer_idx * layer_x_spacing

            # Vertical spacing for nodes in this layer
            num_in_layer = len(layer_nodes)
            if num_in_layer == 1:
                y_positions = [height // 2]
            else:
                layer_height = height - 2 * margin
                y_spacing = layer_height / (num_in_layer + 1)
                y_positions = [margin + (i + 1) * y_spacing for i in range(num_in_layer)]

            # Shuffle y positions for more crossing potential
            if crossing_prob > 0.3:
                random.shuffle(y_positions)

            for i, node in enumerate(layer_nodes):
                positions[node] = (int(x), int(y_positions[i]))

        return positions

    def _select_query(
        self, nodes: list[str], edges: list[tuple[str, str]]
    ) -> tuple[str, str]:
        """Select a node to ask about that has exactly one outgoing edge.

        Returns:
            Tuple of (query_node, answer_node)
        """
        # Count outgoing edges for each node
        outgoing: dict[str, list[str]] = {node: [] for node in nodes}
        for source, target in edges:
            outgoing[source].append(target)

        # Find nodes with exactly one outgoing edge
        single_outgoing = [(node, targets[0]) for node, targets in outgoing.items() if len(targets) == 1]

        if single_outgoing:
            return random.choice(single_outgoing)

        # Fallback: pick any node with outgoing edges and use first target
        for node, targets in outgoing.items():
            if targets:
                return (node, targets[0])

        # Should never happen in a valid DAG
        return (nodes[0], nodes[1])

    def _draw_edges(
        self,
        draw: ImageDraw.Draw,
        edges: list[tuple[str, str]],
        positions: dict[str, tuple[int, int]],
    ) -> None:
        """Draw arrows between connected nodes."""
        for source, target in edges:
            x1, y1 = positions[source]
            x2, y2 = positions[target]

            # Adjust start/end points to box edges
            x1 += self.BOX_WIDTH // 2
            x2 -= self.BOX_WIDTH // 2

            # Draw line
            draw.line([(x1, y1), (x2, y2)], fill=self.ARROW_COLOR, width=2)

            # Draw arrowhead
            self._draw_arrowhead(draw, x1, y1, x2, y2)

    def _draw_arrowhead(
        self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        """Draw an arrowhead at the end of a line."""
        import math

        arrow_length = 10
        arrow_angle = math.pi / 6  # 30 degrees

        # Calculate angle of the line
        angle = math.atan2(y2 - y1, x2 - x1)

        # Calculate arrowhead points
        left_x = x2 - arrow_length * math.cos(angle - arrow_angle)
        left_y = y2 - arrow_length * math.sin(angle - arrow_angle)
        right_x = x2 - arrow_length * math.cos(angle + arrow_angle)
        right_y = y2 - arrow_length * math.sin(angle + arrow_angle)

        # Draw arrowhead
        draw.polygon(
            [(x2, y2), (left_x, left_y), (right_x, right_y)],
            fill=self.ARROW_COLOR,
        )

    def _draw_nodes(
        self,
        draw: ImageDraw.Draw,
        nodes: list[str],
        positions: dict[str, tuple[int, int]],
    ) -> None:
        """Draw rectangular boxes with labels."""
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for node in nodes:
            x, y = positions[node]

            # Draw box
            box_coords = [
                x - self.BOX_WIDTH // 2,
                y - self.BOX_HEIGHT // 2,
                x + self.BOX_WIDTH // 2,
                y + self.BOX_HEIGHT // 2,
            ]
            draw.rectangle(box_coords, fill=self.BOX_COLOR, outline=self.BOX_BORDER_COLOR, width=2)

            # Draw label centered in box
            text_bbox = draw.textbbox((0, 0), node, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x - text_width // 2
            text_y = y - text_height // 2
            draw.text((text_x, text_y), node, fill=self.TEXT_COLOR, font=font)

    def _count_crossings(
        self, edges: list[tuple[str, str]], positions: dict[str, tuple[int, int]]
    ) -> int:
        """Count the number of edge crossings in the layout."""
        crossings = 0

        edge_segments = []
        for source, target in edges:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            x1 += self.BOX_WIDTH // 2
            x2 -= self.BOX_WIDTH // 2
            edge_segments.append((x1, y1, x2, y2))

        # Check each pair of edges for crossing
        for i in range(len(edge_segments)):
            for j in range(i + 1, len(edge_segments)):
                if self._segments_cross(edge_segments[i], edge_segments[j]):
                    crossings += 1

        return crossings

    def _segments_cross(
        self, seg1: tuple[int, int, int, int], seg2: tuple[int, int, int, int]
    ) -> bool:
        """Check if two line segments cross each other."""
        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2

        def ccw(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> bool:
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        # Check if segments intersect (excluding shared endpoints)
        if (x1, y1) == (x3, y3) or (x1, y1) == (x4, y4) or (x2, y2) == (x3, y3) or (x2, y2) == (x4, y4):
            return False

        return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(x1, y1, x2, y2, x3, y3) != ccw(
            x1, y1, x2, y2, x4, y4
        )
