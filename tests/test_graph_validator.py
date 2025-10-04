"""Unit tests for graph validation and connectivity checks."""

import pytest
from dynhnsw.graph_validator import GraphValidator


class TestGraphConstruction:
    """Test basic graph construction operations."""

    def test_initial_graph_empty(self):
        """New validator should have empty graph."""
        validator = GraphValidator()
        assert len(validator.get_all_nodes()) == 0

    def test_add_single_edge(self):
        """Adding edge should create both nodes."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        nodes = validator.get_all_nodes()
        assert 1 in nodes
        assert 2 in nodes

    def test_add_edge_creates_bidirectional_connection(self):
        """Edge should be bidirectional (undirected graph)."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert 2 in validator.get_neighbors(1)
        assert 1 in validator.get_neighbors(2)

    def test_add_multiple_edges(self):
        """Should handle multiple edges correctly."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)
        validator.add_edge(1, 3)

        assert len(validator.get_all_nodes()) == 3

    def test_remove_edge(self):
        """Should remove edge from both directions."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.remove_edge(1, 2)

        assert 2 not in validator.get_neighbors(1)
        assert 1 not in validator.get_neighbors(2)


class TestConnectivityChecks:
    """Test connectivity and reachability checks."""

    def test_directly_connected_nodes(self):
        """Directly connected nodes should be reachable."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert validator.is_connected(1, 2)
        assert validator.is_connected(2, 1)  # Symmetric

    def test_indirectly_connected_nodes(self):
        """Nodes connected via path should be reachable."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)

        assert validator.is_connected(1, 3)
        assert validator.is_connected(3, 1)

    def test_disconnected_nodes(self):
        """Disconnected nodes should not be reachable."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(3, 4)

        assert not validator.is_connected(1, 3)
        assert not validator.is_connected(2, 4)

    def test_self_connection(self):
        """Node should always be connected to itself."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert validator.is_connected(1, 1)
        assert validator.is_connected(2, 2)

    def test_nonexistent_node(self):
        """Non-existent node should not be connected."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert not validator.is_connected(1, 99)


class TestDisconnectionDetection:
    """Test detection of edges whose removal would disconnect graph."""

    def test_removing_bridge_edge_disconnects(self):
        """Removing a bridge edge should disconnect graph."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)

        # Edge (2,3) is a bridge - removing it disconnects 3 from rest
        assert validator.would_disconnect_graph(2, 3)

    def test_removing_non_bridge_edge_stays_connected(self):
        """Removing non-bridge edge with alternative path should keep connectivity."""
        validator = GraphValidator()
        # Create triangle: 1-2, 2-3, 1-3
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)
        validator.add_edge(1, 3)

        # Any edge can be removed without disconnecting
        assert not validator.would_disconnect_graph(1, 2)
        assert not validator.would_disconnect_graph(2, 3)
        assert not validator.would_disconnect_graph(1, 3)

    def test_simple_chain_all_bridges(self):
        """In a chain, all edges are bridges."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)
        validator.add_edge(3, 4)

        assert validator.would_disconnect_graph(1, 2)
        assert validator.would_disconnect_graph(2, 3)
        assert validator.would_disconnect_graph(3, 4)

    def test_disconnection_check_restores_edge(self):
        """Disconnection check should not permanently modify graph."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        validator.would_disconnect_graph(1, 2)

        # Edge should still exist after check
        assert validator.is_connected(1, 2)


class TestNodeDegree:
    """Test node degree computation."""

    def test_isolated_node_degree_zero(self):
        """Node with no edges should have degree 0."""
        validator = GraphValidator()
        validator.graph[1] = set()  # Add isolated node

        assert validator.get_node_degree(1) == 0

    def test_node_with_one_neighbor(self):
        """Node with one edge should have degree 1."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert validator.get_node_degree(1) == 1
        assert validator.get_node_degree(2) == 1

    def test_node_with_multiple_neighbors(self):
        """Node degree should equal number of neighbors."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(1, 3)
        validator.add_edge(1, 4)

        assert validator.get_node_degree(1) == 3

    def test_nonexistent_node_degree(self):
        """Non-existent node should have degree 0."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert validator.get_node_degree(99) == 0


class TestPathLength:
    """Test shortest path length computation."""

    def test_path_length_direct_connection(self):
        """Direct connection should have path length 1."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert validator._bfs_path_length(1, 2) == 1

    def test_path_length_two_hops(self):
        """Path through intermediate node should have correct length."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)

        assert validator._bfs_path_length(1, 3) == 2

    def test_path_length_self(self):
        """Path to self should be 0."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        assert validator._bfs_path_length(1, 1) == 0

    def test_path_length_no_connection(self):
        """No path should return -1."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(3, 4)

        assert validator._bfs_path_length(1, 3) == -1

    def test_path_length_complex_graph(self):
        """Should find shortest path in complex graph."""
        validator = GraphValidator()
        # Create graph: 1-2-3-4 with shortcut 1-4
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)
        validator.add_edge(3, 4)
        validator.add_edge(1, 4)  # Shortcut

        # Should use shortcut (length 1) not long path (length 3)
        assert validator._bfs_path_length(1, 4) == 1


class TestGraphStatistics:
    """Test graph statistics computation."""

    def test_statistics_empty_graph(self):
        """Empty graph should return zero statistics."""
        validator = GraphValidator()

        stats = validator.get_graph_statistics()

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["avg_degree"] == 0.0

    def test_statistics_single_edge(self):
        """Single edge should give correct statistics."""
        validator = GraphValidator()
        validator.add_edge(1, 2)

        stats = validator.get_graph_statistics()

        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["avg_degree"] == 1.0

    def test_statistics_triangle_graph(self):
        """Triangle graph should have correct statistics."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)
        validator.add_edge(1, 3)

        stats = validator.get_graph_statistics()

        assert stats["node_count"] == 3
        assert stats["edge_count"] == 3
        assert stats["avg_degree"] == pytest.approx(2.0)
        assert stats["min_degree"] == 2
        assert stats["max_degree"] == 2

    def test_statistics_star_graph(self):
        """Star graph should show correct degree distribution."""
        validator = GraphValidator()
        # Center node (1) connected to 4 others
        for i in range(2, 6):
            validator.add_edge(1, i)

        stats = validator.get_graph_statistics()

        assert stats["node_count"] == 5
        assert stats["edge_count"] == 4
        assert stats["min_degree"] == 1  # Leaf nodes
        assert stats["max_degree"] == 4  # Center node


class TestAveragePathLength:
    """Test average path length computation."""

    def test_average_path_length_small_graph(self):
        """Should compute correct average for small graph."""
        validator = GraphValidator()
        validator.add_edge(1, 2)
        validator.add_edge(2, 3)

        # Paths: 1-2 (1), 2-3 (1), 1-3 (2) → avg = 1.33
        avg_path = validator.compute_average_path_length(sample_size=10)

        assert avg_path > 0
        assert avg_path < 3  # Should be reasonable

    def test_average_path_length_empty_graph(self):
        """Empty graph should return -1."""
        validator = GraphValidator()

        avg_path = validator.compute_average_path_length()

        assert avg_path == -1.0

    def test_average_path_length_single_node(self):
        """Single node graph should return -1."""
        validator = GraphValidator()
        validator.graph[1] = set()

        avg_path = validator.compute_average_path_length()

        assert avg_path == -1.0
