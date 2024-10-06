from collections import deque
import copy
from typing import Callable
import warnings
from functools import total_ordering

import torch
import bisect


from gne.utils.geometries import Geometry

# TODO: Better names? In particular will prob. want Complex -> SimplicialComplex


@total_ordering
class Vertex:
    """
    Base class for a vertex in a weighted SimplexTree.

    This class represents a node (vertex) within a SimplexTree structure. Each vertex
    can have a weight, a unique identifier, and can be connected to child vertices
    to form simplices.

    Attributes:
        id (str): The unique identifier for the vertex. Defaults to an empty string.
        weight (Optional[float]): The weight of the vertex. None if unweighted.
        children (list[Vertex]): A list of child vertices that form connections
             with this vertex to constitute simplices.

    Public Methods:
        add_child(vertex): Adds a given vertex as a child to this vertex.
        get_child(vertex): Searches for and returns a child vertex matching
                           the given vertex. Returns None if not found.
    """

    def __init__(self, id: str = "", weight: float = None):
        self.id = str(id)
        self.weight = weight
        self.children = []

    def __eq__(self, other):
        # Determines equality with another Vertex based only on the id.
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        out = f"gne.Vertex('{self.id}'"
        if self.weight is not None:
            out += f", weight={self.weight}"
        out += ")"
        return out

    def add_child(self, vertex):
        """Add a child vertex to this vertex's list of children."""
        if any(child == vertex for child in self.children):
            warnings.warn(
                f"Vertex {vertex.id} is already a child of Vertex {self.id}.",
                UserWarning,
            )
        else:
            # self.children.append(vertex)
            bisect.insort(self.children, vertex)

    def get_child(self, input):
        """
        Retrieve child vertex that matches the given Vertex instance or id, if present.

        :param input: A Vertex instance or a string representing the unique identifier.
        :return: The matching child Vertex if found, None otherwise.
        """
        id = str(input) if not isinstance(input, Vertex) else input.id
        for child in self.children:
            if child.id == id:
                return child
        return None

    def update_weight(self, weight):
        self.weight = weight


class Simplex:
    """
    Represents a weighted simplex as a lexicographically ordered list of Vertex ids.

    Remark:
    As opposed to a Complex, a Simplex implicitely includes sub-simplices formed by
    its vertices! For example, Simplex([1, 2, 3]) represents not only the triangular
    area defined by these three points, but also the edges [1,2], [1,3], [2,3] and
    the vertices [1], [2], [3].

    Attributes:
        vertices (list): A list of vertex ids that form the simplex, sorted
            lexicographically. Defaults to an empty list if not provided.
        weight (Optional[float]): An optional weight assigned to the simplex. Defaults
            to None if not provided.
    """

    def __init__(self, vertices: list = None, weight: float = None):
        if vertices is None:
            vertices = []  # Avoid mutable default arguments
        self.vertices = sorted(
            [str(v) for v in vertices]
        )  # Sort vertices lexicographically
        self.weight = weight

    def __eq__(self, other):
        # Determines equality with another Simplex based on vertices.
        return isinstance(other, Simplex) and self.vertices == other.vertices

    def __hash__(self):
        # Provide a hash based on the simplex's vertices and weight.
        return hash((tuple(self.vertices), self.weight))

    def __repr__(self):
        # Return a string representation of the simplex.
        out = f"gne.Simplex({self.vertices}"
        if self.weight is not None:
            out += f", weight={self.weight}"
        out += ")"
        return out


class Complex:
    """
    Represents a simplicial complex as a SimplexTree[^1] for efficient traversal
    and manipulation.

    Attributes:
        root (Vertex): The root vertex of the complex, serving as an entry point
                       for depth-first search (DFS) and other operations.

    Methods:
        _bfs(node=None, simplex=Simplex()): A private method implementing breadth-first
            search starting from a given node.
        _dfs(node=None, simplex=Simplex()): A private method implementing depth-first
            search starting from a given node.
        add_simplex(simplex): Adds a simplex to the complex.
        TODO: remove_simplex(simplex): ...
        update_weight(simplex, weight): Update the weight of a specific simplex.
        get_weight(simplex): Retrieves the weight of a specific simplex.

    References:
    [1] Boissonnat, Jean-Daniel; Maria, Clément (November 2014). "The Simplex Tree:
        an Efficient Data Structure for General Simplicial Complexes". Algorithmica.
        70 (3): 406-427. arXiv:2001.02581. doi:10.1007/s00453-014-9887-3
    """

    def __init__(self, simplices=None):
        self.root = Vertex()
        if simplices is not None:
            if not isinstance(simplices, list):
                simplices = [simplices]  # Ensure simplices is a list
            for simplex in simplices:
                self.add_simplex(simplex)

    def __iter__(self):
        yield from self._bfs(self.root)

    def __eq__(self, other):
        # Determines equality with another Complex based on list of simplices.
        self_simplices = list(self.__iter__())
        other_simplices = list(other.__iter__())
        return isinstance(other, Complex) and self_simplices == other_simplices

    def __repr__(self):
        return f"gne.Complex({[simplex for simplex in self]})"

    def __call__(self, list_of_vertex_ids):
        subcomplex = Complex()

        set_of_vertex_ids = set(list_of_vertex_ids)
        for simplex in self.__iter__():
            if set(simplex.vertices).issubset(set_of_vertex_ids):
                subcomplex.add_simplex(simplex)

        return subcomplex

    def _bfs(self, node=None, simplex=Simplex()):
        """
        Implements an iterative breadth-first search starting from the root. Yields a
        list of (weighted) simplices.

        :param node: The starting node for the BFS.
        :param simplex: The initial simplex being expanded during the BFS.
        """
        if node is None:
            return

        queue = deque([(node, simplex)])

        while queue:
            current_node, current_simplex = queue.popleft()

            if current_simplex.vertices:
                yield current_simplex

            for child in current_node.children:
                new_simplex = Simplex(
                    current_simplex.vertices + [child.id], child.weight
                )
                queue.append((child, new_simplex))

    def _dfs(self, node=None, simplex=Simplex()):
        """
        Implements a recursive depth-first search starting from the root. Yields a list
        of (weighted) simplices.

        :param node: The starting node for the DFS.
        :param simplex: The current simplex being constructed during the DFS.
        """
        if node is None:
            return

        if simplex.vertices:
            yield simplex

        for child in node.children:
            new_simplex = Simplex(simplex.vertices + [child.id], child.weight)
            yield from self._dfs(child, new_simplex)

    def _get_terminal_node(self, simplex: Simplex):
        """
        Helper method to find the terminal node of a simplex in the simplex tree.

        :param simplex: The simplex to be located.
        :return: The terminal Vertex object if found, None otherwise.
        """
        current_node = self.root
        for vertex_id in simplex.vertices:
            current_node = current_node.get_child(vertex_id)
            if current_node is None:
                return None
        return current_node

    def _add_nodes(self, current_node, vertices, weight, depth=0, terminal_depth=None):
        """
        Recursively adds nodes that represent a simplex and all its subsimplices.

        :param current_node: The current node in the simplex tree being processed.
        :param vertices: The remaining vertices of the simplex to add.
        :param weight: The weight to assign to the full simplex.
        :param is_terminal: Indicates if the current path represents the full simplex.
        """
        if not vertices:
            return

        if terminal_depth is None:
            terminal_depth = len(vertices) - 1

        # Add new node and move forward
        for i, vertex_id in enumerate(copy.deepcopy(vertices)):
            new_node = Vertex(vertex_id, weight if depth == terminal_depth else None)
            existing_node = current_node.get_child(new_node)
            if existing_node is None:  # If this child does not exist, add new_node
                current_node.add_child(new_node)
                next_node = new_node
            else:  # If the child exists, just move to it
                if depth == terminal_depth:
                    return
                next_node = existing_node

            # Recursion over remaining vertices, mark as terminal if we're moving to
            # the last vertex and length of chain is
            self._add_nodes(
                next_node, vertices[i + 1 :], weight, depth + 1, terminal_depth
            )

    def add_simplex(self, simplex: Simplex):
        """
        Adds a simplex and all its subsimplices to the complex by traversing from the
        root and adding vertices as needed to represent each simplex in the tree.

        :param simplex: The simplex to be added.
        """
        # Start recursively adding nodes from the root
        self._add_nodes(self.root, simplex.vertices, simplex.weight)

    def update_weight(self, simplex: Simplex, weight: torch.float):
        """
        Updates the weight of a specific simplex.

        :param simplex: The simplex whose weight is to be set.
        :param weight: The new weight of the simplex.
        """
        terminal_node = self._get_terminal_node(simplex)
        if terminal_node:
            terminal_node.update_weight(weight)

    def get_weight(self, simplex: Simplex):
        """
        Retrieves the weight of a specific simplex in the complex.

        :param simplex: The simplex whose weight is to be retrieved.
        :return: The weight of the simplex if found, None otherwise.
        """
        terminal_node = self._get_terminal_node(simplex)
        return terminal_node.weight if terminal_node else None

    def get_weights(self, simplices: list = None):
        """
        Retrieves the weights for a list of simplices in the complex, organized
        by dimension.

        :param simplices: The list of simplices for which weights are to be retrieved.
        :return: Dictionary with dimensions as keys and torch tensors of
            weights as values.
        """
        if simplices is None:
            simplices = list(
                self.__iter__()
            )  # Assuming __iter__ yields all simplices in the complex

        weights_by_dimension = {}

        for simplex in simplices:
            dimension = len(simplex.vertices) - 1  # dimension of the simplex
            weight = self.get_weight(simplex)

            # Initialize the list for this dimension if it does not exist
            if dimension not in weights_by_dimension:
                weights_by_dimension[dimension] = []

            if weight is not None:
                weights_by_dimension[dimension].append(weight)
            else:
                weights_by_dimension[dimension].append(torch.tensor(torch.nan))

        # remove dims where all entries are nan
        keys_to_remove = []

        for dim in weights_by_dimension:
            # Convert list to tensor and check for all NaNs
            tensor = torch.stack(weights_by_dimension[dim])
            if torch.isnan(tensor).all():
                keys_to_remove.append(dim)
            else:
                weights_by_dimension[dim] = tensor

        for key in keys_to_remove:
            del weights_by_dimension[key]

        return weights_by_dimension

    def copy(self, include_weights=True):
        copy = Complex()
        for s in self:
            if include_weights:
                copy.add_simplex(s)
            else:
                copy.add_simplex(Simplex(s.vertices))
        return copy

    def normalize(self):
        max_depth = max([len(s.vertices) for s in self])
        for k in range(1, max_depth + 1):
            simplices = [
                s for s in self if len(s.vertices) == k and s.weight is not None
            ]
            weights = self.get_weights(simplices)
            weights = weights / weights.sum()
            for i, s in enumerate(simplices):
                self.update_weight(s, weights[i])

    # TODO: Make sure this is consistent with easy creation for subclasses like
    # KneighboursComplex, where one wants to automatically use
    # creator_func = kneighbours_complex.
    # In particular, I think that I want to redefine this method for subclasses, s.t.
    # creator_func is fixed. E.g. KneighbourComplex is a subclass of Complex
    @classmethod
    def creator(
        cls,
        creator_func: Callable[..., "Geometry"],
        *args,
        **kwargs,
    ) -> "Complex":
        # # validate obj before proceeding
        # if not isinstance(geometry, Geometry):
        #     raise ValueError(f"{geometry} is not an instance of the Geometry class")
        # if not hasattr(geometry, "sample"):
        #     raise ValueError("Geometry does not have sample points")

        # Use the creator function to create an instance
        instance = creator_func(*args, **kwargs)

        # Validate creator_func returns a Complex instance
        if not isinstance(instance, Complex):
            raise TypeError("creator_func must return an instance of Complex")

        return instance
