from gne.utils.complex import Vertex, Simplex, Complex

import torch
import pytest


# =============================================================================
# Tests for Vertex Class
# =============================================================================


def test_vertex_creation():
    v = Vertex("1", 2.5)
    assert v.id == "1", "Vertex ID should be initialized to '1'"
    assert v.weight == 2.5, "Vertex weight should be initialized to 2.5"
    assert v.children == [], "Vertex children list should be initialized empty"


def test_add_and_get_child():
    parent = Vertex("1")
    child = Vertex("2", 1.0)
    parent.add_child(child)

    assert child in parent.children, "Child vertex should be in parent's children list"
    assert (
        parent.get_child(child) == child
    ), "get_child should return the correct child vertex when passed as an object"
    assert (
        parent.get_child("2") == child
    ), "get_child should return the correct child vertex when passed by ID"


def test_add_child_already_exists():
    parent = Vertex("1")
    child = Vertex("2")
    parent.add_child(child)

    # Expecting a warning when adding an existing child
    with pytest.warns(UserWarning) as record:
        parent.add_child(child)
    assert len(record) == 1, "A warning should be raised when adding an existing child"


def test_vertex_equality():
    v1 = Vertex("1")
    v2 = Vertex("1")
    v3 = Vertex("3")

    assert v1 == v2, "Vertices with the same ID should be considered equal"
    assert v1 != v3, "Vertices with different IDs should not be considered equal"


def test_update_weight():
    v = Vertex("1")
    v.update_weight(3.14)
    assert v.weight == 3.14, "Vertex weight should be updated to 3.14"


# =============================================================================
# Tests for Simplex Class
# =============================================================================


def test_simplex_creation():
    simplex = Simplex([3, 1, 2], 0.5)
    assert simplex.vertices == [
        "1",
        "2",
        "3",
    ], "Vertices should be sorted lexicographically as strings"
    assert simplex.weight == 0.5, "Weight should be assigned correctly"


def test_simplex_creation_with_default_weight():
    simplex = Simplex([1, 2, 3])
    assert simplex.vertices == [
        "1",
        "2",
        "3",
    ], "Vertices should be sorted lexicographically as strings"
    assert simplex.weight is None, "Default weight should be None"


def test_simplex_equality():
    simplex1 = Simplex([1, 2, 3], 1.0)
    simplex2 = Simplex([3, 2, 1], 1.0)
    simplex3 = Simplex([1, 2], 1.0)
    assert (
        simplex1 == simplex2
    ), "Simplexes with the same vertices and weight should be equal"
    assert (
        simplex1 != simplex3
    ), "Simplexes with different vertices or weights should not be equal"


def test_simplex_hash():
    simplex1 = Simplex([1, 2, 3], 1.0)
    simplex2 = Simplex([3, 2, 1], 1.0)
    assert hash(simplex1) == hash(
        simplex2
    ), "Hash should be the same for simplexes with the same vertices and weight"


def test_simplex_repr():
    simplex = Simplex([1, 2, 3], 0.5)
    expected_repr = "gne.Simplex(['1', '2', '3'], weight=0.5)"
    assert repr(simplex) == expected_repr, f"Representation should be '{expected_repr}'"


# =============================================================================
# Tests for Complex Class
# =============================================================================


def verify_simplices(complex_instance, expected_simplices):
    """
    Helper function to verify that all expected simplices are present in the complex.

    :param complex_instance: The instance of Complex to verify.
    :param expected_faces: A list of Simplex instances expected to be in the complex.
    """
    simplices = list(iter(complex_instance))
    for expected_simplex in expected_simplices:
        assert (
            expected_simplex in simplices
        ), f"Expected face {expected_simplex} is missing in the complex"


def test_complex_creation():
    # Test creation with no simplices
    complex_empty = Complex()
    assert (
        complex_empty.root.children == []
    ), "Complex initialized with no simplices should have an empty root children list"

    # Test creation with a single simplex
    simplex = Simplex([1, 2], 0.1)
    complex = Complex(simplex)
    expected_simplices = [Simplex([1], None), Simplex([2], None), Simplex([1, 2], 0.1)]
    verify_simplices(complex, expected_simplices)

    # Test creation with a list of simplices
    simplex_list = [Simplex([1, 2], 0.2), Simplex([2, 3], 0.3)]
    complex = Complex(simplices=simplex_list)
    expected_simplices = [
        Simplex([1], None),
        Simplex([2], None),
        Simplex([3], None),
        Simplex([1, 2], 0.2),
        Simplex([2, 3], 0.3),
    ]
    verify_simplices(complex, expected_simplices)


def test_complex_add_simplex():
    # Initialize a Complex instance
    complex_instance = Complex()
    simplex = Simplex([1, 2, 3], 0.5)

    # Add a simplex to the complex
    complex_instance.add_simplex(simplex)

    # Expected simplices in the complex, including the simplex itself and its subfaces
    expected_simplices = [
        Simplex([1], None),
        Simplex([2], None),
        Simplex([3], None),
        Simplex([1, 2], None),
        Simplex([1, 3], None),
        Simplex([2, 3], None),
        Simplex([1, 2, 3], 0.5),
    ]

    # Verify that all expected faces are present in the complex
    verify_simplices(complex_instance, expected_simplices)


def test_complex_update_weight():
    complex = Complex()
    simplex = Simplex([1, 2, 3], 0.5)
    complex.add_simplex(simplex)

    # Update the weight of the simplex
    complex.update_weight(simplex, weight=0.75)

    # Verify the weight has been updated
    assert (
        complex.get_weight(Simplex([1, 2, 3])) == 0.75
    ), "Simplex weight should be updated"


# TODO: Decide wether to include warnings or not.
# def test_complex_get_weight_nonexistent_simplex():
#     complex = Complex()
#     simplex = Simplex([1, 2, 3], 0.5)

#     # Attempt to get the weight of a simplex not added to the complex
#     with pytest.warns(UserWarning):
#         weight = complex.get_weight(simplex)
#         # Verify that the weight is None
#         assert weight is None, "Weight of a nonexistent simplex should be None"


# def test_complex_warning_on_duplicate_simplex():
#     complex = Complex()
#     simplex = Simplex([1, 2, 3], 0.5)
#     complex.add_simplex(simplex)

#     # Attempt to add a duplicate simplex should trigger a warning
#     with pytest.warns(UserWarning):
#         complex.add_simplex(simplex)


def test_complex_copy():
    original = Complex()
    original.add_simplex(Simplex([0]))
    original.add_simplex(Simplex([1]))
    original.add_simplex(Simplex([2]))
    original.add_simplex(Simplex([0, 2]))
    original.add_simplex(Simplex([0, 1, 2]))

    copy = original.copy()

    check_simplices = [s == t for s, t in zip(original, copy)]
    assert all(check_simplices)

    check_weights = [s.weight == t.weight for s, t in zip(original, copy)]
    assert all(check_weights)


def test_complex_get_weights():
    # Test case 1: Complex with no simplices
    complex_empty = Complex()
    weights_empty = complex_empty.get_weights()
    assert weights_empty == {}, "Empty complex should return an empty dictionary"

    # Test case 2: Complex with simplices of different dimensions and weights
    simplex0 = Simplex([1, 2], torch.tensor(0.3))
    simplex1 = Simplex([1, 2, 3], torch.tensor(0.5))
    simplex2 = Simplex([2, 3, 4], torch.tensor(0.8))
    complex = Complex([simplex0, simplex1, simplex2])

    expected_weights = {
        1: torch.tensor([0.3, float("nan"), float("nan"), float("nan"), float("nan")]),
        2: torch.tensor([0.5, 0.8]),
    }
    weights = complex.get_weights()
    assert (
        weights.keys() == expected_weights.keys()
    ), "Incorrect dimensions in weights dictionary"
    for key in weights:
        assert torch.allclose(
            weights[key], expected_weights[key], equal_nan=True
        ), f"Incorrect weight in dimension {key}"

    # Test case 3: Complex with simplices and specified list of simplices
    simplex3 = Simplex([4, 5, 6], torch.tensor(0.2))
    simplex4 = Simplex([5, 6, 7, 8], torch.tensor(0.7))
    complex.add_simplex(simplex3)
    complex.add_simplex(simplex4)

    simplex5 = Simplex([2])
    complex.update_weight(simplex5, torch.tensor(1.0))

    expected_weights_subset = {
        0: torch.tensor([1.0]),
        1: torch.tensor([0.3]),
        2: torch.tensor([0.5, 0.8, 0.2]),
        3: torch.tensor([0.7]),
    }

    weights_subset = complex.get_weights(
        [simplex0, simplex1, simplex2, simplex3, simplex4, simplex5]
    )
    assert (
        weights_subset.keys() == expected_weights_subset.keys()
    ), "Incorrect dimensions in weights dictionary"
    for key in weights_subset:
        assert torch.allclose(
            weights_subset[key], expected_weights_subset[key], equal_nan=True
        ), f"Incorrect weight in dimension {key}"
