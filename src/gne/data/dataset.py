from gne.utils.complex import Complex

import torch


class SimplicialDataset(torch.utils.data.Dataset):
    # TODO: write better docstring
    """
    Dataset class that extracts weighted simplices.
    """

    def __init__(self, complex: Complex):
        self.simplices = [s for s in complex if s.weight is not None]

    def __len__(self):
        return len(self.simplices)

    def __getitem__(self, idx):
        return self.simplices[idx]


def collate_unique_vertices(batch: list[Complex]):
    """
    Collate function for DataLoader that combines vertex IDs from a batch of
    Simplex objects into a unique list.

    Args:
        batch (list of Simplex): The batch of Simplex objects from DataLoader.

    Returns:
        list: A list containing unique vertex IDs collected from all simplices
        in the batch.
    """
    set_of_vertex_ids = set()
    for simplex in batch:
        set_of_vertex_ids.update(simplex.vertices)
    return set_of_vertex_ids
