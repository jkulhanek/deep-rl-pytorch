import unittest
import torch
from deep_rl.actor_critic.unreal.storage import BatchExperienceReplay


class StorageTest(unittest.TestCase):
    def testRpZerosOnly(self):
        return
        torch.manual_seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(torch.tensor([1, 5, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([2, 6, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 0, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([3, 7, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([4, 8, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        torch.testing.assert_array_equal(sequence[2], torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]))

    def testRpOnesOnly(self):
        torch.manual_seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(torch.tensor([1, 5, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([2, 6, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 0, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([3, 7, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([4, 8, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 1, 1]), torch.tensor([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        torch.testing.assert_allclose(sequence[2], torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]))

    def testRpNormal(self):
        torch.manual_seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(torch.tensor([1, 5, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([2, 6, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 0, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([3, 7, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([4, 8, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 1, 0]), torch.tensor([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        torch.testing.assert_allclose(sequence[2], torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]]))


if __name__ == '__main__':
    unittest.main()

