import unittest
import torch
from deep_rl.actor_critic.unreal.storage import BatchExperienceReplay, SequenceStorage as ExperienceReplay, SequenceSampler, BatchSequenceStorage, LambdaSampler, PlusOneSampler, merge_batches


class SequenceStorageTest(unittest.TestCase):
    def testShouldStoreAll(self):
        replay = ExperienceReplay(4, samplers=(SequenceSampler(2),))
        replay.insert(1, 0, 0.0, False)
        replay.insert(5, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay[0][0], 2)
        self.assertEqual(replay[1][0], 4)
        self.assertEqual(replay[2][0], 6)
        self.assertEqual(replay[3][0], 7)

    def testNegativeIndex(self):
        replay = ExperienceReplay(4, samplers=(SequenceSampler(2),))
        replay.insert(1, 0, 0.0, False)
        replay.insert(5, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay[-4][0], 2)
        self.assertEqual(replay[-3][0], 4)
        self.assertEqual(replay[-2][0], 6)
        self.assertEqual(replay[-1][0], 7)

    def testLength(self):
        replay = ExperienceReplay(4, samplers=(SequenceSampler(2),))
        self.assertEqual(len(replay), 0)
        replay.insert(1, 0, 0.0, False)
        self.assertEqual(len(replay), 1)
        replay.insert(2, 0, 0.0, False)
        self.assertEqual(len(replay), 2)
        replay.insert(4, 0, 0.0, False)
        self.assertEqual(len(replay), 3)
        replay.insert(6, 0, 0.0, False)
        self.assertEqual(len(replay), 4)
        replay.insert(7, 0, 0.0, False)
        self.assertEqual(len(replay), 4)

    def testSamplerStats(self):
        replay = ExperienceReplay(4, samplers=(LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(1, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay.selector_lengths[0], 2)

    def testSamplerStatsRemove(self):
        replay = ExperienceReplay(4, samplers=(LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(6, 0, 0.0, False)
        torch.testing.assert_allclose(replay.selector_data[:, 0], [False, False, False, False])
        replay.insert(2, 0, 0.0, False)
        torch.testing.assert_allclose(replay.selector_data[:, 0], [False, True, False, False])
        replay.insert(4, 0, 0.0, False)
        torch.testing.assert_allclose(replay.selector_data[:, 0], [False, True, True, False])
        replay.insert(6, 0, 0.0, False)
        torch.testing.assert_allclose(replay.selector_data[:, 0], [False, True, True, True])
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay.selector_lengths[0], 2)
        torch.testing.assert_allclose(replay.selector_data[:, 0], [False, False, True, True])

    def testSamplingWithEpisodeEnd(self):
        torch.manual_seed(1)

        replay = ExperienceReplay(4, samplers=(LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(2), torch.tensor(0), torch.tensor(0.0), torch.tensor(True))
        replay.insert(torch.tensor(4), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(True))
        replay.insert(torch.tensor(7), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][1].item())
            wasFirst.add(batch[0][0].item())
            self.assertEqual(batch[0].shape[0], 2)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([6]))

    def testResampling(self):
        torch.manual_seed(1)
        replay = ExperienceReplay(4, samplers=(LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(2), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(4), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(7), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))

        toBeSampled = set([4, 6])
        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][1].item())
            wasFirst.add(batch[0][0].item())
            self.assertEqual(batch[0].shape[0], 2)

        self.assertEqual(len(toBeSampled - wasSampled), 0, 'something was not sampled')
        self.assertEqual(len(wasSampled), len(toBeSampled), 'something was not supposed to be sampled')
        self.assertSetEqual(wasFirst, set([2, 4]))

    def testPlusOneSampling(self):
        torch.manual_seed(1)
        replay = ExperienceReplay(4, samplers=(PlusOneSampler(2),))

        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(2), torch.tensor(0), torch.tensor(0.0), torch.tensor(True))
        replay.insert(torch.tensor(4), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(True))
        replay.insert(torch.tensor(7), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][-1].item())
            wasFirst.add(batch[0][0].item())
            self.assertEqual(batch[0].shape[0], 3)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([7]))

    def testPlusOneResampling(self):
        torch.manual_seed(1)
        replay = ExperienceReplay(4, samplers=(PlusOneSampler(2),))

        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(2), torch.tensor(0), torch.tensor(0.0), torch.tensor(True))
        replay.insert(torch.tensor(4), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(6), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))
        replay.insert(torch.tensor(7), torch.tensor(0), torch.tensor(0.0), torch.tensor(False))

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][-1].item())
            wasFirst.add(batch[0][0].item())
            self.assertEqual(batch[0].shape[0], 3)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([7]))

    def testPlusOneShortMemory(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, samplers=(PlusOneSampler(2),))

        replay.insert(1, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)

        for _ in range(100):
            batch = replay.sample(0)
            self.assertIsNone(batch)


class BatchSequenceStorageTest(unittest.TestCase):
    def testStore(self):
        replay = BatchSequenceStorage(2, 4, samplers=[SequenceSampler(2)])

        replay.insert(torch.tensor([1, 2]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))
        replay.insert(torch.tensor([3, 4]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))
        replay.insert(torch.tensor([5, 6]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))
        replay.insert(torch.tensor([7, 8]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))

    def testSampleShape(self):
        replay = BatchSequenceStorage(2, 4, samplers=[SequenceSampler(2)])
        replay.insert(torch.tensor([1, 2]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))
        replay.insert(torch.tensor([3, 4]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))
        replay.insert(torch.tensor([5, 6]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))
        replay.insert(torch.tensor([7, 8]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), torch.tensor([False, False]))

        sample = replay.sample(0, batch_size=3)
        self.assertEqual(sample[0].shape, (3, 2,))
        self.assertEqual(sample[1].shape, (3, 2,))
        self.assertEqual(sample[2].shape, (3, 2,))
        self.assertEqual(sample[3].shape, (3, 2,))


class StorageUtilTest(unittest.TestCase):
    def testMergeBatches(self):
        batch1 = (torch.ones((2, 5)), [torch.zeros((2, 7)), torch.ones((2,))])
        batch2 = (torch.ones((3, 5)), [torch.zeros((3, 7)), torch.ones((3,))])
        merges = merge_batches(batch1, batch2)

        self.assertIsInstance(merges, tuple)
        self.assertIsInstance(merges[1], list)
        self.assertIsInstance(merges[0], torch.Tensor)

        self.assertTupleEqual(merges[0].shape, (5, 5))
        self.assertTupleEqual(merges[1][0].shape, (5, 7))
        self.assertTupleEqual(merges[1][1].shape, (5,))

    def testZeroBatch(self):
        batch1 = (torch.ones((2, 5)), [torch.zeros((2, 7)), torch.ones((2,))])
        batch2 = []
        merges = merge_batches(batch1, batch2)

        self.assertIsInstance(merges, tuple)
        self.assertIsInstance(merges[1], list)
        self.assertIsInstance(merges[0], torch.Tensor)

        self.assertTupleEqual(merges[0].shape, (2, 5))
        self.assertTupleEqual(merges[1][0].shape, (2, 7))
        self.assertTupleEqual(merges[1][1].shape, (2,))


if __name__ == '__main__':
    unittest.main()


class StorageTest(unittest.TestCase):
    def testRpZerosOnly(self):
        torch.manual_seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(torch.tensor([1, 5, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([2, 6, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 0, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([3, 7, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([4, 8, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        torch.testing.assert_allclose(sequence[2][:, -1], torch.tensor([0, 0, 0]))

    def testRpOnesOnly(self):
        torch.manual_seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(torch.tensor([1, 5, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([2, 6, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 0, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([3, 7, 3]), torch.tensor([1, 1, 2]), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 0]))
        s.insert(torch.tensor([4, 8, 3]), torch.tensor([1, 1, 2]), torch.tensor([1, 1, 1]), torch.tensor([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        torch.testing.assert_allclose(sequence[2][:, -1], torch.tensor([1, 1, 1]))

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
