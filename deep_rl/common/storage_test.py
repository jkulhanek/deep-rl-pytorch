import unittest
import numpy as np
from .storage import SequenceStorage as ExperienceReplay, SequenceSampler, LambdaSampler, PlusOneSampler

class SequenceStorageTest(unittest.TestCase):
    def assertNumpyArrayEqual(self, a1, a2, msg = 'Arrays must be equal'):
        if not np.array_equal(a1, a2):
            self.fail(msg=f"{a1} != {a2} : " + msg)

    def testShouldStoreAll(self):
        replay = ExperienceReplay(4, 3, samplers = (SequenceSampler(2),))
        replay.insert(1, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay[0][0], 2)
        self.assertEqual(replay[1][0], 4)
        self.assertEqual(replay[2][0], 6)
        self.assertEqual(replay[3][0], 7)

    def testNegativeIndex(self):
        replay = ExperienceReplay(4, 3, samplers = (SequenceSampler(2),))
        replay.insert(1, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay[-4][0], 2)
        self.assertEqual(replay[-3][0], 4)
        self.assertEqual(replay[-2][0], 6)
        self.assertEqual(replay[-1][0], 7)

    def testLength(self):
        replay = ExperienceReplay(4, 3, samplers = (SequenceSampler(2),))
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
        replay = ExperienceReplay(4, 3, samplers = (LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(1, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay.selector_lengths[0], 2)

    def testSamplerStatsRemove(self):
        replay = ExperienceReplay(4, 3, samplers = (LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(6, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [False, False, False, False])
        replay.insert(2, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [False, True, False, False])
        replay.insert(4, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [False, True, True, False])
        replay.insert(6, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [False, True, True, True])
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay.selector_lengths[0], 2)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [False, False, True, True])

    def testSamplingWithEpisodeEnd(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, 3, samplers = (LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, True)
        replay.insert(7, 0, 0.0, False)

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 2)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([6]))

    def testResampling(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, 3, samplers = (LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        toBeSampled = set([4, 6])
        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 2)

        self.assertEqual(len(toBeSampled - wasSampled), 0, 'something was not sampled')
        self.assertEqual(len(wasSampled), len(toBeSampled), 'something was not supposed to be sampled')
        self.assertSetEqual(wasFirst, set([2,4]))

    def testPlusOneSampling(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, 3, samplers = (PlusOneSampler(2),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, True)
        replay.insert(7, 0, 0.0, False)

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][-1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 3)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([7]))

    def testPlusOneResampling(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, 3, samplers = (PlusOneSampler(2),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][-1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 3)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([7]))

if __name__ == '__main__':
    unittest.main()