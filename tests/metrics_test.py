import unittest
from deep_rl.metrics import MetricsContext


class MetricContextTest(unittest.TestCase):
    def testPickableMetricContext(self):
        import pickle
        ctx = MetricsContext()

        class MyFile(object):
            def __init__(self):
                self.data = []

            def write(self, stuff):
                self.data.append(stuff)

        pickle.dump(ctx, MyFile())


if __name__ == '__main__':
    unittest.main()
