from deep_rl.deepq.util import qlearning, double_qlearning
import unittest
import torch


class QLearningTest(unittest.TestCase):
    def testQLearningLoss(self):
        q_tm1 = torch.Tensor([[1, 1, 0], [1, 2, 0]]).float()
        q_t = torch.Tensor([[0, 1, 0], [1, 2, 0]]).float()
        a_tm1 = torch.Tensor([0, 1]).long()
        pcont_t = torch.Tensor([0, 1]).float()
        r_t = torch.Tensor([1, 1]).float()
        loss = qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t)

        # Loss is 0.5 * td_error^2
        self.assertEqual(loss.item(), 0.25)

    def testDoubleQLearningLoss(self):
        q_tm1 = torch.Tensor([[1, 1, 0], [1, 2, 0]]).float()
        q_t = torch.Tensor([[99, 1, 98], [91, 2, 66]]).float()
        a_tm1 = torch.Tensor([0, 1]).long()
        pcont_t = torch.Tensor([0, 1]).float()
        r_t = torch.Tensor([1, 1]).float()
        q_tsel = torch.Tensor([[2, 10, 1], [11, 20, 1]]).float()
        loss = double_qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t, q_tsel)

        # Loss is 0.5 * td_error^2
        self.assertEqual(loss.item(), 0.25)


if __name__ == '__main__':
    unittest.main()
