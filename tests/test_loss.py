import unittest
import torch

from loss import FocalLoss


class TestFocalLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.num_classes = 1  # Binary classification

        # Create sample logits and targets
        self.logits = torch.randn(self.batch_size, self.num_classes)
        targets_shape = (self.batch_size, self.num_classes)
        self.targets = torch.randint(0, 2, targets_shape).float()

    def test_focal_loss_init(self):
        loss_fn = FocalLoss()
        self.assertEqual(loss_fn.alpha_neg, 0.75)  # 1 - 0.25
        self.assertEqual(loss_fn.alpha_pos, 0.25)
        self.assertEqual(loss_fn.gamma, 2.0)
        self.assertEqual(loss_fn.reduction, 'mean')

    def test_focal_loss_custom_alpha(self):
        loss_fn = FocalLoss(alpha=0.9)
        self.assertAlmostEqual(loss_fn.alpha_neg, 0.1)
        self.assertEqual(loss_fn.alpha_pos, 0.9)

    def test_focal_loss_tuple_alpha(self):
        loss_fn = FocalLoss(alpha=(0.3, 0.7))
        self.assertEqual(loss_fn.alpha_neg, 0.3)
        self.assertEqual(loss_fn.alpha_pos, 0.7)

    def test_focal_loss_forward(self):
        loss_fn = FocalLoss()
        loss = loss_fn(self.logits, self.targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)  # Scalar
        self.assertGreater(loss.item(), 0)

    def test_focal_loss_reduction_none(self):
        loss_fn = FocalLoss(reduction='none')
        loss = loss_fn(self.logits, self.targets)
        self.assertEqual(loss.shape, (self.batch_size,))

    def test_focal_loss_reduction_sum(self):
        loss_fn = FocalLoss(reduction='sum')
        loss = loss_fn(self.logits, self.targets)
        self.assertEqual(loss.ndim, 0)

    def test_focal_loss_gamma_effect(self):
        loss_fn_gamma0 = FocalLoss(gamma=0.0)
        loss_fn_gamma2 = FocalLoss(gamma=2.0)

        loss_gamma0 = loss_fn_gamma0(self.logits, self.targets)
        loss_gamma2 = loss_fn_gamma2(self.logits, self.targets)

        # Higher gamma should generally give different loss
        self.assertNotEqual(loss_gamma0.item(), loss_gamma2.item())

    def test_focal_loss_alpha_effect(self):
        loss_fn_alpha_low = FocalLoss(alpha=0.1)
        loss_fn_alpha_high = FocalLoss(alpha=0.9)

        loss_alpha_low = loss_fn_alpha_low(self.logits, self.targets)
        loss_alpha_high = loss_fn_alpha_high(self.logits, self.targets)

        # Different alpha should give different loss
        self.assertNotEqual(loss_alpha_low.item(), loss_alpha_high.item())

    def test_focal_loss_perfect_predictions(self):
        # When predictions are perfect, loss should be low
        logits_perfect = torch.tensor([[10.0], [-10.0], [10.0], [-10.0]])
        targets_perfect = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        loss_fn = FocalLoss()
        loss = loss_fn(logits_perfect, targets_perfect)
        self.assertLess(loss.item(), 0.1)  # Should be very low


if __name__ == '__main__':
    unittest.main()
