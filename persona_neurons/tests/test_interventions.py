"""Small CPU tensor tests for the intervention contract; no checkpoints needed."""
import sys
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from runtime import Lab
from experiment import signed_mean, sweep_jobs


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Identity()

    def forward(self, x):
        return self.mlp.down_proj(x)


class ToyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(), ToyLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def toy_lab():
    stack = ToyStack()
    lab = Lab.__new__(Lab)
    lab.model = SimpleNamespace(model=stack)
    lab.layers = stack.layers
    return lab, stack


def edit(index, value, scope='all'):
    return dict(layer=0, indices=[index], mode='add', values=value, scope=scope)


class InterventionTests(unittest.TestCase):
    def test_padding_rows_and_cache_steps(self):
        lab, stack = toy_lab()
        jobs = [{'edits': [edit(1, 7)]}, {'edits': [edit(3, -4)]}]
        x = torch.zeros(2, 3, 5)
        mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        with lab.edits(jobs, mask):
            y = stack(x)
            expected = torch.zeros_like(x)
            expected[0, 1:, 1] = 7
            expected[1, :, 3] = -4
            self.assertTrue(torch.equal(y, expected))
            step = stack(torch.zeros(2, 1, 5))
            self.assertEqual(step[0, 0, 1].item(), 7)
            self.assertEqual(step[1, 0, 3].item(), -4)
        self.assertTrue(torch.equal(x, torch.zeros_like(x)))
        self.assertTrue(torch.equal(stack(x), x))

    def test_scopes_and_exception_cleanup(self):
        lab, stack = toy_lab()
        jobs = [{'edits': [edit(0, 2, 'prefill'), edit(1, 3, 'decode'), edit(2, 4, 'last_prefill')]}]
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with lab.edits(jobs, torch.ones(1, 2)):
                first = stack(torch.zeros(1, 2, 5))
                self.assertTrue(torch.equal(first[0, :, 0], torch.tensor([2., 2.])))
                self.assertEqual(first[0, :, 1].sum().item(), 0)
                self.assertTrue(torch.equal(first[0, :, 2], torch.tensor([0., 4.])))
                later = stack(torch.zeros(1, 1, 5))
                self.assertEqual(later[0, 0, 0].item(), 0)
                self.assertEqual(later[0, 0, 1].item(), 3)
                self.assertEqual(later[0, 0, 2].item(), 0)
                raise RuntimeError('deliberate')
        self.assertTrue(all(not m._forward_pre_hooks for m in stack.modules()))

    def test_signed_mean_preserves_cancellation(self):
        coupling = torch.tensor([[2., -3.]])
        delta = torch.tensor([[[4., 2.]], [[-4., 2.]]])
        self.assertTrue(torch.equal(signed_mean(coupling, delta), torch.tensor([[0., -6.]])))

    def test_control_dose_matches_injected_norm(self):
        weight = torch.tensor([[3., 1., 0.], [4., 0., 2.]])
        lab = SimpleNamespace(L=1, N=3, weight=lambda layer: weight)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ranking.json'
            path.write_text(json.dumps([{'layer': 0, 'index': 0, 'score': 1.}]))
            jobs = sweep_jobs(lab, path, {'evaluation': ['Who are you?'], 'controls': []}, [-10., 10.], 1, 4)
        for dose in [-10., 10.]:
            rows = [r for r in jobs if r.get('dose') == dose]
            norms = []
            for row in rows:
                edit_ = row['edits'][0]
                norms.append(float((weight[:, edit_['indices'][0]] * edit_['values']).norm()))
            self.assertEqual(len(norms), 2)
            self.assertAlmostEqual(norms[0], norms[1])


if __name__ == '__main__':
    unittest.main()
