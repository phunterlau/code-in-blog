import math
import unittest
from method.paper_method import coupling, signed_mean_scores, rank_signed, control_geometry


class PaperMethodTests(unittest.TestCase):
    def test_opposite_responses_cancel(self):
        # RMS would incorrectly prioritize coordinate 0 for this signed target.
        scores = signed_mean_scores([1, 1, 1], [[0,0,0],[0,0,0]], [[10,2,-3],[-10,2,-3]])
        self.assertEqual(scores, [0,2,-3])
        self.assertEqual(rank_signed(scores), [1,0,2])
        self.assertEqual(rank_signed(scores, positive_only=True), [1])

    def test_coupling_sign(self):
        self.assertEqual(coupling([1,-1], [[2,0],[0,3]]), [2,-3])

    def test_geometry_matches_direct_write(self):
        r, v = [2.,1.], [-1.,3.]
        for t in [-2.,-.1,0.,.5,3.]:
            h = [x+t*y for x,y in zip(r,v)]
            expected = sum(x*y for x,y in zip(h,v))/math.sqrt(sum(x*x for x in h)*sum(y*y for y in v))
            self.assertAlmostEqual(control_geometry(r,v,t)['cosine'], expected)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): signed_mean_scores([1], [], [])
        with self.assertRaises(ValueError): rank_signed([float('nan')])
        with self.assertRaises(ValueError): control_geometry([1],[1],-1)

if __name__ == '__main__': unittest.main()
