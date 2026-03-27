import unittest

import torch

from trace_interpreter import exclusive_counter_scan, naive_slot_memory, vectorized_slot_memory


class TraceInterpreterTests(unittest.TestCase):
    def test_vectorized_slot_memory_matches_naive_scan(self):
        torch.manual_seed(0)
        batch, seq_len, slots, mem_dim = 2, 11, 5, 7
        slot_ids = torch.randint(0, slots, (batch, seq_len))
        write_slots = torch.nn.functional.one_hot(slot_ids, num_classes=slots).float()
        write_values = torch.randn(batch, seq_len, mem_dim)

        expected = naive_slot_memory(write_slots, write_values)
        actual, _ = vectorized_slot_memory(write_slots, write_values)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_exclusive_counter_scan(self):
        deltas = torch.tensor(
            [[[1.0, -1.0], [2.0, 3.0], [-4.0, 5.0], [1.0, 1.0]]]
        )
        expected = torch.tensor(
            [[[0.0, 0.0], [1.0, -1.0], [3.0, 2.0], [-1.0, 7.0]]]
        )
        actual = exclusive_counter_scan(deltas)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
