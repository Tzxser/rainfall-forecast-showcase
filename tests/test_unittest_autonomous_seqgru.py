"""Synthetic tests to run only on the authorized remote competition platform.

No test in this source has been executed locally.  No project data, trained
checkpoint, CAL or HELD examples are used by these tests.
"""
import unittest
from unittest import mock

import torch
from torch import nn

from precip_forecast.models.autonomous_seqgru import (
    AutonomousSeqGRU,
    CausalSpatialConvGRU,
    EXPECTED_MODEL_PARAMETER_COUNT,
    FRAME_OFFSETS_MINUTES,
    INPUT_KEYS,
    MIXER_PARAMETER_COUNT,
    input_only,
    repeat_latest,
)
from precip_forecast.models.task65_seq4 import (
    INPUT_KEYS as TASK65_INPUT_KEYS,
    Task65Sequence,
    input_only as task65_input_only,
)
from precip_forecast.models.task54_rank import ResidualBlock
from precip_forecast.realdata.task65_radar_sequence import OFFSETS


def synthetic_batch(batch_size=1, height=32, width=48):
    sequence = torch.rand(batch_size, 6, 4, height // 4, width // 4)
    sequence[:, :, 2:] = 1.0
    return dict(
        observation=torch.rand(batch_size, 18, height, width),
        radar=torch.rand(batch_size, 18, height // 8, width // 8),
        nwp=torch.rand(batch_size, 3, 84, height // 8, width // 8),
        nwp_prior_raw_mm=torch.rand(batch_size, 3, height, width),
        nwp_prior_valid=torch.ones(batch_size, 3, height, width),
        modality_available=torch.ones(batch_size, 3),
        radar_seq4=sequence,
        seq_available=torch.ones(batch_size, dtype=torch.bool),
        seq_slot_valid=torch.ones(batch_size, 6, dtype=torch.bool),
    )


class CausalSpatialConvGRUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261001)
        self.module = CausalSpatialConvGRU().eval()

    def test_shape_finite_and_exact_parameter_count(self):
        output = self.module(torch.randn(2, 192, 7, 9))
        self.assertEqual(tuple(output.shape), (2, 64, 7, 9))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(sum(p.numel() for p in self.module.parameters()), MIXER_PARAMETER_COUNT)
        self.assertEqual(MIXER_PARAMETER_COUNT, 166080)
        self.assertEqual(self.module.gates.kernel_size, (3, 3))
        self.assertEqual(self.module.candidate.kernel_size, (3, 3))

    def test_gradient_reaches_all_six_stem_feature_blocks(self):
        value = torch.randn(2, 192, 7, 9, requires_grad=True)
        self.module(value).square().mean().backward()
        gradient = value.grad.reshape(2, 6, 32, 7, 9)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertTrue((gradient.abs().sum(dim=(0, 2, 3, 4)) > 0).all().item())
        for parameter in self.module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_chronological_blocks_reach_gate_in_exact_order(self):
        frames = torch.stack([torch.full((1, 32, 5, 7), float(i)) for i in range(6)], dim=1)
        observed = []
        handle = self.module.gates.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0][:, :32].detach().clone())
        )
        try:
            self.module(frames.reshape(1, 192, 5, 7))
        finally:
            handle.remove()
        self.assertEqual(len(observed), 6)
        for index, frame in enumerate(observed):
            self.assertTrue(torch.equal(frame, frames[:, index]))
        self.assertEqual(FRAME_OFFSETS_MINUTES, OFFSETS)

    def test_later_frame_cannot_change_earlier_gate_outputs(self):
        frames = torch.randn(1, 6, 32, 5, 7)
        changed = frames.clone()
        changed[:, 5] += 2.0
        traces = []
        for value in (frames, changed):
            trace = []
            handle = self.module.gates.register_forward_hook(
                lambda _module, _args, output: trace.append(output.detach().clone())
            )
            try:
                self.module(value.reshape(1, 192, 5, 7))
            finally:
                handle.remove()
            traces.append(trace)
        for first, second in zip(traces[0][:5], traces[1][:5]):
            self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(traces[0][5], traces[1][5]))

    def test_eval_has_no_cross_call_hidden_state(self):
        value = torch.randn(1, 192, 5, 7)
        with torch.no_grad():
            first = self.module(value)
            self.module(torch.randn_like(value))
            repeated = self.module(value)
        self.assertTrue(torch.equal(first, repeated))

    def test_channels_last_has_same_semantic_order(self):
        value = torch.randn(1, 192, 5, 7)
        with torch.no_grad():
            reference = self.module(value)
            actual = self.module(value.contiguous(memory_format=torch.channels_last))
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    def test_spatial_candidate_can_couple_adjacent_grid_cells(self):
        # A controlled synthetic setting proves a 3x3 spatial path exists;
        # it is not a learned-motion/forecast-skill test.
        with torch.no_grad():
            for parameter in self.module.parameters():
                parameter.zero_()
            self.module.candidate.weight[0, 0] = 1.0
            frames = torch.zeros(1, 6, 32, 7, 9)
            frames[0, 5, 0, 3, 4] = 1.0
            result = self.module(frames.reshape(1, 192, 7, 9))
        self.assertGreater(float(result[0, 0, 3, 3]), 0.0)
        self.assertGreater(float(result[0, 0, 2, 4]), 0.0)
        self.assertEqual(float(result[0, 0, 0, 0]), 0.0)

    def test_invalid_schema_rejected(self):
        for shape in ((1, 191, 5, 7), (1, 6, 32, 5, 7), (0, 192, 5, 7), (1, 192, 0, 7)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.module(torch.zeros(shape))
        with self.assertRaises(TypeError):
            self.module(torch.zeros(1, 192, 5, 7, dtype=torch.int64))


class AutonomousSeqGRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261002)
        # These are newly initialized synthetic-test weights, not prior assets.
        cls.model = AutonomousSeqGRU([0.6, 0.25, 0.1, 0.04, 0.01]).eval()
        cls.control = AutonomousSeqGRU([0.6, 0.25, 0.1, 0.04, 0.01], control=True).eval()
        cls.control.load_state_dict(cls.model.state_dict(), strict=True)

    def setUp(self):
        torch.manual_seed(20261003)

    def test_inherited_forward_whitelist_and_post_projection_tail(self):
        self.assertIs(AutonomousSeqGRU.forward, Task65Sequence.forward)
        self.assertEqual(INPUT_KEYS, TASK65_INPUT_KEYS)
        self.assertIs(input_only, task65_input_only)
        self.assertEqual(len(self.model.seq_temporal), 4)
        self.assertIsInstance(self.model.seq_temporal[0], CausalSpatialConvGRU)
        self.assertIsInstance(self.model.seq_temporal[1], nn.GroupNorm)
        self.assertIsInstance(self.model.seq_temporal[2], nn.SiLU)
        self.assertIsInstance(self.model.seq_temporal[3], ResidualBlock)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), EXPECTED_MODEL_PARAMETER_COUNT)
        self.assertEqual(EXPECTED_MODEL_PARAMETER_COUNT, 13415877)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))

    def test_full_model_shape_finite_and_lead_count(self):
        batch = synthetic_batch()
        with torch.no_grad():
            result = self.model(batch)
        self.assertEqual(set(result), {"class_logits"})
        self.assertEqual(tuple(result["class_logits"].shape), (1, 3, 5, 32, 48))
        self.assertTrue(torch.isfinite(result["class_logits"]).all().item())

    def test_same_seed_keeps_all_other_initial_parameters_exact(self):
        torch.manual_seed(20261004)
        original = Task65Sequence()
        torch.manual_seed(20261004)
        changed = AutonomousSeqGRU()
        old_state = {
            key: value for key, value in original.state_dict().items()
            if not key.startswith("seq_temporal.0.")
        }
        new_state = {
            key: value for key, value in changed.state_dict().items()
            if not key.startswith("seq_temporal.0.")
        }
        self.assertEqual(set(old_state), set(new_state))
        for key in old_state:
            self.assertTrue(torch.equal(old_state[key], new_state[key]), key)

    def test_target_or_unapproved_feature_rejected(self):
        for key in ("target_456_mm", "target_valid", "truth", "sampler_category", "is_natural", "future_radar"):
            batch = synthetic_batch()
            batch[key] = torch.zeros(1)
            with self.subTest(key=key), self.assertRaises(PermissionError):
                self.model(batch)
        batch = synthetic_batch()
        del batch["seq_available"]
        with self.assertRaises(PermissionError):
            self.model(batch)

    def test_input_only_drops_training_metadata(self):
        batch = synthetic_batch()
        batch["target_456_mm"] = torch.zeros(1, 3, 32, 48)
        batch["sampler_category"] = "synthetic-only"
        projected = input_only(batch, torch.device("cpu"))
        self.assertEqual(set(projected), set(INPUT_KEYS))
        self.assertNotIn("target_456_mm", projected)

    def test_control_exactly_repeats_latest_usable_frame(self):
        batch = synthetic_batch()
        batch["seq_slot_valid"] = torch.tensor([[True, False, True, False, True, False]])
        masks_before = batch["seq_slot_valid"].clone()
        available_before = batch["seq_available"].clone()
        explicit = dict(batch)
        explicit["radar_seq4"] = repeat_latest(batch["radar_seq4"], batch["seq_slot_valid"])
        with torch.no_grad():
            control_logits = self.control(batch)["class_logits"]
            explicit_logits = self.model(explicit)["class_logits"]
        self.assertTrue(torch.equal(control_logits, explicit_logits))
        self.assertTrue(torch.equal(batch["seq_slot_valid"], masks_before))
        self.assertTrue(torch.equal(batch["seq_available"], available_before))

    def test_all_missing_control_returns_zero_frames(self):
        batch = synthetic_batch()
        batch["seq_available"].zero_()
        batch["seq_slot_valid"].zero_()
        repeated = repeat_latest(batch["radar_seq4"], batch["seq_slot_valid"])
        self.assertEqual(float(repeated.abs().sum()), 0.0)
        with torch.no_grad():
            first = self.control(batch)["class_logits"]
            batch["radar_seq4"] = torch.randn_like(batch["radar_seq4"])
            second = self.control(batch)["class_logits"]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(second).all().item())

    def test_unavailable_sequence_gate_erases_sequence_effect_exactly(self):
        for gate in ("sequence", "radar"):
            batch = synthetic_batch()
            if gate == "sequence":
                batch["seq_available"].zero_()
            else:
                batch["modality_available"][:, 1] = 0
            with torch.no_grad():
                first = self.model(batch)["class_logits"]
                batch["radar_seq4"] = torch.randn_like(batch["radar_seq4"])
                second = self.model(batch)["class_logits"]
            with self.subTest(gate=gate):
                self.assertTrue(torch.equal(first, second))

    def test_nonfinite_missing_sequence_is_not_silently_accepted(self):
        batch = synthetic_batch()
        batch["seq_available"].zero_()
        batch["radar_seq4"][0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(batch)

    def test_raw_sequence_gradient_reaches_all_frames(self):
        batch = synthetic_batch()
        batch["radar_seq4"].requires_grad_()
        self.model.zero_grad(set_to_none=True)
        self.model(batch)["class_logits"].square().mean().backward()
        gradient = batch["radar_seq4"].grad
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertTrue((gradient[:, :, :2].abs().sum(dim=(0, 2, 3, 4)) > 0).all().item())
        self.model.zero_grad(set_to_none=True)

    def test_full_model_eval_repeats_exactly_without_mutating_inputs(self):
        batch = synthetic_batch()
        original = {key: value.clone() for key, value in batch.items()}
        with torch.no_grad():
            first = self.model(batch)["class_logits"]
            second = self.model(batch)["class_logits"]
        self.assertTrue(torch.equal(first, second))
        for key in INPUT_KEYS:
            self.assertTrue(torch.equal(batch[key], original[key]), key)

    def test_constructor_does_not_load_any_checkpoint(self):
        with mock.patch("torch.load", side_effect=AssertionError("Checkpoint access forbidden")):
            fresh = AutonomousSeqGRU()
        self.assertEqual(sum(p.numel() for p in fresh.parameters()), EXPECTED_MODEL_PARAMETER_COUNT)


if __name__ == "__main__":
    unittest.main()
