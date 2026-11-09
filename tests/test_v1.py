"""Fast integration tests for Grounded Event Graph Reasoner V1."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data import ControlledSCMDataset, padding_collate
from src.models import GroundedEventGraphReasoner, GroundedReasonerConfig
from src.training.losses import EventSetCriterion


class V1IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch = padding_collate(
            [ControlledSCMDataset(size=2, seed=7)[index] for index in range(2)]
        )
        self.model = GroundedEventGraphReasoner(
            GroundedReasonerConfig(
                max_channels=6,
                d_model=32,
                nhead=4,
                num_event_types=9,
                num_attributes=7,
                pred_len=12,
            )
        )

    def test_all_task_shapes_and_finite_values(self) -> None:
        output = self.model(
            self.batch["x"],
            self.batch["time_mask"],
            self.batch["channel_mask"],
            task="all",
        )
        self.assertEqual(output["events"]["type"].shape[-1], 9)
        self.assertEqual(output["events"]["attributes"].shape[-1], 7)
        self.assertEqual(output["outputs"]["anomaly"].shape[-1], self.batch["x"].shape[-1])
        self.assertEqual(output["outputs"]["forecast"].shape[1], 12)
        self.assertTrue(torch.isfinite(output["reconstruction"]).all())

    def test_event_loss_backpropagates(self) -> None:
        output = self.model(
            self.batch["x"],
            self.batch["time_mask"],
            self.batch["channel_mask"],
            task="all",
        )
        targets = []
        for rows in self.batch["events_gt"]:
            channels = torch.zeros(rows.shape[0], self.batch["x"].shape[1])
            channels.scatter_(1, rows[:, 3].long().unsqueeze(1), 1.0)
            targets.append(
                {
                    "types": rows[:, 0].long(),
                    "boundaries": rows[:, 1:3],
                    "channels": channels,
                }
            )
        loss, _ = EventSetCriterion()(output["events"], targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_alpha_zero_matches_base_output(self) -> None:
        self.model._disable_graph_correction = True
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.batch["x"],
                self.batch["time_mask"],
                self.batch["channel_mask"],
                task="all",
            )
        for task in ("morphology", "event_classification", "forecast", "anomaly"):
            base = output["outputs"][f"base/{task}"]
            final = output["outputs"][task]
            self.assertTrue(torch.allclose(base, final, atol=1e-5, rtol=1e-4), task)

    def test_graph_path_is_trainable_at_init(self) -> None:
        self.assertGreater(float(torch.tanh(self.model.graph_gate)), 0.0)
        delta = self.model.graph_to_temporal.project.weight.abs().max()
        self.assertGreater(float(delta), 0.0)

    def test_event_postprocess_rejects_degenerate_periodic(self) -> None:
        from src.evaluation.event_postprocess import (
            EventPostprocessConfig,
            postprocess_event_nodes,
        )

        events = [
            {
                "slot": 0,
                "type": "periodic",
                "start": 0.0,
                "end": 0.0002,
                "start_index": 0,
                "end_index": 0,
                "channel": 0,
                "existence": 0.9,
                "confidence": 0.2,
                "layer": "background",
            }
        ]
        signal = torch.randn(1, 32).numpy()
        kept, rejected = postprocess_event_nodes(
            events,
            signal,
            32,
            config=EventPostprocessConfig(),
        )
        self.assertEqual(kept, [])
        self.assertTrue(rejected)

    def test_extract_events_without_forced_argmax(self) -> None:
        from src.evaluation.sample_detail_log import _extract_events

        events = {
            "existence": torch.tensor([[0.05, 0.03, 0.02, 0.01]]),
            "type": torch.softmax(torch.randn(1, 4, 9), dim=-1),
            "start": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "end": torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
            "channel_scores": torch.tensor([[[1.0], [1.0], [1.0], [1.0]]]),
            "attributes": torch.zeros(1, 4, 7),
            "attribute_mask": torch.zeros(1, 4, 7),
            "confidence": torch.tensor([[0.01, 0.02, 0.03, 0.04]]),
        }
        nodes, valid = _extract_events(events, 0, 32, existence_threshold=0.5)
        self.assertFalse(valid)
        self.assertEqual(nodes, [])

    def test_morphology_event_targets_and_loss(self) -> None:
        from src.data.morphology_events import (
            MorphologyEventDataset,
            discover_morphology_splits,
            labels_to_event_target,
        )

        splits = discover_morphology_splits()
        self.assertGreater(len(splits), 0)
        dataset = MorphologyEventDataset(
            smoke=True, max_events=6, use_raw_channel_only=True
        )
        sample = dataset[0]
        self.assertGreater(sample.x.shape[0], 0)
        self.assertGreater(sample.x.shape[1], 0)
        events = sample.events_gt
        self.assertIsInstance(events, dict)
        self.assertGreater(events["types"].numel(), 0)
        target = labels_to_event_target(
            [
                {
                    "type": "shift_up",
                    "start": 10,
                    "end": 20,
                    "attrs": {"duration": 11.0, "delta_mean": 1.5},
                }
            ],
            length=128,
            raw_channel=np.linspace(0.0, 1.0, 128, dtype=np.float32),
            max_events=6,
            n_channels=13,
        )
        self.assertEqual(int(target["types"][0]), 5)  # level_shift_up
        batch = padding_collate([sample, dataset[min(1, len(dataset) - 1)]])
        collated = batch["events_gt"]
        targets = [
            {key: collated[key][index] for key in collated}
            for index in range(len(collated["types"]))
        ]
        output = self.model(
            batch["x"],
            batch["time_mask"],
            batch["channel_mask"],
            task="all",
        )
        loss, _ = EventSetCriterion()(output["events"], targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_identity_graph_skips_reasoner(self) -> None:
        """identity_graph must feed field embeddings as reasoned_events."""
        self.model.eval()
        self.model._identity_graph = True
        with torch.no_grad():
            output = self.model(
                self.batch["x"],
                self.batch["time_mask"],
                self.batch["channel_mask"],
                task="morphology",
            )
        # reasoned equals the event embeddings produced before GraphReasoner.
        self.assertTrue(
            torch.allclose(output["reasoned_events"], output["events"]["embeddings"])
        )
        self.model._identity_graph = False

    def test_shuffle_edge_pairs_preserves_multiset(self) -> None:
        from src.models.event_graph import shuffle_edge_pairs

        nodes = 3
        scores = torch.zeros(1, nodes, nodes, 2)
        scores[0, 0, 1] = torch.tensor([1.0, 0.0])
        scores[0, 1, 2] = torch.tensor([0.0, 1.0])
        scores[0, 2, 0] = torch.tensor([0.5, 0.5])
        weight = torch.zeros(1, nodes, nodes)
        weight[0, 0, 1] = 1.0
        weight[0, 1, 2] = 1.0
        weight[0, 2, 0] = 1.0
        node_mask = torch.ones(1, nodes, dtype=torch.bool)
        graph = {
            "relation_scores": scores,
            "edge_mask": weight > 0,
            "edge_weight": weight,
            "node_mask": node_mask,
            "existence_probs": node_mask.float(),
            "relation_names": ["A", "B"],
        }
        shuffled = shuffle_edge_pairs(graph, seed=7)
        self.assertTrue(torch.equal(shuffled["node_mask"], node_mask))
        eye = torch.eye(nodes, dtype=torch.bool)
        src, tgt = (~eye).nonzero(as_tuple=True)
        before = scores[0, src, tgt].reshape(-1).sort().values
        after = shuffled["relation_scores"][0, src, tgt].reshape(-1).sort().values
        self.assertTrue(torch.allclose(before, after))
        moved = False
        for seed in range(32):
            alt = shuffle_edge_pairs(graph, seed=seed)
            if not torch.allclose(alt["relation_scores"], scores):
                moved = True
                break
        self.assertTrue(moved)

    def test_shuffle_relation_vectors_keeps_topology(self) -> None:
        from src.models.event_graph import shuffle_relation_vectors

        nodes = 3
        scores = torch.zeros(1, nodes, nodes, 10)
        scores[0, 0, 1] = torch.arange(10, dtype=torch.float)
        scores[0, 1, 2] = torch.arange(10, dtype=torch.float) + 3
        scores[0, 2, 0] = torch.arange(10, dtype=torch.float) - 1
        scores[0, 0, 0] = 9
        weight = torch.zeros(1, nodes, nodes)
        weight[0, 0, 1] = 0.4
        weight[0, 1, 2] = 0.7
        weight[0, 2, 0] = 0.2
        mask = weight > 0
        node_mask = torch.ones(1, nodes, dtype=torch.bool)
        graph = {
            "relation_scores": scores,
            "edge_mask": mask,
            "edge_weight": weight,
            "node_mask": node_mask,
            "existence_probs": node_mask.float(),
            "relation_names": [f"r{i}" for i in range(10)],
        }
        shuffled, stats = shuffle_relation_vectors(graph, seed=3)
        self.assertTrue(torch.equal(shuffled["edge_mask"], mask))
        self.assertTrue(torch.equal(shuffled["edge_weight"], weight))
        self.assertTrue(torch.equal(shuffled["relation_scores"][0, 0, 0], scores[0, 0, 0]))
        src, tgt = mask[0].nonzero(as_tuple=True)
        before = scores[0, src, tgt].reshape(-1).sort().values
        after = shuffled["relation_scores"][0, src, tgt].reshape(-1).sort().values
        self.assertTrue(torch.allclose(before, after))
        self.assertGreater(stats["n_edges"], 1)
        self.assertGreater(stats["change_rate"], 0.0)

    def test_relation_modes_keep_edge_mask(self) -> None:
        self.model.eval()
        with torch.no_grad():
            original = self.model(
                self.batch["x"],
                self.batch["time_mask"],
                self.batch["channel_mask"],
                task="morphology",
            )
        mask = original["graph"]["edge_mask"].clone()
        weight = original["graph"]["edge_weight"].clone()
        for mode in ("shuffled", "no_relation_bias", "no_relation_features"):
            self.model._relation_mode = mode
            self.model._shuffle_relation_seed = 11
            with torch.no_grad():
                output = self.model(
                    self.batch["x"],
                    self.batch["time_mask"],
                    self.batch["channel_mask"],
                    task="morphology",
                )
            self.assertTrue(torch.equal(output["graph"]["edge_mask"], mask))
            self.assertTrue(torch.equal(output["graph"]["edge_weight"], weight))
            self.assertIn("alpha", self.model._last_relation_diag)
            self.assertIn("residual_ratio", self.model._last_relation_diag)
        self.assertEqual(self.model.graph_reasoner.ablate, "no_edges")
        self.model._relation_mode = "no_relation_bias"
        with torch.no_grad():
            self.model(
                self.batch["x"],
                self.batch["time_mask"],
                self.batch["channel_mask"],
                task="morphology",
            )
        self.assertEqual(self.model.graph_reasoner.ablate, "no_edge_bias")
        self.model._relation_mode = ""
        self.model.graph_reasoner.ablate = ""



if __name__ == "__main__":
    unittest.main()
