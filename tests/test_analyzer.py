"""Unit tests for analyzer module."""

import unittest
import pandas as pd

from analyzer import (
    DEFAULT_TRIVIAL_TAGS,
    TOPIC_DAG,
    compute_tag_metrics,
    get_priority_weak_tags,
    get_stuck_problems,
    get_verdict_and_wall_stats,
)


class TestAnalyzer(unittest.TestCase):
    """Test suite for compute_tag_metrics and get_priority_weak_tags in analyzer.py."""

    def setUp(self):
        # Sample submissions dataset:
        # Problem 1: 1915G (tags: ["graphs", "shortest paths"], rating: 1800)
        #   Submissions: WA, WA, OK -> struggled (2 failed attempts, solved)
        # Problem 2: 1800A (tags: ["greedy", "two pointers"], rating: 900)
        #   Submissions: OK -> 0 failed attempts, solved
        # Problem 3: 1700B (tags: ["graphs", "dfs and similar"], rating: 1600)
        #   Submissions: WA, TLE -> never solved (2 failed attempts)
        # Problem 4: 1600C (tags: ["dp", "trees"], rating: 1500)
        #   Submissions: WA -> never solved (1 failed attempt)
        # Problem 5: 1500D (tags: ["implementation", "*special"], rating: 800)
        #   Submissions: WA -> trivial tags, should be filtered out
        self.submissions_df = pd.DataFrame([
            {
                "problem_id": "1915G",
                "name": "Bicycles",
                "rating": 1800,
                "tags": ["graphs", "shortest paths"],
                "verdict": "WRONG_ANSWER",
                "creationTimeSeconds": 1705000000,
            },
            {
                "problem_id": "1915G",
                "name": "Bicycles",
                "rating": 1800,
                "tags": ["graphs", "shortest paths"],
                "verdict": "WRONG_ANSWER",
                "creationTimeSeconds": 1705000100,
            },
            {
                "problem_id": "1915G",
                "name": "Bicycles",
                "rating": 1800,
                "tags": ["graphs", "shortest paths"],
                "verdict": "OK",
                "creationTimeSeconds": 1705000200,
            },
            {
                "problem_id": "1800A",
                "name": "Greedy Easy",
                "rating": 900,
                "tags": ["greedy", "two pointers"],
                "verdict": "OK",
                "creationTimeSeconds": 1705000300,
            },
            {
                "problem_id": "1700B",
                "name": "Graph Hard",
                "rating": 1600,
                "tags": ["graphs", "dfs and similar"],
                "verdict": "WRONG_ANSWER",
                "creationTimeSeconds": 1705000400,
            },
            {
                "problem_id": "1700B",
                "name": "Graph Hard",
                "rating": 1600,
                "tags": ["graphs", "dfs and similar"],
                "verdict": "TIME_LIMIT_EXCEEDED",
                "creationTimeSeconds": 1705000500,
            },
            {
                "problem_id": "1600C",
                "name": "Tree DP",
                "rating": 1500,
                "tags": ["dp", "trees"],
                "verdict": "WRONG_ANSWER",
                "creationTimeSeconds": 1705000600,
            },
            {
                "problem_id": "1500D",
                "name": "Special Trivial",
                "rating": 800,
                "tags": ["implementation", "*special"],
                "verdict": "WRONG_ANSWER",
                "creationTimeSeconds": 1705000700,
            },
        ])

    def test_compute_tag_metrics_computations(self):
        """Test tag grouping, unique_solved, failed_attempts, and avg_solved_rating."""
        user_rating = 1700
        metrics = compute_tag_metrics(self.submissions_df, user_rating=user_rating)

        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertFalse(metrics.empty)

        # Ensure trivial tags were filtered out
        tag_list = metrics["tag"].tolist()
        self.assertNotIn("implementation", tag_list)
        self.assertNotIn("*special", tag_list)

        # Inspect 'graphs' tag:
        # Problems attempted: 1915G (solved, 2 WA) and 1700B (unsolved, 1 WA + 1 TLE)
        # unique_solved = 1 (1915G)
        # failed_attempts = 4 (2 on 1915G + 2 on 1700B)
        # avg_solved_rating = 1800.0 (from 1915G)
        graphs_row = metrics[metrics["tag"] == "graphs"].iloc[0]
        self.assertEqual(graphs_row["unique_solved"], 1)
        self.assertEqual(graphs_row["problems_attempted"], 2)
        self.assertEqual(graphs_row["failed_attempts"], 4)
        self.assertEqual(graphs_row["avg_solved_rating"], 1800.0)

        # Inspect 'shortest paths':
        # Problem 1915G: unique_solved = 1, failed_attempts = 2, avg_solved_rating = 1800.0
        sp_row = metrics[metrics["tag"] == "shortest paths"].iloc[0]
        self.assertEqual(sp_row["unique_solved"], 1)
        self.assertEqual(sp_row["failed_attempts"], 2)
        self.assertEqual(sp_row["avg_solved_rating"], 1800.0)

        # Inspect 'dfs and similar':
        # Problem 1700B: unique_solved = 0, failed_attempts = 2, avg_solved_rating = 0.0
        dfs_row = metrics[metrics["tag"] == "dfs and similar"].iloc[0]
        self.assertEqual(dfs_row["unique_solved"], 0)
        self.assertEqual(dfs_row["failed_attempts"], 2)
        self.assertEqual(dfs_row["avg_solved_rating"], 0.0)

        # Ensure sorted from weakest (highest weakness_score) to strongest
        scores = metrics["weakness_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_compute_tag_metrics_empty(self):
        """Test that empty submissions DataFrame returns empty metrics DataFrame."""
        empty_df = pd.DataFrame()
        res = compute_tag_metrics(empty_df, user_rating=1500)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue(res.empty)
        self.assertIn("weakness_score", res.columns)

    def test_topic_dag_contains_expected_mappings(self):
        """Verify TOPIC_DAG dictionary maps foundational topics to advanced extensions."""
        self.assertIn("dp", TOPIC_DAG)
        self.assertIn("trees", TOPIC_DAG["dp"])
        self.assertIn("bitmasks", TOPIC_DAG["dp"])
        self.assertIn("graphs", TOPIC_DAG["dp"])

        self.assertIn("binary search", TOPIC_DAG)
        self.assertIn("greedy", TOPIC_DAG["binary search"])
        self.assertIn("two pointers", TOPIC_DAG["binary search"])

        self.assertIn("graphs", TOPIC_DAG)
        self.assertIn("shortest paths", TOPIC_DAG["graphs"])
        self.assertIn("dfs and similar", TOPIC_DAG["graphs"])
        self.assertIn("dsu", TOPIC_DAG["graphs"])

    def test_get_priority_weak_tags_dag_boost(self):
        """Test foundational prerequisite boosting when both parent and child appear weak."""
        # Construct synthetic weakness_df where:
        # child 'shortest paths' initially has score 50.0
        # child 'dfs and similar' initially has score 48.0
        # parent 'graphs' initially has score 45.0 (ranked #3)
        # unrelated 'dp' has score 40.0
        weakness_df = pd.DataFrame([
            {"tag": "shortest paths", "weakness_score": 50.0},
            {"tag": "dfs and similar", "weakness_score": 48.0},
            {"tag": "graphs", "weakness_score": 45.0},
            {"tag": "dp", "weakness_score": 40.0},
        ])

        # Without boost, order is: ["shortest paths", "dfs and similar", "graphs"]
        # With DAG boost, 'graphs' has 2 weak children ('shortest paths', 'dfs and similar').
        # With boost_factor=0.35, graphs gets boosted by: 45.0 * (1 + 0.35 * 2) = 45.0 * 1.7 = 76.5
        # So 'graphs' should jump to rank #1!
        top_tags = get_priority_weak_tags(weakness_df, top_k=3, boost_factor=0.35)
        self.assertIsInstance(top_tags, list)
        self.assertEqual(len(top_tags), 3)
        self.assertEqual(top_tags[0], "graphs")

    def test_get_priority_weak_tags_details(self):
        """Test returning detailed DataFrame with boost explanation."""
        weakness_df = pd.DataFrame([
            {"tag": "shortest paths", "weakness_score": 50.0},
            {"tag": "graphs", "weakness_score": 45.0},
        ])

        details_df = get_priority_weak_tags(weakness_df, top_k=2, return_details=True)
        self.assertIsInstance(details_df, pd.DataFrame)
        self.assertIn("boosted_score", details_df.columns)
        self.assertIn("boost_reason", details_df.columns)

        graphs_entry = details_df[details_df["tag"] == "graphs"].iloc[0]
        self.assertGreater(graphs_entry["boosted_score"], 45.0)
        self.assertIn("shortest paths", graphs_entry["boost_reason"])

    def test_get_stuck_problems(self):
        """Test finding unsolved problems sorted by attempt_count descending."""
        stuck_df = get_stuck_problems(self.submissions_df)
        self.assertIsInstance(stuck_df, pd.DataFrame)
        self.assertFalse(stuck_df.empty)

        # Check required columns
        expected_cols = [
            "problem_id",
            "name",
            "rating",
            "tags",
            "attempt_count",
            "most_frequent_verdict",
            "url",
        ]
        self.assertListEqual(list(stuck_df.columns), expected_cols)

        # Problem 1915G was solved -> must NOT be in stuck problems
        stuck_pids = stuck_df["problem_id"].tolist()
        self.assertNotIn("1915G", stuck_pids)
        self.assertNotIn("1800A", stuck_pids)

        # 1700B (2 failed attempts) and 1600C (1 failed attempt) must be present
        self.assertIn("1700B", stuck_pids)
        self.assertIn("1600C", stuck_pids)

        # 1700B should be ranked before 1600C (2 attempts vs 1 attempt)
        self.assertEqual(stuck_df.iloc[0]["problem_id"], "1700B")
        self.assertEqual(stuck_df.iloc[0]["attempt_count"], 2)
        self.assertEqual(stuck_df.iloc[0]["url"], "https://codeforces.com/contest/1700/problem/B")

    def test_get_verdict_and_wall_stats(self):
        """Test tag-wise verdict failure breakdown and rating wall detection."""
        stats = get_verdict_and_wall_stats(self.submissions_df, user_rating=1600)
        self.assertIsInstance(stats, dict)
        self.assertIn("tag_verdicts_df", stats)
        self.assertIn("rating_stats_df", stats)
        self.assertIn("rating_wall", stats)
        self.assertIn("rating_wall_stats", stats)
        self.assertIn("automated_diagnoses", stats)

        # Verify tag_verdicts_df structure
        tv_df = stats["tag_verdicts_df"]
        self.assertIsInstance(tv_df, pd.DataFrame)
        self.assertIn("WRONG_ANSWER", tv_df.columns)
        self.assertIn("TIME_LIMIT_EXCEEDED", tv_df.columns)

        # Test Rating Wall detection with synthetic rating wall:
        # Create dataset with ratings:
        # 1200: 3 attempted, 3 solved (solve_rate = 1.0)
        # 1300: 4 attempted, 1 solved (solve_rate = 0.25) -> Rating Wall!
        synthetic_df = pd.DataFrame([
            # 1200 problems (3 attempted, all solved)
            {"problem_id": "1200A", "rating": 1200, "tags": ["math"], "verdict": "OK"},
            {"problem_id": "1200B", "rating": 1200, "tags": ["math"], "verdict": "OK"},
            {"problem_id": "1200C", "rating": 1200, "tags": ["math"], "verdict": "OK"},
            # 1300 problems (4 attempted, 1 solved)
            {"problem_id": "1300A", "rating": 1300, "tags": ["dp"], "verdict": "OK"},
            {"problem_id": "1300B", "rating": 1300, "tags": ["dp"], "verdict": "WRONG_ANSWER"},
            {"problem_id": "1300C", "rating": 1300, "tags": ["dp"], "verdict": "TIME_LIMIT_EXCEEDED"},
            {"problem_id": "1300D", "rating": 1300, "tags": ["dp"], "verdict": "WRONG_ANSWER"},
        ])

        wall_res = get_verdict_and_wall_stats(synthetic_df, user_rating=1500)
        self.assertEqual(wall_res["rating_wall"], 1300)
        self.assertIsNotNone(wall_res["rating_wall_stats"])
        self.assertEqual(wall_res["rating_wall_stats"]["attempted"], 4)
        self.assertEqual(wall_res["rating_wall_stats"]["solved"], 1)
        self.assertEqual(wall_res["rating_wall_stats"]["solve_rate"], 0.25)
        self.assertIn("1300", wall_res["primary_diagnosis"])


if __name__ == "__main__":
    unittest.main()
