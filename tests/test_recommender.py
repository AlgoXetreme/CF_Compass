"""Unit tests for recommender module including TF-IDF and Upsolving Ladder."""

import os
import unittest
import pandas as pd

from recommender import (
    CF_TO_CSES_MAP,
    format_tag_for_tfidf,
    format_tags_string,
    generate_upsolving_ladder,
    get_cp31_topic_ladder,
    get_cp31_unique_tags,
    get_solved_problem_ids,
    load_cses_progress,
    recommend_cp31,
    recommend_cses,
    save_cses_progress,
)


class TestRecommender(unittest.TestCase):
    """Test suite for TF-IDF cosine similarity upsolving ladder."""

    def setUp(self):
        self.solved_ids = {"1000A", "1000B"}

        # Weak tags dataframe: Top 4 weak tags
        self.weak_tags_df = pd.DataFrame([
            {"tag": "graphs", "weakness_score": 85.0, "boosted_score": 95.0},
            {"tag": "dp", "weakness_score": 70.0, "boosted_score": 75.0},
            {"tag": "binary search", "weakness_score": 50.0, "boosted_score": 50.0},
            {"tag": "two pointers", "weakness_score": 30.0, "boosted_score": 30.0},
        ])

        # Candidate problemset covering multiple ratings and tags
        self.problemset_df = pd.DataFrame([
            # Solved problems - should be excluded
            {
                "problem_id": "1000A",
                "name": "Solved Graphs",
                "rating": 1500,
                "tags": ["graphs"],
                "solvedCount": 10000,
                "url": "https://codeforces.com/contest/1000/problem/A",
            },
            {
                "problem_id": "1000B",
                "name": "Solved DP",
                "rating": 1600,
                "tags": ["dp"],
                "solvedCount": 8000,
                "url": "https://codeforces.com/contest/1000/problem/B",
            },
            # Missing rating problem - should be excluded
            {
                "problem_id": "9999X",
                "name": "Unrated Problem",
                "rating": None,
                "tags": ["graphs"],
                "solvedCount": 50,
                "url": "https://codeforces.com/contest/9999/problem/X",
            },
            # Rating 1500 candidates
            {
                "problem_id": "1500G",
                "name": "Graph Basics",
                "rating": 1500,
                "tags": ["graphs", "dfs and similar"],
                "solvedCount": 12000,
                "url": "https://codeforces.com/contest/1500/problem/G",
            },
            {
                "problem_id": "1500D",
                "name": "DP Basics",
                "rating": 1500,
                "tags": ["dp"],
                "solvedCount": 9000,
                "url": "https://codeforces.com/contest/1500/problem/D",
            },
            # Rating 1600 candidates (for buckets 2 and 3)
            {
                "problem_id": "1600G",
                "name": "Shortest Paths Journey",
                "rating": 1600,
                "tags": ["graphs", "shortest paths"],
                "solvedCount": 7500,
                "url": "https://codeforces.com/contest/1600/problem/G",
            },
            {
                "problem_id": "1600D",
                "name": "DP Grid Optimization",
                "rating": 1600,
                "tags": ["dp"],
                "solvedCount": 8500,
                "url": "https://codeforces.com/contest/1600/problem/D",
            },
            {
                "problem_id": "1600B",
                "name": "Binary Search Array",
                "rating": 1600,
                "tags": ["binary search", "two pointers"],
                "solvedCount": 11000,
                "url": "https://codeforces.com/contest/1600/problem/B",
            },
            # Rating 1700 candidates (for bucket 4)
            {
                "problem_id": "1700G",
                "name": "Graph Cycles",
                "rating": 1700,
                "tags": ["graphs"],
                "solvedCount": 5000,
                "url": "https://codeforces.com/contest/1700/problem/G",
            },
            {
                "problem_id": "1700B",
                "name": "Binary Search Hard",
                "rating": 1700,
                "tags": ["binary search"],
                "solvedCount": 6200,
                "url": "https://codeforces.com/contest/1700/problem/B",
            },
            # Rating 1800 candidates (for bucket 5)
            {
                "problem_id": "1800D",
                "name": "DP Tree Counting",
                "rating": 1800,
                "tags": ["dp", "trees"],
                "solvedCount": 4200,
                "url": "https://codeforces.com/contest/1800/problem/D",
            },
            {
                "problem_id": "1800G",
                "name": "Dijkstra Variations",
                "rating": 1800,
                "tags": ["graphs", "shortest paths"],
                "solvedCount": 3800,
                "url": "https://codeforces.com/contest/1800/problem/G",
            },
        ])

    def test_format_tag_for_tfidf(self):
        """Test tag formatting with spaces replaced by underscores."""
        self.assertEqual(format_tag_for_tfidf("binary search"), "binary_search")
        self.assertEqual(format_tag_for_tfidf("shortest paths"), "shortest_paths")
        self.assertEqual(format_tag_for_tfidf("dfs and similar"), "dfs_and_similar")
        self.assertEqual(format_tag_for_tfidf("dp"), "dp")
        self.assertEqual(format_tag_for_tfidf(""), "")

    def test_format_tags_string(self):
        """Test converting tag list to space-joined underscored tokens."""
        tags = ["binary search", "two pointers", "dp"]
        formatted = format_tags_string(tags)
        self.assertEqual(formatted, "binary_search two_pointers dp")

    def test_generate_upsolving_ladder_structure_and_variety(self):
        """Test generation of 5-problem progressive ladder across [R, R+100, R+100, R+200, R+300]."""
        user_rating = 1500
        ladder = generate_upsolving_ladder(
            problemset_df=self.problemset_df,
            solved_ids=self.solved_ids,
            weak_tags_df=self.weak_tags_df,
            user_rating=user_rating,
        )

        self.assertIsInstance(ladder, list)
        self.assertEqual(len(ladder), 5)

        # Check ladder steps and target ratings
        expected_steps = [1, 2, 3, 4, 5]
        expected_targets = [1500, 1600, 1600, 1700, 1800]
        actual_steps = [item["ladder_step"] for item in ladder]
        actual_targets = [item["target_rating"] for item in ladder]
        self.assertEqual(actual_steps, expected_steps)
        self.assertEqual(actual_targets, expected_targets)

        # Ensure solved problems are never included
        recommended_pids = [item["problem_id"] for item in ladder]
        for pid in recommended_pids:
            self.assertNotIn(pid, self.solved_ids)

        # Ensure all recommended problem IDs are unique
        self.assertEqual(len(recommended_pids), len(set(recommended_pids)))

        # Ensure required keys exist
        required_keys = {
            "ladder_step",
            "target_rating",
            "problem_id",
            "name",
            "rating",
            "tags",
            "solvedCount",
            "url",
            "similarity_score",
            "combined_score",
            "primary_tag",
        }
        for item in ladder:
            self.assertTrue(required_keys.issubset(item.keys()))

        # Ensure topic variety: all 5 problems must NOT share the same primary tag
        primary_tags = [item["primary_tag"] for item in ladder]
        self.assertGreater(len(set(primary_tags)), 1)

    def test_generate_upsolving_ladder_unrated_minimum_800(self):
        """Test that unrated user (rating=0 or 500) defaults base rating to minimum 800."""
        ladder = generate_upsolving_ladder(
            problemset_df=self.problemset_df,
            solved_ids=set(),
            weak_tags_df=self.weak_tags_df,
            user_rating=400,  # Below minimum 800
        )
        self.assertIsInstance(ladder, list)
        if ladder:
            # First bucket should be minimum 800
            self.assertEqual(ladder[0]["target_rating"], 800)

    def test_generate_upsolving_ladder_empty_inputs(self):
        """Test graceful handling of empty problemset."""
        empty_ladder = generate_upsolving_ladder(
            problemset_df=pd.DataFrame(),
            solved_ids=set(),
            weak_tags_df=self.weak_tags_df,
            user_rating=1500,
        )
        self.assertEqual(empty_ladder, [])

    def test_recommend_cp31(self):
        """Test recommending CP-31 problems matching weak tags and within rating window."""
        cp31_sample = pd.DataFrame([
            {"problem_id": "1500G", "rating": 1500, "sheet_tier": 1500},
            {"problem_id": "1500D", "rating": 1500, "sheet_tier": 1500},
            {"problem_id": "1600G", "rating": 1600, "sheet_tier": 1600},
            {"problem_id": "1600D", "rating": 1600, "sheet_tier": 1600},
            {"problem_id": "1700G", "rating": 1700, "sheet_tier": 1700},
            {"problem_id": "1000A", "rating": 1000, "sheet_tier": 1000},  # Solved problem
            {"problem_id": "2200X", "rating": 2200, "sheet_tier": 2200},  # Outside rating window
        ])

        recs = recommend_cp31(
            cp31_df=cp31_sample,
            problemset_df=self.problemset_df,
            solved_ids={"1000A"},
            weak_tags_df=self.weak_tags_df,
            user_rating=1500,  # window: 1500 to 1800
        )

        self.assertIsInstance(recs, pd.DataFrame)
        self.assertLessEqual(len(recs), 5)
        self.assertGreater(len(recs), 0)

        rec_pids = recs["problem_id"].tolist()
        self.assertNotIn("1000A", rec_pids)  # Solved problem excluded
        self.assertNotIn("2200X", rec_pids)  # Rating > 1800 excluded

        # Check required columns
        for col in ["problem_id", "name", "rating", "sheet_tier", "tags", "url"]:
            self.assertIn(col, recs.columns)

    def test_recommend_cses(self):
        """Test recommending CSES problems mapped from weak tags ordered Easy to Hard."""
        cses_sample = pd.DataFrame([
            {"id": 1633, "name": "Dice Combinations", "category": "Dynamic Programming", "difficulty": "Easy", "url": "https://cses.fi/task/1633"},
            {"id": 1636, "name": "Coin Combinations II", "category": "Dynamic Programming", "difficulty": "Medium", "url": "https://cses.fi/task/1636"},
            {"id": 1653, "name": "Elevator Rides", "category": "Dynamic Programming", "difficulty": "Hard", "url": "https://cses.fi/task/1653"},
            {"id": 1192, "name": "Counting Rooms", "category": "Graph Algorithms", "difficulty": "Easy", "url": "https://cses.fi/task/1192"},
            {"id": 1671, "name": "Shortest Routes I", "category": "Graph Algorithms", "difficulty": "Medium", "url": "https://cses.fi/task/1671"},
            {"id": 9999, "name": "Completed Problem", "category": "Dynamic Programming", "difficulty": "Easy", "url": "https://cses.fi/task/9999"},
        ])

        # Weak tags: Top 1 is "graphs", Top 2 is "dp" -> maps to Graph Algorithms and Dynamic Programming
        recs = recommend_cses(
            cses_df=cses_sample,
            weak_tags_df=self.weak_tags_df,
            completed_cses_ids={9999},  # Completed problem
        )

        self.assertIsInstance(recs, pd.DataFrame)
        self.assertLessEqual(len(recs), 5)
        rec_ids = recs["id"].tolist()
        self.assertNotIn(9999, rec_ids)

        # Check ordering: Easy before Medium, Medium before Hard
        difficulties = recs["difficulty"].tolist()
        diff_order = {"Easy": 1, "Medium": 2, "Hard": 3}
        numerical_order = [diff_order.get(d, 2) for d in difficulties]
        self.assertEqual(numerical_order, sorted(numerical_order))

    def test_cses_progress_save_and_load(self):
        """Test persisting and loading completed CSES progress."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sample_ids = {1633, 1192, 1666}
            save_cses_progress(sample_ids, path=tmp_path)
            loaded = load_cses_progress(path=tmp_path)
            self.assertEqual(loaded, sample_ids)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_get_cp31_unique_tags(self):
        """Test extracting unique algorithmic tags from CP-31 dataset."""
        sample_cp31 = pd.DataFrame([
            {"problem_id": "1", "name": "P1", "rating": 800, "sheet_tier": 800, "tags": ["math", "greedy"]},
            {"problem_id": "2", "name": "P2", "rating": 900, "sheet_tier": 900, "tags": ["dp", "math"]},
        ])
        tags = get_cp31_unique_tags(sample_cp31)
        self.assertIsInstance(tags, list)
        self.assertIn("math", tags)
        self.assertIn("greedy", tags)
        self.assertIn("dp", tags)
        self.assertEqual(tags, sorted(tags))

    def test_get_cp31_topic_ladder(self):
        """Test topic ladder generation, rating-ascending sort, and status assignment."""
        sample_cp31 = pd.DataFrame([
            {"problem_id": "1100A", "name": "Prob 1100", "rating": 1100, "sheet_tier": 1100, "tags": ["binary search"]},
            {"problem_id": "1000A", "name": "Prob 1000", "rating": 1000, "sheet_tier": 1000, "tags": ["binary search", "greedy"]},
            {"problem_id": "1200A", "name": "Prob 1200", "rating": 1200, "sheet_tier": 1200, "tags": ["binary search"]},
            {"problem_id": "1000B", "name": "Prob 1000 B", "rating": 1000, "sheet_tier": 1000, "tags": ["binary search"]},
            {"problem_id": "1300A", "name": "Prob 1300", "rating": 1300, "sheet_tier": 1300, "tags": ["dp"]},  # Different tag
            {"problem_id": "1700A", "name": "Prob 1700", "rating": 1700, "sheet_tier": 1700, "tags": ["binary search"]},  # Rating outside 900-1500
        ])

        sample_problemset = pd.DataFrame([
            {"problem_id": "1100A", "solvedCount": 12000, "url": "https://codeforces.com/contest/1100/problem/A"},
            {"problem_id": "1000A", "solvedCount": 5000, "url": "https://codeforces.com/contest/1000/problem/A"},
            {"problem_id": "1200A", "solvedCount": 8000, "url": "https://codeforces.com/contest/1200/problem/A"},
            {"problem_id": "1000B", "solvedCount": 15000, "url": "https://codeforces.com/contest/1000/problem/B"},
        ])

        # Submissions:
        # 1000A -> Solved (has OK)
        # 1100A -> Stuck (attempted with WA, no OK)
        # 1000B -> Todo (never attempted)
        # 1200A -> Todo (never attempted)
        sample_subs = pd.DataFrame([
            {"problem_id": "1000A", "verdict": "WRONG_ANSWER"},
            {"problem_id": "1000A", "verdict": "OK"},
            {"problem_id": "1100A", "verdict": "WRONG_ANSWER"},
            {"problem_id": "1100A", "verdict": "TIME_LIMIT_EXCEEDED"},
        ])

        ladder_df, stats = get_cp31_topic_ladder(
            cp31_df=sample_cp31,
            problemset_df=sample_problemset,
            submissions_df=sample_subs,
            selected_tag="binary search",
            min_rating=900,
            max_rating=1500,
        )

        self.assertIsInstance(ladder_df, pd.DataFrame)
        self.assertIsInstance(stats, dict)

        # 4 problems should match (1000A, 1000B, 1100A, 1200A)
        self.assertEqual(len(ladder_df), 4)
        self.assertEqual(stats["total_count"], 4)
        self.assertEqual(stats["solved_count"], 1)  # 1000A
        self.assertEqual(stats["stuck_count"], 1)   # 1100A
        self.assertEqual(stats["todo_count"], 2)    # 1000B, 1200A
        self.assertEqual(stats["solve_rate"], 0.25)

        # Check sorting strictly by rating ascending (1000 -> 1000 -> 1100 -> 1200)
        ratings = ladder_df["rating"].tolist()
        self.assertEqual(ratings, sorted(ratings))

        # Check tie breaker: at rating 1000, 1000B has solvedCount 15000, 1000A has 5000 -> 1000B comes first
        at_1000 = ladder_df[ladder_df["rating"] == 1000]
        self.assertEqual(at_1000.iloc[0]["problem_id"], "1000B")
        self.assertEqual(at_1000.iloc[1]["problem_id"], "1000A")

        # Check statuses
        row_1000a = ladder_df[ladder_df["problem_id"] == "1000A"].iloc[0]
        self.assertEqual(row_1000a["status"], "Solved")

        row_1100a = ladder_df[ladder_df["problem_id"] == "1100A"].iloc[0]
        self.assertEqual(row_1100a["status"], "Stuck")

        row_1000b = ladder_df[ladder_df["problem_id"] == "1000B"].iloc[0]
        self.assertEqual(row_1000b["status"], "Todo")


if __name__ == "__main__":
    unittest.main()
