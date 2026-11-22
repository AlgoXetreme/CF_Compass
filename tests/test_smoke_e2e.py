"""End-to-End Smoke Test for Codeforces Weak-Topic Recommender.

Tests the complete backend pipeline across:
(a) Active rated user with rich submissions (e.g. tourist).
(b) Low-rated or unrated user with 0 or few submissions.
(c) Non-existent or invalid handles with proper error handling.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import analyzer
import cf_api
import recommender


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-End smoke test verifying the entire analytical & recommendation flow."""

    def setUp(self):
        # Shared mock problemset
        self.mock_problemset = pd.DataFrame([
            {
                "problem_id": f"{cid}A",
                "contestId": cid,
                "index": "A",
                "name": f"Problem {cid}A",
                "rating": r,
                "tags": tags,
                "solvedCount": 5000,
                "url": f"https://codeforces.com/contest/{cid}/problem/A",
            }
            for cid, r, tags in [
                (1001, 800, ["greedy", "math"]),
                (1002, 900, ["binary search", "greedy"]),
                (1003, 1000, ["dp", "greedy"]),
                (1004, 1100, ["dp", "trees"]),
                (1005, 1200, ["graphs", "dfs and similar"]),
                (1006, 1300, ["shortest paths", "graphs"]),
                (1007, 1400, ["data structures", "dsu"]),
                (1008, 1500, ["binary search", "two pointers"]),
                (1009, 1600, ["dp", "bitmasks"]),
            ]
        ])

    @patch("cf_api.requests.get")
    def test_case_a_active_user_pipeline(self, mock_get):
        """(a) Test complete pipeline for an active rated user with rich submission history."""
        # 1. User info
        user_info_resp = MagicMock()
        user_info_resp.status_code = 200
        user_info_resp.json.return_value = {
            "status": "OK",
            "result": [
                {
                    "handle": "tourist",
                    "rating": 3979,
                    "maxRating": 4009,
                    "rank": "legendary grandmaster",
                    "avatar": "https://userpic.codeforces.org/avatar.jpg",
                }
            ],
        }

        # 2. Submissions
        subs_resp = MagicMock()
        subs_resp.status_code = 200
        subs_resp.json.return_value = {
            "status": "OK",
            "result": [
                # Solved dp problem
                {
                    "problem": {
                        "contestId": 1003,
                        "index": "A",
                        "name": "Problem 1003A",
                        "rating": 1000,
                        "tags": ["dp", "greedy"],
                    },
                    "verdict": "OK",
                    "creationTimeSeconds": 1700000000,
                },
                # Struggled/failed graphs problem
                {
                    "problem": {
                        "contestId": 1005,
                        "index": "A",
                        "name": "Problem 1005A",
                        "rating": 1200,
                        "tags": ["graphs", "dfs and similar"],
                    },
                    "verdict": "WRONG_ANSWER",
                    "creationTimeSeconds": 1700001000,
                },
                {
                    "problem": {
                        "contestId": 1005,
                        "index": "A",
                        "name": "Problem 1005A",
                        "rating": 1200,
                        "tags": ["graphs", "dfs and similar"],
                    },
                    "verdict": "TIME_LIMIT_EXCEEDED",
                    "creationTimeSeconds": 1700002000,
                },
            ],
        }

        # 3. Contests
        contests_resp = MagicMock()
        contests_resp.status_code = 200
        contests_resp.json.return_value = {
            "status": "OK",
            "result": [
                {
                    "id": 9999,
                    "name": "Codeforces Round 1000 (Div. 2)",
                    "phase": "BEFORE",
                    "startTimeSeconds": 1800000000,
                    "durationSeconds": 7200,
                }
            ],
        }

        mock_get.side_effect = [user_info_resp, subs_resp, contests_resp]

        # Step 1: User Info
        user_info = cf_api.get_user_info("tourist")
        self.assertEqual(user_info["handle"], "tourist")
        self.assertEqual(user_info["rating"], 3979)

        # Step 2: Submissions
        subs_df = cf_api.get_user_submissions("tourist")
        self.assertEqual(len(subs_df), 3)

        # Step 3: Tag Metrics & Prerequisite Boosting
        weakness_df = analyzer.compute_tag_metrics(subs_df, user_rating=user_info["rating"])
        self.assertFalse(weakness_df.empty)
        priority_df = analyzer.get_priority_weak_tags(weakness_df, top_k=3, return_details=True)
        self.assertFalse(priority_df.empty)

        # Step 4: Solved Problem IDs
        solved_ids = recommender.get_solved_problem_ids(subs_df)
        self.assertIn("1003A", solved_ids)
        self.assertNotIn("1005A", solved_ids)

        # Step 5: Progressive 5-Problem Upsolving Ladder
        ladder = recommender.generate_upsolving_ladder(
            problemset_df=self.mock_problemset,
            solved_ids=solved_ids,
            weak_tags_df=priority_df,
            user_rating=1200,
            target_stretch=300,
        )
        self.assertTrue(len(ladder) > 0)
        # Ensure no solved problems in ladder
        for item in ladder:
            self.assertNotIn(item["problem_id"], solved_ids)

        # Step 6: CP-31 and CSES Recommendations
        cp31_df = recommender.load_cp31_dataset()
        self.assertFalse(cp31_df.empty)
        cp31_recs = recommender.recommend_cp31(
            cp31_df=cp31_df,
            problemset_df=self.mock_problemset,
            solved_ids=solved_ids,
            weak_tags_df=priority_df,
            user_rating=1000,
        )
        self.assertIsInstance(cp31_recs, pd.DataFrame)

        cses_df = recommender.load_cses_dataset()
        self.assertFalse(cses_df.empty)
        cses_recs = recommender.recommend_cses(
            cses_df=cses_df,
            weak_tags_df=priority_df,
            completed_cses_ids=set(),
        )
        self.assertEqual(len(cses_recs), 5)

        # Step 7: Stuck Diagnostics
        stuck_df = analyzer.get_stuck_problems(subs_df)
        self.assertEqual(len(stuck_df), 1)
        self.assertEqual(stuck_df.iloc[0]["problem_id"], "1005A")
        self.assertEqual(stuck_df.iloc[0]["attempt_count"], 2)

        wall_stats = analyzer.get_verdict_and_wall_stats(subs_df, user_rating=1200)
        self.assertIn("tag_verdicts_df", wall_stats)
        self.assertIn("rating_stats_df", wall_stats)

        # Step 8: Upcoming Contests
        contests = cf_api.get_upcoming_contests()
        self.assertEqual(len(contests), 1)
        self.assertEqual(contests[0]["division"], "Div. 2")
        self.assertIn("calendar.google.com", contests[0]["gcal_url"])

    @patch("cf_api.requests.get")
    def test_case_b_unrated_new_user_pipeline(self, mock_get):
        """(b) Test complete pipeline for an unrated user with 0 submissions."""
        # Unrated user info
        user_info_resp = MagicMock()
        user_info_resp.status_code = 200
        user_info_resp.json.return_value = {
            "status": "OK",
            "result": [
                {
                    "handle": "brand_new_coder",
                    # No rating, no maxRating, no rank
                    "avatar": "//userpic.codeforces.org/no-avatar.jpg",
                }
            ],
        }

        # 0 submissions
        subs_resp = MagicMock()
        subs_resp.status_code = 200
        subs_resp.json.return_value = {
            "status": "OK",
            "result": [],
        }

        mock_get.side_effect = [user_info_resp, subs_resp]

        user_info = cf_api.get_user_info("brand_new_coder")
        self.assertEqual(user_info["rating"], 800)  # Defaulted to 800
        self.assertEqual(user_info["rank"], "unrated")
        self.assertTrue(user_info["avatar"].startswith("https://"))

        subs_df = cf_api.get_user_submissions("brand_new_coder")
        self.assertTrue(subs_df.empty)

        # Ensure analytics handle empty submissions without crashing
        weakness_df = analyzer.compute_tag_metrics(subs_df, user_rating=800)
        self.assertTrue(weakness_df.empty)

        priority_df = analyzer.get_priority_weak_tags(weakness_df, return_details=True)
        self.assertTrue(priority_df.empty)

        stuck_df = analyzer.get_stuck_problems(subs_df)
        self.assertTrue(stuck_df.empty)

        wall_stats = analyzer.get_verdict_and_wall_stats(subs_df, user_rating=800)
        self.assertIsNone(wall_stats["rating_wall"])
        self.assertEqual(wall_stats["primary_diagnosis"], "No submission history available to diagnose.")

        # Recommender generates starter ladder even with 0 submissions
        ladder = recommender.generate_upsolving_ladder(
            problemset_df=self.mock_problemset,
            solved_ids=set(),
            weak_tags_df=weakness_df,
            user_rating=800,
        )
        self.assertTrue(len(ladder) > 0)

    @patch("cf_api.requests.get")
    def test_case_c_invalid_and_nonexistent_handles(self, mock_get):
        """(c) Test non-existent and empty handles raise proper exceptions."""
        # 1. Empty handle raises ValueError immediately without API call
        with self.assertRaises(ValueError):
            cf_api.get_user_info("")
        with self.assertRaises(ValueError):
            cf_api.get_user_info("   ")
        with self.assertRaises(ValueError):
            cf_api.get_user_submissions("")

        # 2. Non-existent handle returns 400 or status != OK with "not found" comment
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "status": "FAILED",
            "comment": "handles: User with such handle not found",
        }
        mock_get.return_value = mock_response

        with self.assertRaises(cf_api.UserNotFoundError):
            cf_api.get_user_info("definitely_non_existent_handle_99999")

        with self.assertRaises(cf_api.UserNotFoundError):
            cf_api.get_user_submissions("definitely_non_existent_handle_99999")


if __name__ == "__main__":
    unittest.main()
