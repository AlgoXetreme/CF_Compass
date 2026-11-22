"""Unit tests for cf_api module using unittest and unittest.mock."""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from cf_api import (
    DEFAULT_UNRATED_RATING,
    CodeforcesAPIError,
    NetworkError,
    RateLimitError,
    UserNotFoundError,
    format_countdown,
    format_duration,
    get_problemset,
    get_upcoming_contests,
    get_user_info,
    get_user_submissions,
    is_contest_recommended_for_rating,
    parse_contest_division,
)


class TestGetUserInfo(unittest.TestCase):
    """Tests for get_user_info function."""

    @patch("cf_api.requests.get")
    def test_get_user_info_rated(self, mock_get):
        """Test getting information for an active rated user."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        mock_get.return_value = mock_response

        info = get_user_info("tourist")
        self.assertEqual(info["handle"], "tourist")
        self.assertEqual(info["rating"], 3979)
        self.assertEqual(info["maxRating"], 4009)
        self.assertEqual(info["rank"], "legendary grandmaster")
        self.assertEqual(info["avatar"], "https://userpic.codeforces.org/avatar.jpg")

    @patch("cf_api.requests.get")
    def test_get_user_info_unrated_defaults_to_800(self, mock_get):
        """Test that unrated user gets default rating of 800 and proper defaults."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "result": [
                {
                    "handle": "newbie_coder",
                    # 'rating' is missing
                    # 'maxRating' is missing
                    # 'rank' is missing
                    "avatar": "//userpic.codeforces.org/no-avatar.jpg",
                }
            ],
        }
        mock_get.return_value = mock_response

        info = get_user_info("newbie_coder")
        self.assertEqual(info["handle"], "newbie_coder")
        self.assertEqual(info["rating"], DEFAULT_UNRATED_RATING)
        self.assertEqual(info["maxRating"], DEFAULT_UNRATED_RATING)
        self.assertEqual(info["rank"], "unrated")
        # Ensure protocol-relative avatar normalized
        self.assertEqual(info["avatar"], "https://userpic.codeforces.org/no-avatar.jpg")

    @patch("cf_api.requests.get")
    def test_get_user_info_not_found(self, mock_get):
        """Test when Codeforces reports user not found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "FAILED",
            "comment": "handles: User not found",
        }
        mock_get.return_value = mock_response

        with self.assertRaises(UserNotFoundError):
            get_user_info("non_existent_handle_xyz")

    def test_get_user_info_empty_handle(self):
        """Test that empty or whitespace handle raises ValueError."""
        with self.assertRaises(ValueError):
            get_user_info("")
        with self.assertRaises(ValueError):
            get_user_info("   ")

    @patch("cf_api.requests.get")
    def test_get_user_info_timeout(self, mock_get):
        """Test network timeout raises NetworkError."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        with self.assertRaises(NetworkError):
            get_user_info("tourist")


class TestGetUserSubmissions(unittest.TestCase):
    """Tests for get_user_submissions function."""

    @patch("cf_api.requests.get")
    def test_get_user_submissions_success(self, mock_get):
        """Test fetching and parsing user submissions into structured DataFrame."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "result": [
                {
                    "id": 101,
                    "contestId": 1915,
                    "creationTimeSeconds": 1705330456,
                    "problem": {
                        "contestId": 1915,
                        "index": "G",
                        "name": "Bicycles",
                        "rating": 1800,
                        "tags": ["dijkstra", "graphs", "shortest paths"],
                    },
                    "verdict": "OK",
                },
                {
                    "id": 102,
                    "contestId": 1800,
                    "creationTimeSeconds": 1705000000,
                    "problem": {
                        "contestId": 1800,
                        "index": "A",
                        "name": "Is It a Cat?",
                        "rating": 800,
                        "tags": ["strings"],
                    },
                    "verdict": "WRONG_ANSWER",
                },
            ],
        }
        mock_get.return_value = mock_response

        df = get_user_submissions("sample_user")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

        # Check required columns
        expected_columns = [
            "problem_id",
            "name",
            "rating",
            "tags",
            "verdict",
            "creationTimeSeconds",
        ]
        self.assertListEqual(list(df.columns), expected_columns)

        # Verify problem_id format (e.g. 1915G)
        self.assertEqual(df.iloc[0]["problem_id"], "1915G")
        self.assertEqual(df.iloc[0]["name"], "Bicycles")
        self.assertEqual(df.iloc[0]["rating"], 1800)
        self.assertEqual(df.iloc[0]["tags"], ["dijkstra", "graphs", "shortest paths"])
        self.assertEqual(df.iloc[0]["verdict"], "OK")
        self.assertEqual(df.iloc[0]["creationTimeSeconds"], 1705330456)

        self.assertEqual(df.iloc[1]["problem_id"], "1800A")
        self.assertEqual(df.iloc[1]["verdict"], "WRONG_ANSWER")

    @patch("cf_api.requests.get")
    def test_get_user_submissions_empty(self, mock_get):
        """Test user with 0 submissions returns empty DataFrame with correct columns."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "result": [],
        }
        mock_get.return_value = mock_response

        df = get_user_submissions("inactive_user")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)
        expected_columns = [
            "problem_id",
            "name",
            "rating",
            "tags",
            "verdict",
            "creationTimeSeconds",
        ]
        self.assertListEqual(list(df.columns), expected_columns)


class TestGetProblemset(unittest.TestCase):
    """Tests for get_problemset function and caching."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.temp_dir.name, "cache", "test_problems.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("cf_api.requests.get")
    def test_fetch_fresh_problemset_merges_and_caches(self, mock_get):
        """Test problemset fetching, statistics merge, URL creation, and rated problem filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "result": {
                "problems": [
                    {
                        "contestId": 1915,
                        "index": "G",
                        "name": "Bicycles",
                        "rating": 1800,
                        "tags": ["dijkstra", "graphs"],
                    },
                    {
                        "contestId": 1915,
                        "index": "A",
                        "name": "Odd One Out",
                        "rating": 800,
                        "tags": ["bitmasks", "implementation"],
                    },
                    {
                        "contestId": 9999,
                        "index": "X",
                        "name": "Unrated Problem",
                        # rating is missing / unrated
                        "tags": [],
                    },
                ],
                "problemStatistics": [
                    {
                        "contestId": 1915,
                        "index": "G",
                        "solvedCount": 4210,
                    },
                    {
                        "contestId": 1915,
                        "index": "A",
                        "solvedCount": 35200,
                    },
                    {
                        "contestId": 9999,
                        "index": "X",
                        "solvedCount": 12,
                    },
                ],
            },
        }
        mock_get.return_value = mock_response

        df = get_problemset(cache_path=self.cache_path)

        # Ensure API was called
        mock_get.assert_called_once()

        # Cache file should now exist on disk
        self.assertTrue(os.path.exists(self.cache_path))

        # Check DataFrame: unrated problem (9999X) should be filtered out
        self.assertEqual(len(df), 2)
        expected_columns = [
            "problem_id",
            "contestId",
            "index",
            "name",
            "rating",
            "tags",
            "solvedCount",
            "url",
        ]
        self.assertListEqual(list(df.columns), expected_columns)

        # Check problem 1915G details
        row_g = df[df["problem_id"] == "1915G"].iloc[0]
        self.assertEqual(row_g["name"], "Bicycles")
        self.assertEqual(row_g["rating"], 1800)
        self.assertEqual(row_g["solvedCount"], 4210)
        self.assertEqual(row_g["url"], "https://codeforces.com/contest/1915/problem/G")

        # Check problem 1915A details
        row_a = df[df["problem_id"] == "1915A"].iloc[0]
        self.assertEqual(row_a["rating"], 800)
        self.assertEqual(row_a["solvedCount"], 35200)
        self.assertEqual(row_a["url"], "https://codeforces.com/contest/1915/problem/A")

    @patch("cf_api.requests.get")
    def test_cache_hits_within_24_hours(self, mock_get):
        """Test that cache is read from disk and API is NOT called when cache is fresh."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        sample_cached_problems = [
            {
                "problem_id": "1800B",
                "contestId": 1800,
                "index": "B",
                "name": "Count the Number of Pairs",
                "rating": 1000,
                "tags": ["greedy", "strings"],
                "solvedCount": 18500,
                "url": "https://codeforces.com/contest/1800/problem/B",
            }
        ]
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(sample_cached_problems, f)

        # Call get_problemset with fresh cache
        df = get_problemset(cache_path=self.cache_path)

        # API should NOT have been called
        mock_get.assert_not_called()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["problem_id"], "1800B")
        self.assertEqual(df.iloc[0]["solvedCount"], 18500)

    @patch("cf_api.requests.get")
    def test_expired_cache_triggers_refresh(self, mock_get):
        """Test that cache older than 24 hours triggers API fetch."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        # Write old cache
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        # Set mtime to 25 hours ago
        twenty_five_hours_ago = time.time() - (25 * 3600)
        os.utime(self.cache_path, (twenty_five_hours_ago, twenty_five_hours_ago))

        # Setup mock for API refresh
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "result": {
                "problems": [
                    {
                        "contestId": 1234,
                        "index": "A",
                        "name": "Fresh Problem",
                        "rating": 1200,
                        "tags": ["math"],
                    }
                ],
                "problemStatistics": [
                    {
                        "contestId": 1234,
                        "index": "A",
                        "solvedCount": 999,
                    }
                ],
            },
        }
        mock_get.return_value = mock_response

        df = get_problemset(cache_path=self.cache_path)
        mock_get.assert_called_once()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["problem_id"], "1234A")


class TestUpcomingContests(unittest.TestCase):
    """Tests for get_upcoming_contests and contest helper utilities."""

    def test_parse_contest_division(self):
        """Test accurate parsing of division from contest names."""
        self.assertEqual(
            parse_contest_division("Codeforces Round 999 (Div. 1 + Div. 2)"),
            "Div. 1 + Div. 2",
        )
        self.assertEqual(
            parse_contest_division("Codeforces Round 1000 (Div. 2)"),
            "Div. 2",
        )
        self.assertEqual(
            parse_contest_division("Codeforces Round 1001 (Div. 3)"),
            "Div. 3",
        )
        self.assertEqual(
            parse_contest_division("Codeforces Round 1002 (Div. 4)"),
            "Div. 4",
        )
        self.assertEqual(
            parse_contest_division("Codeforces Round 1003 (Div. 1)"),
            "Div. 1",
        )
        self.assertEqual(
            parse_contest_division("Educational Codeforces Round 172 (Rated for Div. 2)"),
            "Educational",
        )
        self.assertEqual(
            parse_contest_division("Codeforces Global Round 28"),
            "Div. 1 + Div. 2",
        )
        self.assertEqual(
            parse_contest_division("Kotlin Heroes: Episode 11"),
            "Other",
        )

    def test_format_countdown(self):
        """Test formatting countdown seconds into human-readable strings."""
        # 1 day, 4 hours, 30 minutes, 15 seconds
        sec_1d4h30m = 86400 + 4 * 3600 + 30 * 60 + 15
        self.assertEqual(format_countdown(sec_1d4h30m), "1d 4h 30m")

        # 3 hours, 20 minutes
        sec_3h20m = 3 * 3600 + 20 * 60
        self.assertEqual(format_countdown(sec_3h20m), "3h 20m")

        # 45 minutes
        sec_45m = 45 * 60
        self.assertEqual(format_countdown(sec_45m), "45m")

        # Past or 0
        self.assertEqual(format_countdown(0), "Starting now")
        self.assertEqual(format_countdown(-50), "Starting now")

    def test_format_duration(self):
        """Test duration formatting."""
        self.assertEqual(format_duration(7200), "2 hours")
        self.assertEqual(format_duration(3600), "1 hour")
        self.assertEqual(format_duration(8100), "2h 15m")
        self.assertEqual(format_duration(1800), "30m")

    def test_is_contest_recommended_for_rating(self):
        """Test rating-based contest recommendation rules."""
        # Under 1400: Div. 4, Div. 3, Div. 2
        self.assertTrue(is_contest_recommended_for_rating("Div. 4", 1200))
        self.assertTrue(is_contest_recommended_for_rating("Div. 3", 1350))
        self.assertTrue(is_contest_recommended_for_rating("Div. 2", 800))
        self.assertFalse(is_contest_recommended_for_rating("Div. 1", 1200))

        # 1400 to 1899: Div. 3, Div. 2, Educational, Div. 1 + Div. 2
        self.assertTrue(is_contest_recommended_for_rating("Div. 2", 1500))
        self.assertTrue(is_contest_recommended_for_rating("Educational", 1750))
        self.assertTrue(is_contest_recommended_for_rating("Div. 1 + Div. 2", 1600))
        self.assertFalse(is_contest_recommended_for_rating("Div. 4", 1600))

        # 1900+: Div. 1, Div. 2, Div. 1 + Div. 2, Educational
        self.assertTrue(is_contest_recommended_for_rating("Div. 1", 2100))
        self.assertTrue(is_contest_recommended_for_rating("Div. 1 + Div. 2", 2400))
        self.assertFalse(is_contest_recommended_for_rating("Div. 4", 2000))
        self.assertFalse(is_contest_recommended_for_rating("Div. 3", 2200))

    @patch("cf_api.requests.get")
    def test_get_upcoming_contests_filtering_and_sorting(self, mock_get):
        """Test fetching, filtering to phase=='BEFORE', sorting, and metadata enrichment."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulated fixed time: 2026-11-20 00:00:00 UTC (timestamp: 1795132800)
        fixed_now = 1795132800.0

        mock_response.json.return_value = {
            "status": "OK",
            "result": [
                {
                    "id": 2001,
                    "name": "Codeforces Round 1001 (Div. 3)",
                    "phase": "BEFORE",
                    "startTimeSeconds": int(fixed_now + 100000),  # Later
                    "durationSeconds": 7200,
                },
                {
                    "id": 1999,
                    "name": "Past Contest Finished",
                    "phase": "FINISHED",
                    "startTimeSeconds": int(fixed_now - 50000),
                    "durationSeconds": 7200,
                },
                {
                    "id": 2000,
                    "name": "Educational Codeforces Round 172 (Rated for Div. 2)",
                    "phase": "BEFORE",
                    "startTimeSeconds": int(fixed_now + 50000),  # Sooner
                    "durationSeconds": 7200,
                },
                {
                    "id": 1998,
                    "name": "Live Contest In Progress",
                    "phase": "CODING",
                    "startTimeSeconds": int(fixed_now - 1000),
                    "durationSeconds": 7200,
                },
            ],
        }
        mock_get.return_value = mock_response

        contests = get_upcoming_contests(current_time=fixed_now)

        # Should only return the 2 contests with phase == "BEFORE"
        self.assertEqual(len(contests), 2)

        # Should be sorted soonest first: contest 2000 before contest 2001
        self.assertEqual(contests[0]["id"], 2000)
        self.assertEqual(contests[0]["division"], "Educational")
        self.assertEqual(contests[0]["url"], "https://codeforces.com/contests/2000")
        self.assertIn("calendar.google.com", contests[0]["gcal_url"])
        self.assertIn("IST", contests[0]["start_time_ist"])
        self.assertTrue(contests[0]["countdown"])

        self.assertEqual(contests[1]["id"], 2001)
        self.assertEqual(contests[1]["division"], "Div. 3")


if __name__ == "__main__":
    unittest.main()
