"""Codeforces API Wrapper Module for Weak-Topic Recommender.

Provides clean, modular, and robust functions for:
- Fetching user profile information (with unrated defaults)
- Fetching user submission history as a structured Pandas DataFrame
- Fetching and caching the problemset with merged statistics and direct URLs
"""

import json
import logging
import os
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from zoneinfo import ZoneInfo

    IST_ZONE = ZoneInfo("Asia/Kolkata")
except Exception:
    IST_ZONE = timezone(timedelta(hours=5, minutes=30))

import pandas as pd
import requests

# Set up module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# Constants
CF_API_BASE_URL = "https://codeforces.com/api"
DEFAULT_TIMEOUT = 10  # Timeout in seconds
CACHE_EXPIRATION_SECONDS = 24 * 3600  # 24 hours
DEFAULT_UNRATED_RATING = 800


class CodeforcesAPIError(Exception):
    """Base exception for Codeforces API related errors."""
    pass


class UserNotFoundError(CodeforcesAPIError):
    """Raised when the specified Codeforces user handle does not exist."""
    pass


class RateLimitError(CodeforcesAPIError):
    """Raised when Codeforces API rate limits requests."""
    pass


class NetworkError(CodeforcesAPIError):
    """Raised when a network timeout or connection failure occurs."""
    pass


def _make_api_request(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Any:
    """Helper function to execute GET requests against the Codeforces API with error handling.

    Args:
        endpoint: API method endpoint (e.g., 'user.info', 'user.status', 'problemset.problems').
        params: Query parameters dictionary.
        timeout: Request timeout in seconds.
        session: Optional requests.Session instance for connection reuse.

    Returns:
        The 'result' field from the Codeforces JSON response.

    Raises:
        UserNotFoundError: If Codeforces reports handle not found.
        RateLimitError: If HTTP 429 / 503 or rate limit message returned.
        NetworkError: On timeout or connection failure.
        CodeforcesAPIError: On other API errors.
    """
    url = f"{CF_API_BASE_URL}/{endpoint}"
    requester = session or requests

    try:
        response = requester.get(url, params=params, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        logger.error(f"Timeout connecting to Codeforces API ({url}): {exc}")
        raise NetworkError(f"Request to Codeforces API timed out after {timeout}s.") from exc
    except requests.exceptions.RequestException as exc:
        logger.error(f"Network error querying Codeforces API ({url}): {exc}")
        raise NetworkError(f"Network error while connecting to Codeforces API: {exc}") from exc

    # Check for rate limiting / server errors via HTTP status
    if response.status_code == 429:
        raise RateLimitError("Codeforces API rate limit exceeded. Please wait a moment.")
    if response.status_code in (502, 503, 504):
        raise NetworkError(f"Codeforces API is currently unavailable (HTTP {response.status_code}).")

    try:
        data = response.json()
    except ValueError as exc:
        raise CodeforcesAPIError(
            f"Failed to parse JSON response from Codeforces API (HTTP {response.status_code})."
        ) from exc

    status = data.get("status")
    if status != "OK":
        comment = data.get("comment", "Unknown Codeforces API error")
        comment_lower = comment.lower()

        if "not found" in comment_lower or "user with such handle" in comment_lower:
            raise UserNotFoundError(f"User not found: {comment}")
        if "limit exceeded" in comment_lower or "call limit" in comment_lower:
            raise RateLimitError(f"Codeforces call limit exceeded: {comment}")

        raise CodeforcesAPIError(f"Codeforces API returned error status: {comment}")

    return data.get("result")


def get_user_info(
    handle: str,
    timeout: int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Fetch user profile information from Codeforces.

    Calls: https://codeforces.com/api/user.info?handles={handle}

    Args:
        handle: Codeforces username handle.
        timeout: Request timeout in seconds.
        session: Optional requests.Session for connection reuse.

    Returns:
        dict: A dictionary containing:
            - 'handle': str (Codeforces handle)
            - 'rating': int (User rating; defaults to 800 if unrated)
            - 'maxRating': int (User max rating; defaults to rating if unrated)
            - 'rank': str (User rank string; defaults to 'unrated')
            - 'avatar': str (Direct URL to user avatar photo)

    Raises:
        ValueError: If handle is empty or invalid.
        UserNotFoundError: If user handle does not exist.
        CodeforcesAPIError: On API or network failures.
    """
    clean_handle = handle.strip() if handle else ""
    if not clean_handle:
        raise ValueError("Codeforces handle cannot be empty.")

    result = _make_api_request(
        endpoint="user.info",
        params={"handles": clean_handle},
        timeout=timeout,
        session=session,
    )

    if not result or not isinstance(result, list):
        raise UserNotFoundError(f"No user info returned for handle '{clean_handle}'.")

    user_data = result[0]

    # Handle unrated users: Codeforces omits 'rating', 'maxRating', or 'rank' for unrated users
    rating = user_data.get("rating")
    if rating is None:
        rating = DEFAULT_UNRATED_RATING
    else:
        rating = int(rating)

    max_rating = user_data.get("maxRating")
    if max_rating is None:
        max_rating = rating
    else:
        max_rating = int(max_rating)

    rank = user_data.get("rank")
    if not rank:
        rank = "unrated"

    avatar = user_data.get("avatar", "")
    # Normalize protocol-relative URL if present (e.g. "//userpic.codeforces.org/...")
    if avatar.startswith("//"):
        avatar = f"https:{avatar}"

    return {
        "handle": user_data.get("handle", clean_handle),
        "rating": rating,
        "maxRating": max_rating,
        "rank": rank,
        "avatar": avatar,
    }


def get_user_submissions(
    handle: str,
    count: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch all submissions for a Codeforces user and return as a structured DataFrame.

    Calls: https://codeforces.com/api/user.status?handle={handle}

    Args:
        handle: Codeforces username handle.
        count: Optional limit on number of recent submissions to fetch.
        timeout: Request timeout in seconds.
        session: Optional requests.Session for connection reuse.

    Returns:
        pd.DataFrame: Structured DataFrame containing:
            - 'problem_id': str (e.g., '1915G')
            - 'name': str (Problem title)
            - 'rating': float/int (Problem rating or NaN if unrated)
            - 'tags': list of str (List of topic tags)
            - 'verdict': str (Submission verdict, e.g., 'OK', 'WRONG_ANSWER')
            - 'creationTimeSeconds': int (Epoch timestamp of submission)

    Raises:
        ValueError: If handle is empty.
        UserNotFoundError: If user handle does not exist.
        CodeforcesAPIError: On API or network failures.
    """
    clean_handle = handle.strip() if handle else ""
    if not clean_handle:
        raise ValueError("Codeforces handle cannot be empty.")

    params: Dict[str, Any] = {"handle": clean_handle}
    if count is not None and count > 0:
        params["count"] = count

    result = _make_api_request(
        endpoint="user.status",
        params=params,
        timeout=timeout,
        session=session,
    )

    if result is None:
        result = []

    records = []
    for sub in result:
        problem = sub.get("problem", {})
        contest_id = problem.get("contestId")
        index = str(problem.get("index", "")).strip()

        # Build clean problem_id (e.g., '1915G')
        if contest_id is not None:
            problem_id = f"{contest_id}{index}"
        else:
            problem_id = index

        name = problem.get("name", "")
        rating = problem.get("rating")  # None if unrated
        tags = problem.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        verdict = sub.get("verdict", "UNKNOWN")
        creation_time = sub.get("creationTimeSeconds", 0)

        records.append({
            "problem_id": problem_id,
            "name": name,
            "rating": rating,
            "tags": tags,
            "verdict": verdict,
            "creationTimeSeconds": creation_time,
        })

    columns = [
        "problem_id",
        "name",
        "rating",
        "tags",
        "verdict",
        "creationTimeSeconds",
    ]

    df = pd.DataFrame(records, columns=columns)

    # Ensure types
    if not df.empty:
        df["problem_id"] = df["problem_id"].astype(str)
        df["name"] = df["name"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["verdict"] = df["verdict"].astype(str)
        df["creationTimeSeconds"] = pd.to_numeric(df["creationTimeSeconds"], errors="coerce").fillna(0).astype(int)

    return df


def get_problemset(
    cache_path: str = "data/problems_cache.json",
    force_refresh: bool = False,
    timeout: int = 15,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch the Codeforces problemset with caching and merged problem statistics.

    Checks if cache_path exists and is less than 24 hours old; if so, loads it.
    Otherwise fetches https://codeforces.com/api/problemset.problems, merges problem stats
    (solvedCount from problemStatistics), saves to JSON, and returns a clean DataFrame
    of rated problems with direct URLs (https://codeforces.com/contest/{contestId}/problem/{index}).

    Args:
        cache_path: Filepath for the JSON cache file. Defaults to 'data/problems_cache.json'.
        force_refresh: If True, bypasses cache and fetches fresh data from API.
        timeout: Request timeout in seconds.
        session: Optional requests.Session for connection reuse.

    Returns:
        pd.DataFrame: Clean DataFrame containing rated problems with columns:
            - 'problem_id': str (e.g. '1915G')
            - 'contestId': int
            - 'index': str
            - 'name': str
            - 'rating': int
            - 'tags': list of str
            - 'solvedCount': int
            - 'url': str (Direct link to problem)

    Raises:
        CodeforcesAPIError: On API or network failures when fetching from CF.
    """
    cache_file = Path(cache_path)
    use_cache = False

    # Check cache validity (exists and younger than 24 hours)
    if not force_refresh and cache_file.is_file():
        try:
            mtime = cache_file.stat().st_mtime
            age = time.time() - mtime
            if age < CACHE_EXPIRATION_SECONDS:
                logger.info(f"Loading Codeforces problemset from cache ({cache_path}, age: {age/3600:.1f}h).")
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                use_cache = True
            else:
                logger.info(f"Cache expired ({age/3600:.1f}h old). Fetching fresh problemset.")
        except Exception as exc:
            logger.warning(f"Failed to read cache at {cache_path}: {exc}. Fetching fresh problemset.")
            use_cache = False

    if use_cache:
        merged_problems = cached_data
    else:
        logger.info("Fetching problemset from Codeforces API...")
        result = _make_api_request(
            endpoint="problemset.problems",
            params=None,
            timeout=timeout,
            session=session,
        )

        problems_list = result.get("problems", [])
        stats_list = result.get("problemStatistics", [])

        # Build lookup table for solvedCount: (contestId, index) -> solvedCount
        stats_map: Dict[tuple, int] = {}
        for stat in stats_list:
            cid = stat.get("contestId")
            idx = str(stat.get("index", "")).strip()
            if cid is not None and idx:
                stats_map[(cid, idx)] = int(stat.get("solvedCount", 0))

        # Merge problem info with solvedCount and generate URL
        merged_problems = []
        for prob in problems_list:
            cid = prob.get("contestId")
            idx = str(prob.get("index", "")).strip()
            name = prob.get("name", "")
            rating = prob.get("rating")
            tags = prob.get("tags", [])
            if not isinstance(tags, list):
                tags = []

            # Direct Codeforces problem URL
            if cid is not None and idx:
                direct_url = f"https://codeforces.com/contest/{cid}/problem/{idx}"
                problem_id = f"{cid}{idx}"
            else:
                direct_url = ""
                problem_id = idx

            solved_count = stats_map.get((cid, idx), 0)

            merged_record = {
                "problem_id": problem_id,
                "contestId": cid,
                "index": idx,
                "name": name,
                "rating": rating,
                "tags": tags,
                "solvedCount": solved_count,
                "url": direct_url,
            }
            merged_problems.append(merged_record)

        # Save to JSON cache
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(merged_problems, f, indent=2)
            logger.info(f"Saved {len(merged_problems)} problems to cache at {cache_path}.")
        except Exception as exc:
            logger.warning(f"Failed to write problemset cache to {cache_path}: {exc}")

    # Build DataFrame
    df = pd.DataFrame(merged_problems)
    if df.empty:
        return pd.DataFrame(columns=[
            "problem_id", "contestId", "index", "name", "rating", "tags", "solvedCount", "url"
        ])

    # Filter to only rated problems (rating not null, numeric and > 0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    rated_df = df[df["rating"].notna() & (df["rating"] > 0)].copy()

    # Format data types
    rated_df["rating"] = rated_df["rating"].astype(int)
    rated_df["solvedCount"] = pd.to_numeric(rated_df["solvedCount"], errors="coerce").fillna(0).astype(int)
    rated_df["problem_id"] = rated_df["problem_id"].astype(str)
    rated_df["contestId"] = pd.to_numeric(rated_df["contestId"], errors="coerce").fillna(0).astype(int)
    rated_df["index"] = rated_df["index"].astype(str)
    rated_df["name"] = rated_df["name"].astype(str)
    rated_df["url"] = rated_df["url"].astype(str)

    # Reorder columns cleanly
    columns_order = [
        "problem_id",
        "contestId",
        "index",
        "name",
        "rating",
        "tags",
        "solvedCount",
        "url",
    ]
    rated_df = rated_df[columns_order].reset_index(drop=True)

    return rated_df


def parse_contest_division(name: str) -> str:
    """Parse the Codeforces division from the contest name.

    Returns one of:
    - "Div. 4"
    - "Div. 3"
    - "Div. 2"
    - "Div. 1"
    - "Div. 1 + Div. 2"
    - "Educational"
    - "Other"
    """
    n_lower = name.lower()

    # Educational rounds
    if "educational" in n_lower:
        return "Educational"

    # Combined Div. 1 + Div. 2 / Global Rounds
    if (
        "div. 1 + div. 2" in n_lower
        or "div 1 + div 2" in n_lower
        or "div.1 + div.2" in n_lower
        or "div. 1+2" in n_lower
        or "div 1+2" in n_lower
        or "global round" in n_lower
    ):
        return "Div. 1 + Div. 2"

    # Specific divisions
    if "div. 4" in n_lower or "div 4" in n_lower or "div.4" in n_lower:
        return "Div. 4"
    if "div. 3" in n_lower or "div 3" in n_lower or "div.3" in n_lower:
        return "Div. 3"
    if "div. 2" in n_lower or "div 2" in n_lower or "div.2" in n_lower:
        return "Div. 2"
    if "div. 1" in n_lower or "div 1" in n_lower or "div.1" in n_lower:
        return "Div. 1"

    return "Other"


def format_countdown(diff_seconds: int) -> str:
    """Format seconds remaining into a human-readable countdown string like '1d 4h 30m'."""
    if diff_seconds <= 0:
        return "Starting now"

    days = diff_seconds // 86400
    hours = (diff_seconds % 86400) // 3600
    minutes = (diff_seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")

    return " ".join(parts)


def format_duration(duration_seconds: int) -> str:
    """Format duration in seconds into a human-readable duration like '2 hours' or '2h 15m'."""
    if duration_seconds <= 0:
        return "0m"
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    if hours > 0 and minutes > 0:
        return f"{hours}h {minutes}m"
    elif hours > 0:
        return f"{hours} hours" if hours > 1 else "1 hour"
    return f"{minutes}m"


def is_contest_recommended_for_rating(division: str, user_rating: int) -> bool:
    """Check if a contest division is recommended based on user's Codeforces rating.

    Guidelines:
    - under 1400: Div. 4, Div. 3, Div. 2
    - 1400 to 1899: Div. 3, Div. 2, Educational, Div. 1 + Div. 2
    - 1900+: Div. 1, Div. 2, Div. 1 + Div. 2, Educational
    """
    div = division.strip()
    if user_rating < 1400:
        return div in ("Div. 4", "Div. 3", "Div. 2")
    elif 1400 <= user_rating < 1900:
        return div in ("Div. 3", "Div. 2", "Educational", "Div. 1 + Div. 2")
    else:  # 1900+
        return div in ("Div. 1", "Div. 2", "Div. 1 + Div. 2", "Educational")


def get_upcoming_contests(
    gym: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
    current_time: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Fetch upcoming Codeforces contests (phase == 'BEFORE') sorted soonest first.

    Calls: https://codeforces.com/api/contest.list?gym=false

    Args:
        gym: Whether to include Gym contests (defaults to False).
        timeout: Request timeout in seconds.
        session: Optional requests.Session instance.
        current_time: Optional current epoch timestamp (for testing/mocking).

    Returns:
        List of upcoming contest dictionaries sorted by startTimeSeconds ascending.
    """
    now = current_time if current_time is not None else time.time()
    params = {"gym": "true" if gym else "false"}

    result = _make_api_request(
        "contest.list",
        params=params,
        timeout=timeout,
        session=session,
    )

    if not isinstance(result, list):
        return []

    upcoming: List[Dict[str, Any]] = []
    for c in result:
        phase = c.get("phase")
        if phase != "BEFORE":
            continue

        cid = c.get("id")
        name = str(c.get("name", "Unnamed Contest")).strip()
        start_time = c.get("startTimeSeconds")
        duration = c.get("durationSeconds", 7200)

        # Skip contests without valid future start time
        if start_time is None or not isinstance(start_time, (int, float)):
            continue

        start_time = int(start_time)
        duration = int(duration)
        division = parse_contest_division(name)

        # Convert to IST (Asia/Kolkata)
        dt_ist = datetime.fromtimestamp(start_time, tz=IST_ZONE)
        start_time_ist = dt_ist.strftime("%d %b %Y, %I:%M %p IST")

        # Countdown calculation
        diff_sec = max(0, int(start_time - now))
        countdown = format_countdown(diff_sec)
        dur_str = format_duration(duration)

        # Direct CF contest / registration URL
        contest_url = f"https://codeforces.com/contests/{cid}"

        # 1-click Google Calendar reminder URL in UTC format: YYYYMMDDTHHMMSSZ
        dt_utc_start = datetime.fromtimestamp(start_time, tz=timezone.utc)
        dt_utc_end = datetime.fromtimestamp(start_time + duration, tz=timezone.utc)
        utc_start_str = dt_utc_start.strftime("%Y%m%dT%H%M%SZ")
        utc_end_str = dt_utc_end.strftime("%Y%m%dT%H%M%SZ")

        encoded_name = urllib.parse.quote(name)
        encoded_details = urllib.parse.quote(contest_url)
        gcal_url = (
            f"https://calendar.google.com/calendar/render?action=TEMPLATE"
            f"&text={encoded_name}"
            f"&dates={utc_start_str}/{utc_end_str}"
            f"&details={encoded_details}"
        )

        upcoming.append({
            "id": cid,
            "name": name,
            "division": division,
            "startTimeSeconds": start_time,
            "durationSeconds": duration,
            "duration_formatted": dur_str,
            "start_time_ist": start_time_ist,
            "countdown": countdown,
            "seconds_until_start": diff_sec,
            "url": contest_url,
            "gcal_url": gcal_url,
        })

    # Sort ascending by startTimeSeconds (soonest first)
    upcoming.sort(key=lambda x: x["startTimeSeconds"])
    return upcoming


if __name__ == "__main__":
    # Quick sanity check demonstration
    print("Testing cf_api module...")
    try:
        user = get_user_info("tourist")
        print("Tourist user info:", user)
    except Exception as e:
        print("API call failed (expected in offline/sandboxed environments):", e)
