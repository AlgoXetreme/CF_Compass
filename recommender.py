"""TF-IDF + Cosine Similarity Upsolving Ladder Engine.

Uses scikit-learn (TfidfVectorizer and cosine_similarity) to recommend
a personalized 5-problem progressive upsolving ladder based on the user's
weak topics and current rating level.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analyzer import compute_tag_metrics, get_priority_weak_tags

logger = logging.getLogger(__name__)

# Mapping from Codeforces tags to CSES Problem Categories
CF_TO_CSES_MAP: Dict[str, str] = {
    "dp": "Dynamic Programming",
    "graphs": "Graph Algorithms",
    "shortest paths": "Graph Algorithms",
    "dfs and similar": "Graph Algorithms",
    "dsu": "Graph Algorithms",
    "trees": "Tree Algorithms",
    "binary search": "Sorting and Searching",
    "two pointers": "Sorting and Searching",
    "sortings": "Sorting and Searching",
    "greedy": "Sorting and Searching",
    "data structures": "Range Queries",
    "math": "Mathematics",
    "number theory": "Mathematics",
    "combinatorics": "Mathematics",
    "matrices": "Mathematics",
    "strings": "String Algorithms",
    "string suffix structures": "String Algorithms",
    "geometry": "Geometry",
}


def format_tag_for_tfidf(tag: Any) -> str:
    """Format tag string by replacing internal spaces with underscores.

    Example:
        'binary search' -> 'binary_search'
        'shortest paths' -> 'shortest_paths'
        'dp' -> 'dp'
    """
    if not tag or pd.isna(tag):
        return ""
    return str(tag).strip().lower().replace(" ", "_").replace("-", "_")


def format_tags_string(tags: Any) -> str:
    """Convert a problem's tags list into a space-joined single token string."""
    if not isinstance(tags, (list, tuple, set)):
        return ""
    formatted = [format_tag_for_tfidf(t) for t in tags if t and not pd.isna(t)]
    return " ".join(formatted)


def get_solved_problem_ids(submissions_df: pd.DataFrame) -> Set[str]:
    """Extract set of problem_ids solved by the user (verdict == 'OK')."""
    if submissions_df is None or submissions_df.empty:
        return set()
    solved = submissions_df[submissions_df["verdict"] == "OK"]
    return set(solved["problem_id"].dropna().astype(str).unique())


def generate_upsolving_ladder(
    problemset_df: pd.DataFrame,
    solved_ids: Union[Set[str], List[str]],
    weak_tags_df: pd.DataFrame,
    user_rating: int,
    target_stretch: int = 300,
    similarity_weight: float = 0.75,
    popularity_weight: float = 0.25,
) -> List[Dict[str, Any]]:
    """Generate a 5-problem progressive upsolving ladder using TF-IDF and Cosine Similarity.

    Steps:
    1. Filter out problems whose problem_id is in solved_ids or whose rating is missing.
    2. Convert each problem's tags list into a space-joined string (underscores for multi-word tags).
    3. Fit TfidfVectorizer on the candidate problemset's tag strings.
    4. Construct a weighted target query string from Top 4 weak tags (repeating proportionally to weakness_score).
    5. Compute cosine_similarity between weakness vector and candidate problems + log(solvedCount) popularity boost.
    6. Select a 5-problem progressive upsolving ladder across rating buckets [R, R+100, R+100, R+200, R+300]
       (rounded to nearest 100, min 800), ensuring topic variety.

    Args:
        problemset_df: DataFrame of problems containing ['problem_id', 'name', 'rating', 'tags', 'solvedCount', 'url'].
        solved_ids: Set or collection of problem_ids already solved by the user.
        weak_tags_df: DataFrame of weak tags from analyzer containing ['tag', 'weakness_score'].
        user_rating: User's rating (int).
        similarity_weight: Weight for cosine similarity (default: 0.75).
        popularity_weight: Weight for log(solvedCount) popularity boost (default: 0.25).

    Returns:
        List[Dict[str, Any]]: 5 recommended ladder problems as a list of dicts.
    """
    if problemset_df is None or problemset_df.empty:
        return []

    solved_set = set(str(pid) for pid in (solved_ids or []))

    # 1. Filter out solved problems or problems with missing/invalid rating
    candidates = problemset_df.copy()
    candidates["problem_id"] = candidates["problem_id"].astype(str)
    candidates["rating"] = pd.to_numeric(candidates["rating"], errors="coerce")
    candidates = candidates[
        candidates["rating"].notna()
        & (candidates["rating"] > 0)
        & (~candidates["problem_id"].isin(solved_set))
    ].copy()

    if candidates.empty:
        return []

    # 2. Convert each problem's tags list into a space-joined string with underscores
    candidates["tag_tokens"] = candidates["tags"].apply(format_tags_string)
    # Exclude candidates with no tags at all
    candidates = candidates[candidates["tag_tokens"].str.strip() != ""].copy()
    if candidates.empty:
        return []

    # 3. Fit TfidfVectorizer on candidate problemset's tag strings
    # Use token_pattern=r"\S+" so each space-delimited tag (e.g. binary_search, dp) is an exact token
    vectorizer = TfidfVectorizer(token_pattern=r"\S+", lowercase=True)
    problem_matrix = vectorizer.fit_transform(candidates["tag_tokens"].tolist())

    # 4. Construct a weighted target query string from user's Top 4 weak tags
    # Handle empty or missing weak_tags_df
    if weak_tags_df is not None and not weak_tags_df.empty and "tag" in weak_tags_df.columns:
        top_weak = weak_tags_df.head(4).copy()
        # Use boosted_score if present, otherwise weakness_score
        score_col = "boosted_score" if "boosted_score" in top_weak.columns else "weakness_score"
        if score_col not in top_weak.columns:
            top_weak["score"] = 1.0
        else:
            top_weak["score"] = pd.to_numeric(top_weak[score_col], errors="coerce").fillna(1.0)

        # Ensure positive scores
        max_score = float(top_weak["score"].max()) if not top_weak.empty else 1.0
        max_score = max(max_score, 1.0)

        query_tokens: List[str] = []
        top_weak_tag_names: List[str] = []
        for _, row in top_weak.iterrows():
            raw_tag = row["tag"]
            token = format_tag_for_tfidf(raw_tag)
            if not token:
                continue
            top_weak_tag_names.append(raw_tag)
            # Repeat token proportionally to its weakness score (between 1 and 6 times)
            raw_score = float(row["score"])
            repeat_count = max(1, min(6, int(round((raw_score / max_score) * 6))))
            query_tokens.extend([token] * repeat_count)

        query_string = " ".join(query_tokens)
    else:
        # Fallback to standard core competitive programming topics
        query_string = "dp graphs binary_search greedy"
        top_weak_tag_names = ["dp", "graphs", "binary search", "greedy"]

    # Transform query string using fitted vectorizer
    query_vector = vectorizer.transform([query_string])

    # 5. Compute cosine similarity and combine with normalized log(solvedCount) popularity boost
    similarities = cosine_similarity(query_vector, problem_matrix).flatten()
    candidates["similarity_score"] = np.round(similarities, 4)

    # Compute normalized popularity boost via log1p(solvedCount)
    solved_counts = pd.to_numeric(candidates["solvedCount"], errors="coerce").fillna(0).values
    log_solved = np.log1p(np.maximum(0, solved_counts))
    max_log = log_solved.max()
    min_log = log_solved.min()

    if max_log > min_log:
        norm_pop = (log_solved - min_log) / (max_log - min_log)
    else:
        norm_pop = np.zeros_like(log_solved)

    candidates["popularity_boost"] = np.round(norm_pop, 4)

    # Combined score
    candidates["combined_score"] = np.round(
        (similarity_weight * candidates["similarity_score"])
        + (popularity_weight * candidates["popularity_boost"]),
        4,
    )

    # Helper function to extract primary matching weak tag for variety tracking
    def get_primary_tag(tags_list: Any) -> str:
        if not isinstance(tags_list, list):
            return "general"
        # Check against user's top weak tags
        for wt in top_weak_tag_names:
            if wt in tags_list:
                return wt
        return tags_list[0] if tags_list else "general"

    candidates["primary_tag"] = candidates["tags"].apply(get_primary_tag)

    # 6. Select a 5-problem progressive upsolving ladder across 5 rating buckets
    # Default Buckets: [R, R+100, R+100, R+200, R+300] (rounded to nearest 100, min 800)
    base_r = max(800, int(round(user_rating / 100.0) * 100))
    step1_delta = max(0, int(round((target_stretch / 3.0) / 100.0) * 100))
    step2_delta = max(step1_delta, int(round((2.0 * target_stretch / 3.0) / 100.0) * 100))
    target_buckets = [
        base_r,
        base_r + step1_delta,
        base_r + step1_delta,
        base_r + step2_delta,
        base_r + target_stretch,
    ]

    selected_problems: List[Dict[str, Any]] = []
    used_problem_ids: Set[str] = set()
    used_tags_history: List[str] = []

    # Pool of candidate records for fast lookup
    for step_idx, target_r in enumerate(target_buckets, start=1):
        # Look for problems at exact target rating first
        bucket_candidates = candidates[
            (candidates["rating"] == target_r)
            & (~candidates["problem_id"].isin(used_problem_ids))
        ].copy()

        # If no problems at exact target rating, widen search window (+- 50 to +- 100)
        if bucket_candidates.empty:
            bucket_candidates = candidates[
                (candidates["rating"] >= target_r - 100)
                & (candidates["rating"] <= target_r + 100)
                & (~candidates["problem_id"].isin(used_problem_ids))
            ].copy()

        # Fallback to any remaining candidate if still empty
        if bucket_candidates.empty:
            bucket_candidates = candidates[
                ~candidates["problem_id"].isin(used_problem_ids)
            ].copy()

        if bucket_candidates.empty:
            # No more problems available
            break

        # Sort bucket candidates by combined_score descending
        bucket_candidates = bucket_candidates.sort_values(
            by=["combined_score", "similarity_score", "solvedCount"],
            ascending=[False, False, False],
        )

        # Enforce tag variety:
        # Prefer a candidate whose primary tag has not been used recently in the ladder
        chosen_row = None
        # Try to find candidate whose primary tag hasn't been used yet
        for _, row in bucket_candidates.iterrows():
            row_tag = row["primary_tag"]
            if row_tag not in used_tags_history:
                chosen_row = row
                break

        # If all candidates share already-used tags, pick candidate that doesn't match the immediate previous tag
        if chosen_row is None and len(bucket_candidates) > 1 and used_tags_history:
            prev_tag = used_tags_history[-1]
            for _, row in bucket_candidates.iterrows():
                if row["primary_tag"] != prev_tag:
                    chosen_row = row
                    break

        # Otherwise pick top scoring candidate
        if chosen_row is None:
            chosen_row = bucket_candidates.iloc[0]

        # Record selection
        pid = str(chosen_row["problem_id"])
        used_problem_ids.add(pid)
        used_tags_history.append(chosen_row["primary_tag"])

        selected_problems.append({
            "ladder_step": step_idx,
            "target_rating": target_r,
            "problem_id": pid,
            "name": str(chosen_row.get("name", "")),
            "rating": int(chosen_row.get("rating", target_r)),
            "tags": chosen_row.get("tags", []),
            "solvedCount": int(chosen_row.get("solvedCount", 0)),
            "url": str(chosen_row.get("url", "")),
            "similarity_score": float(chosen_row.get("similarity_score", 0.0)),
            "combined_score": float(chosen_row.get("combined_score", 0.0)),
            "primary_tag": str(chosen_row.get("primary_tag", "")),
        })

    return selected_problems


# Backward compatibility wrapper for previous recommender function
def recommend_problems(
    problemset_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    user_rating: int,
    weak_tags: Optional[List[str]] = None,
    rating_range: tuple = (-100, 200),
    top_n: int = 15,
) -> pd.DataFrame:
    """Recommend problems using upsolving ladder or simple tag filter."""
    solved_ids = get_solved_problem_ids(submissions_df)
    weakness_df = compute_tag_metrics(submissions_df, user_rating=user_rating)

    ladder = generate_upsolving_ladder(
        problemset_df=problemset_df,
        solved_ids=solved_ids,
        weak_tags_df=weakness_df,
        user_rating=user_rating,
    )

    if ladder:
        return pd.DataFrame(ladder)

    return pd.DataFrame()


def analyze_topic_performance(submissions_df: pd.DataFrame, user_rating: int = 800) -> pd.DataFrame:
    """Analyze topic performance using analyzer.compute_tag_metrics."""
    return compute_tag_metrics(submissions_df, user_rating=user_rating)


# Starter curated CP-31 problems to auto-seed if file is missing
DEFAULT_STARTER_CP31: List[Dict[str, Any]] = [
    {"problem_id": "1903A", "name": "Halloumi Boxes", "rating": 800, "sheet_tier": 800, "tags": ["sortings", "brute force"]},
    {"problem_id": "1901A", "name": "Line Trip", "rating": 800, "sheet_tier": 800, "tags": ["greedy", "math"]},
    {"problem_id": "1900A", "name": "Cover in Water", "rating": 800, "sheet_tier": 800, "tags": ["constructive algorithms", "greedy"]},
    {"problem_id": "1904A", "name": "Forked!", "rating": 900, "sheet_tier": 900, "tags": ["brute force", "math"]},
    {"problem_id": "1883B", "name": "Chemistry", "rating": 900, "sheet_tier": 900, "tags": ["greedy", "strings"]},
    {"problem_id": "1878C", "name": "Vasilije in Cacak", "rating": 900, "sheet_tier": 900, "tags": ["greedy", "math"]},
    {"problem_id": "1859B", "name": "Olya and Game with Arrays", "rating": 1000, "sheet_tier": 1000, "tags": ["greedy", "math", "sortings"]},
    {"problem_id": "1849B", "name": "Monsters", "rating": 1000, "sheet_tier": 1000, "tags": ["greedy", "sortings"]},
    {"problem_id": "1840C", "name": "Ski Resort", "rating": 1000, "sheet_tier": 1000, "tags": ["combinatorics", "math", "two pointers"]},
    {"problem_id": "1914D", "name": "Three Activities", "rating": 1100, "sheet_tier": 1100, "tags": ["brute force", "dp", "greedy", "sortings"]},
    {"problem_id": "1904B", "name": "Collecting Game", "rating": 1100, "sheet_tier": 1100, "tags": ["binary search", "greedy", "two pointers"]},
    {"problem_id": "1899C", "name": "Yarik and Array", "rating": 1100, "sheet_tier": 1100, "tags": ["dp", "greedy", "two pointers"]},
    {"problem_id": "1904C", "name": "Array Game", "rating": 1200, "sheet_tier": 1200, "tags": ["binary search", "brute force", "sortings", "two pointers"]},
    {"problem_id": "1872E", "name": "Data Structures Fan", "rating": 1200, "sheet_tier": 1200, "tags": ["bitmasks", "data structures"]},
    {"problem_id": "1857C", "name": "Assembly via Minimums", "rating": 1200, "sheet_tier": 1200, "tags": ["greedy", "sortings"]},
    {"problem_id": "1915E", "name": "Romantic Glasses", "rating": 1300, "sheet_tier": 1300, "tags": ["data structures", "sortings"]},
    {"problem_id": "1879C", "name": "Make it Alternating", "rating": 1300, "sheet_tier": 1300, "tags": ["combinatorics", "dp", "greedy"]},
    {"problem_id": "1862D", "name": "Ice Cream Balls", "rating": 1300, "sheet_tier": 1300, "tags": ["binary search", "combinatorics", "math"]},
    {"problem_id": "1914E1", "name": "Game with Marbles (Easy)", "rating": 1400, "sheet_tier": 1400, "tags": ["games", "greedy", "sortings"]},
    {"problem_id": "1872C", "name": "Non-coprime Split", "rating": 1400, "sheet_tier": 1400, "tags": ["math", "number theory"]},
    {"problem_id": "1850F", "name": "We Were Both Children", "rating": 1400, "sheet_tier": 1400, "tags": ["brute force", "math", "number theory"]},
    {"problem_id": "1914E2", "name": "Game with Marbles (Hard)", "rating": 1500, "sheet_tier": 1500, "tags": ["games", "greedy", "sortings"]},
    {"problem_id": "1873G", "name": "ABBC or BACB", "rating": 1500, "sheet_tier": 1500, "tags": ["greedy"]},
    {"problem_id": "1870C", "name": "Colorful Table", "rating": 1500, "sheet_tier": 1500, "tags": ["binary search", "data structures", "two pointers"]},
    {"problem_id": "1883G1", "name": "Dances (Easy version)", "rating": 1600, "sheet_tier": 1600, "tags": ["binary search", "greedy", "two pointers"]},
    {"problem_id": "1872F", "name": "Money Trees", "rating": 1600, "sheet_tier": 1600, "tags": ["binary search", "data structures", "two pointers"]},
    {"problem_id": "1899F", "name": "Alex's whims", "rating": 1700, "sheet_tier": 1700, "tags": ["constructive algorithms", "trees"]},
    {"problem_id": "1878E", "name": "Iva & Pav", "rating": 1700, "sheet_tier": 1700, "tags": ["binary search", "bitmasks", "data structures"]},
    {"problem_id": "1914F", "name": "Programming Competition", "rating": 1800, "sheet_tier": 1800, "tags": ["dfs and similar", "dp", "greedy", "trees"]},
    {"problem_id": "1863E", "name": "Speedrun", "rating": 1800, "sheet_tier": 1800, "tags": ["dp", "graphs", "two pointers"]},
    {"problem_id": "1900D", "name": "Small GCD", "rating": 1900, "sheet_tier": 1900, "tags": ["combinatorics", "dp", "math", "number theory"]},
    {"problem_id": "1856C", "name": "To Become Max", "rating": 1900, "sheet_tier": 1900, "tags": ["binary search", "dp", "greedy"]},
]

# Starter curated CSES problems to auto-seed if file is missing
DEFAULT_STARTER_CSES: List[Dict[str, Any]] = [
    {"id": 1633, "name": "Dice Combinations", "category": "Dynamic Programming", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1633"},
    {"id": 1634, "name": "Minimizing Coins", "category": "Dynamic Programming", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1634"},
    {"id": 1635, "name": "Coin Combinations I", "category": "Dynamic Programming", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1635"},
    {"id": 1636, "name": "Coin Combinations II", "category": "Dynamic Programming", "difficulty": "Medium", "url": "https://cses.fi/problemset/task/1636"},
    {"id": 1637, "name": "Removing Digits", "category": "Dynamic Programming", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1637"},
    {"id": 1638, "name": "Grid Paths", "category": "Dynamic Programming", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1638"},
    {"id": 1192, "name": "Counting Rooms", "category": "Graph Algorithms", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1192"},
    {"id": 1193, "name": "Labyrinth", "category": "Graph Algorithms", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1193"},
    {"id": 1666, "name": "Building Roads", "category": "Graph Algorithms", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1666"},
    {"id": 1667, "name": "Message Route", "category": "Graph Algorithms", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1667"},
    {"id": 1668, "name": "Building Teams", "category": "Graph Algorithms", "difficulty": "Medium", "url": "https://cses.fi/problemset/task/1668"},
    {"id": 1674, "name": "Subordinates", "category": "Tree Algorithms", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1674"},
    {"id": 1130, "name": "Tree Matching", "category": "Tree Algorithms", "difficulty": "Medium", "url": "https://cses.fi/problemset/task/1130"},
    {"id": 1621, "name": "Distinct Numbers", "category": "Sorting and Searching", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1621"},
    {"id": 1084, "name": "Apartments", "category": "Sorting and Searching", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1084"},
    {"id": 1090, "name": "Ferris Wheel", "category": "Sorting and Searching", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1090"},
    {"id": 1640, "name": "Sum of Two Values", "category": "Sorting and Searching", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1640"},
    {"id": 1643, "name": "Maximum Subarray Sum", "category": "Sorting and Searching", "difficulty": "Easy", "url": "https://cses.fi/problemset/task/1643"},
]


def ensure_data_files(data_dir: str = "data") -> None:
    """Ensure data/ directory and default JSON files exist safely."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)

    cp31_p = d / "cp31.json"
    if not cp31_p.is_file():
        try:
            with open(cp31_p, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_STARTER_CP31, f, indent=2)
            logger.info(f"Auto-created default CP-31 dataset at {cp31_p}.")
        except Exception as exc:
            logger.warning(f"Failed to auto-create {cp31_p}: {exc}")

    cses_p = d / "cses.json"
    if not cses_p.is_file():
        try:
            with open(cses_p, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_STARTER_CSES, f, indent=2)
            logger.info(f"Auto-created default CSES dataset at {cses_p}.")
        except Exception as exc:
            logger.warning(f"Failed to auto-create {cses_p}: {exc}")

    progress_p = d / "cses_progress.json"
    if not progress_p.is_file():
        try:
            with open(progress_p, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
            logger.info(f"Auto-created initial empty CSES progress at {progress_p}.")
        except Exception as exc:
            logger.warning(f"Failed to auto-create {progress_p}: {exc}")


def load_cp31_dataset(path: str = "data/cp31.json") -> pd.DataFrame:
    """Load the curated CP-31 dataset from JSON file with auto-creation fallback."""
    p = Path(path)
    if not p.is_file():
        ensure_data_files(str(p.parent))
    if not p.is_file():
        logger.warning(f"CP-31 dataset file not found at {path}. Using embedded starter data.")
        return pd.DataFrame(DEFAULT_STARTER_CP31)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as exc:
        logger.error(f"Error loading CP-31 dataset from {path}: {exc}")
        return pd.DataFrame(DEFAULT_STARTER_CP31)


def load_cses_dataset(path: str = "data/cses.json") -> pd.DataFrame:
    """Load the curated CSES dataset from JSON file with auto-creation fallback."""
    p = Path(path)
    if not p.is_file():
        ensure_data_files(str(p.parent))
    if not p.is_file():
        logger.warning(f"CSES dataset file not found at {path}. Using embedded starter data.")
        return pd.DataFrame(DEFAULT_STARTER_CSES)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as exc:
        logger.error(f"Error loading CSES dataset from {path}: {exc}")
        return pd.DataFrame(DEFAULT_STARTER_CSES)


def load_cses_progress(path: str = "data/cses_progress.json") -> Set[int]:
    """Load the set of completed CSES problem IDs from local progress JSON with auto-creation fallback."""
    p = Path(path)
    if not p.is_file():
        ensure_data_files(str(p.parent))
    if not p.is_file():
        return set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(int(x) for x in data if str(x).isdigit())
    except Exception as exc:
        logger.warning(f"Could not load CSES progress from {path}: {exc}")
        return set()


# Auto-ensure default data files at module import
ensure_data_files()


def save_cses_progress(completed_ids: Union[Set[int], List[int]], path: str = "data/cses_progress.json") -> None:
    """Persist the set of completed CSES problem IDs to local progress JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(list(set(int(x) for x in completed_ids)))
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sorted_ids, f, indent=2)
    except Exception as exc:
        logger.error(f"Error saving CSES progress to {path}: {exc}")


def _build_cf_url(pid: str) -> str:
    """Helper to build Codeforces problem URL from problem_id (e.g. 1915G)."""
    pid_str = str(pid).strip()
    match = re.match(r"^(\d+)([A-Za-z0-9_]+)$", pid_str)
    if match:
        cid, idx = match.group(1), match.group(2)
        return f"https://codeforces.com/contest/{cid}/problem/{idx}"
    return f"https://codeforces.com/problemset/problem/{pid_str}"


def _prepare_merged_cp31(
    cp31_df: pd.DataFrame,
    problemset_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Helper to merge cp31_df with problemset_df preserving names, tags, ratings, and urls."""
    if cp31_df is None or cp31_df.empty:
        return pd.DataFrame()

    merged = cp31_df.copy()
    merged["problem_id"] = merged["problem_id"].astype(str)

    if problemset_df is not None and not problemset_df.empty:
        cols = [c for c in ["problem_id", "name", "tags", "url", "solvedCount", "rating"] if c in problemset_df.columns]
        p_sub = problemset_df[cols].copy()
        p_sub["problem_id"] = p_sub["problem_id"].astype(str)
        # Suffix problemset columns if collision occurs
        merged = pd.merge(merged, p_sub, on="problem_id", how="left", suffixes=("", "_ps"))

        # Fallback names
        if "name_ps" in merged.columns:
            merged["name"] = merged["name"].fillna(merged["name_ps"]).fillna(merged["problem_id"])
        elif "name" in merged.columns:
            merged["name"] = merged["name"].fillna(merged["problem_id"])
        else:
            merged["name"] = merged["problem_id"]

        # Fallback tags
        if "tags_ps" in merged.columns:
            merged["tags"] = merged.apply(
                lambda r: r["tags_ps"] if isinstance(r.get("tags_ps"), list) and r.get("tags_ps")
                else (r["tags"] if isinstance(r.get("tags"), list) else []),
                axis=1,
            )
        elif "tags" in merged.columns:
            merged["tags"] = merged["tags"].apply(lambda t: t if isinstance(t, list) else [])
        else:
            merged["tags"] = [[] for _ in range(len(merged))]

        # Fallback rating
        if "rating_ps" in merged.columns:
            merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce").fillna(merged["rating_ps"])
        merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce").fillna(merged.get("sheet_tier", 800)).astype(int)

        # Fallback solvedCount
        merged["solvedCount"] = pd.to_numeric(merged.get("solvedCount", 0), errors="coerce").fillna(0).astype(int)

        # Fallback URL
        if "url" not in merged.columns or merged["url"].isna().all():
            merged["url"] = merged["problem_id"].apply(_build_cf_url)
        else:
            merged["url"] = merged["url"].fillna("").apply(
                lambda u: u if u else _build_cf_url(merged["problem_id"])
            )
    else:
        if "name" not in merged.columns:
            merged["name"] = merged["problem_id"]
        else:
            merged["name"] = merged["name"].fillna(merged["problem_id"])

        if "tags" not in merged.columns:
            merged["tags"] = [[] for _ in range(len(merged))]
        else:
            merged["tags"] = merged["tags"].apply(lambda t: t if isinstance(t, list) else [])

        merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce").fillna(merged.get("sheet_tier", 800)).astype(int)
        merged["solvedCount"] = 0
        merged["url"] = merged["problem_id"].apply(_build_cf_url)

    return merged


def recommend_cp31(
    cp31_df: pd.DataFrame,
    problemset_df: pd.DataFrame,
    solved_ids: Union[Set[str], List[str]],
    weak_tags_df: pd.DataFrame,
    user_rating: int,
) -> pd.DataFrame:
    """Recommend 5 unsolved CP-31 Sheet problems matching user's weak tags within [user_rating, user_rating + 300].

    Args:
        cp31_df: DataFrame of CP-31 problems with ['problem_id', 'rating', 'sheet_tier'].
        problemset_df: Codeforces problemset DataFrame from cf_api.get_problemset.
        solved_ids: Set of problem IDs solved by the user.
        weak_tags_df: Sorted DataFrame of weak tags from analyzer.
        user_rating: Current rating of the user.

    Returns:
        pd.DataFrame: Top 5 recommended CP-31 problems with columns:
            - 'problem_id'
            - 'name'
            - 'rating'
            - 'sheet_tier'
            - 'tags'
            - 'solvedCount'
            - 'url'
            - 'matched_weak_tags'
    """
    out_cols = ["problem_id", "name", "rating", "sheet_tier", "tags", "solvedCount", "url", "matched_weak_tags"]

    if cp31_df is None or cp31_df.empty:
        return pd.DataFrame(columns=out_cols)

    solved_set = set(str(pid) for pid in (solved_ids or []))

    # Join cp31_df with problemset_df preserving names, tags, ratings, and urls
    merged = _prepare_merged_cp31(cp31_df, problemset_df)
    if merged.empty:
        return pd.DataFrame(columns=out_cols)

    # Filter out already solved problems
    candidates = merged[~merged["problem_id"].isin(solved_set)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=out_cols)

    # Filter within rating window [user_rating, user_rating + 300]
    min_r = max(800, user_rating)
    max_r = user_rating + 300
    window_candidates = candidates[
        (candidates["rating"] >= min_r) & (candidates["rating"] <= max_r)
    ].copy()

    # Fallback to broader window if exact bracket is sparse
    if len(window_candidates) < 5:
        window_candidates = candidates[
            (candidates["rating"] >= min_r - 200) & (candidates["rating"] <= max_r + 200)
        ].copy()

    if window_candidates.empty:
        window_candidates = candidates.copy()

    # Extract user's top weak tags
    top_weak_tags: List[str] = []
    if weak_tags_df is not None and not weak_tags_df.empty and "tag" in weak_tags_df.columns:
        top_weak_tags = [str(t).lower() for t in weak_tags_df.head(5)["tag"]]

    # Score candidates based on weak tag match
    def evaluate_tags(tag_list):
        if not isinstance(tag_list, list) or not top_weak_tags:
            return 0, []
        matched = [t for t in tag_list if str(t).lower() in top_weak_tags]
        # Weight matches: earlier weak tags carry higher weight
        score = 0
        for m in matched:
            idx = top_weak_tags.index(str(m).lower())
            score += (len(top_weak_tags) - idx)
        return score, matched

    scores_and_matches = window_candidates["tags"].apply(evaluate_tags)
    window_candidates["match_score"] = [s[0] for s in scores_and_matches]
    window_candidates["matched_weak_tags"] = [s[1] for s in scores_and_matches]

    # Sort by match_score descending, rating ascending, solvedCount descending
    sorted_df = window_candidates.sort_values(
        by=["match_score", "rating", "solvedCount"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    return sorted_df.head(5)[out_cols]


def recommend_cses(
    cses_df: pd.DataFrame,
    weak_tags_df: pd.DataFrame,
    completed_cses_ids: Union[Set[int], List[int]],
) -> pd.DataFrame:
    """Recommend 5 foundational CSES problems mapped from user's top 3 weak tags ordered Easy to Hard.

    Args:
        cses_df: DataFrame of CSES problems with ['id', 'name', 'category', 'difficulty', 'url'].
        weak_tags_df: Sorted DataFrame of weak tags from analyzer.
        completed_cses_ids: Set of CSES problem IDs already marked completed by the user.

    Returns:
        pd.DataFrame: Top 5 CSES problems ordered from Easy to Hard with columns:
            - 'id'
            - 'name'
            - 'category'
            - 'difficulty'
            - 'url'
    """
    out_cols = ["id", "name", "category", "difficulty", "url"]

    if cses_df is None or cses_df.empty:
        return pd.DataFrame(columns=out_cols)

    completed_set = set(int(cid) for cid in (completed_cses_ids or []))

    # Filter out completed CSES problems
    candidates = cses_df.copy()
    candidates["id"] = candidates["id"].astype(int)
    candidates = candidates[~candidates["id"].isin(completed_set)].copy()

    if candidates.empty:
        return pd.DataFrame(columns=out_cols)

    # Map user's Top 3 Codeforces weak tags to CSES categories
    top_3_cf: List[str] = []
    if weak_tags_df is not None and not weak_tags_df.empty and "tag" in weak_tags_df.columns:
        top_3_cf = [str(t).lower() for t in weak_tags_df.head(3)["tag"]]

    target_categories: List[str] = []
    for tag in top_3_cf:
        cses_cat = CF_TO_CSES_MAP.get(tag)
        if cses_cat and cses_cat not in target_categories:
            target_categories.append(cses_cat)

    if not target_categories:
        target_categories = ["Dynamic Programming", "Graph Algorithms", "Sorting and Searching"]

    # Filter problems belonging to target categories
    matched = candidates[candidates["category"].isin(target_categories)].copy()

    # If fewer than 5 problems in target categories, append other foundational categories
    if len(matched) < 5:
        remaining = candidates[~candidates["id"].isin(matched["id"])].copy()
        matched = pd.concat([matched, remaining], ignore_index=True)

    # Order from Easy to Hard
    difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
    matched["diff_order"] = matched["difficulty"].map(difficulty_map).fillna(2)

    sorted_df = matched.sort_values(
        by=["diff_order", "id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return sorted_df.head(5)[out_cols]


def get_cp31_unique_tags(
    cp31_df: pd.DataFrame,
    problemset_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Extract a sorted list of unique algorithmic tags available in the CP-31 sheet dataset."""
    merged = _prepare_merged_cp31(cp31_df, problemset_df)
    if merged.empty:
        return []

    unique_tags = set()
    for tag_list in merged["tags"]:
        if isinstance(tag_list, list):
            for t in tag_list:
                cleaned = str(t).strip().lower()
                if cleaned and cleaned not in ("*special",):
                    unique_tags.add(cleaned)

    return sorted(list(unique_tags))


def get_cp31_topic_ladder(
    cp31_df: pd.DataFrame,
    problemset_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    selected_tag: str,
    min_rating: int = 800,
    max_rating: int = 1900,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a topic-specific CP-31 rating ladder in ascending difficulty with user solve status.

    Args:
        cp31_df: DataFrame of CP-31 problems from data/cp31.json.
        problemset_df: Problemset DataFrame from cf_api.get_problemset.
        submissions_df: Submissions DataFrame from cf_api.get_user_submissions.
        selected_tag: Topic/tag to filter (e.g. 'binary search', 'dp', 'graphs').
        min_rating: Minimum problem rating (default: 800).
        max_rating: Maximum problem rating (default: 1900).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - ladder_df: Problems matching selected_tag within [min_rating, max_rating] sorted
              strictly by rating ascending (solvedCount descending tie-breaker), with columns:
              ['problem_id', 'name', 'rating', 'sheet_tier', 'tags', 'url', 'solvedCount', 'status']
            - summary_stats: Dict with {'total_count', 'solved_count', 'stuck_count', 'todo_count', 'solve_rate'}
    """
    empty_stats = {
        "total_count": 0,
        "solved_count": 0,
        "stuck_count": 0,
        "todo_count": 0,
        "solve_rate": 0.0,
    }
    out_cols = ["problem_id", "name", "rating", "sheet_tier", "tags", "url", "solvedCount", "status"]

    if cp31_df is None or cp31_df.empty:
        return pd.DataFrame(columns=out_cols), empty_stats

    # 1. Merge cp31_df with problemset_df
    merged = _prepare_merged_cp31(cp31_df, problemset_df)
    if merged.empty:
        return pd.DataFrame(columns=out_cols), empty_stats

    # 2. Filter rows where selected_tag is present (case-insensitive) and rating within [min_rating, max_rating]
    target_tag = str(selected_tag).strip().lower()

    def matches_tag(tags_list):
        if not isinstance(tags_list, list):
            return False
        return any(str(t).strip().lower() == target_tag for t in tags_list)

    filtered = merged[merged["tags"].apply(matches_tag)].copy()

    # Filter rating range
    filtered = filtered[
        (filtered["rating"] >= min_rating) & (filtered["rating"] <= max_rating)
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=out_cols), empty_stats

    # 3. Cross-check against submissions_df to assign status ('Solved', 'Stuck', 'Todo')
    solved_pids: Set[str] = set()
    attempted_pids: Set[str] = set()

    if submissions_df is not None and not submissions_df.empty:
        if "verdict" in submissions_df.columns and "problem_id" in submissions_df.columns:
            solved_pids = set(
                submissions_df[submissions_df["verdict"] == "OK"]["problem_id"].dropna().astype(str)
            )
            attempted_pids = set(
                submissions_df["problem_id"].dropna().astype(str)
            )

    def determine_status(pid):
        pid_str = str(pid)
        if pid_str in solved_pids:
            return "Solved"
        elif pid_str in attempted_pids:
            return "Stuck"
        return "Todo"

    filtered["status"] = filtered["problem_id"].apply(determine_status)

    # 4. Sort strictly by rating ascending (and solvedCount descending as tie-breaker)
    sorted_ladder = filtered.sort_values(
        by=["rating", "solvedCount"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # 5. Calculate summary stats
    total_count = len(sorted_ladder)
    solved_count = int((sorted_ladder["status"] == "Solved").sum())
    stuck_count = int((sorted_ladder["status"] == "Stuck").sum())
    todo_count = int((sorted_ladder["status"] == "Todo").sum())
    solve_rate = round(solved_count / total_count, 3) if total_count > 0 else 0.0

    summary_stats = {
        "total_count": total_count,
        "solved_count": solved_count,
        "stuck_count": stuck_count,
        "todo_count": todo_count,
        "solve_rate": solve_rate,
    }

    # Ensure output columns exist
    for col in out_cols:
        if col not in sorted_ladder.columns:
            sorted_ladder[col] = None

    return sorted_ladder[out_cols], summary_stats


