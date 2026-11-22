"""Weak-Topic Analyzer & DAG Prerequisite Engine for Codeforces Submissions.

Analyzes a user's submission history across algorithmic tags, computes
granular weakness metrics and failure ratios, and applies a Directed Acyclic Graph
(DAG) prerequisite boost to prioritize foundational topics over dependent advanced topics.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Default trivial or meta tags on Codeforces to filter out of core topic weakness analysis
DEFAULT_TRIVIAL_TAGS: Set[str] = {
    "*special",
    "implementation",
}

# Lightweight Directed Acyclic Graph (DAG) mapping foundational prerequisite tags -> advanced tags
TOPIC_DAG: Dict[str, List[str]] = {
    "dp": ["trees", "bitmasks", "graphs", "probabilities"],
    "binary search": ["greedy", "two pointers", "ternary search"],
    "graphs": ["shortest paths", "dfs and similar", "dsu", "flows", "graph matchings"],
    "math": ["number theory", "combinatorics", "matrices", "geometry"],
    "data structures": ["dsu", "trees", "bitmasks"],
    "strings": ["string suffix structures", "expression parsing"],
}


def compute_tag_metrics(
    submissions_df: pd.DataFrame,
    user_rating: int = 800,
    ignored_tags: Optional[Set[str]] = None,
    min_attempts: int = 1,
) -> pd.DataFrame:
    """Analyze submission history by topic tag and calculate granular weakness metrics.

    Groups by each tag across all problems. For each tag:
    - unique_solved: distinct problems solved with verdict == 'OK'
    - failed_attempts: submissions with verdict != 'OK' on problems never solved or struggled on
    - avg_solved_rating: average rating of unique problems solved with this tag
    - weakness_score: computed from failure-to-solve ratio and rating deficit relative to user_rating

    Args:
        submissions_df: DataFrame from cf_api.get_user_submissions containing
            ['problem_id', 'name', 'rating', 'tags', 'verdict', 'creationTimeSeconds'].
        user_rating: User's current Codeforces rating (defaults to 800 for unrated).
        ignored_tags: Set of tags to exclude (defaults to DEFAULT_TRIVIAL_TAGS).
        min_attempts: Minimum total submissions required to evaluate a tag (default: 1).

    Returns:
        pd.DataFrame: Clean DataFrame sorted from weakest to strongest tag with columns:
            - 'tag': Topic tag name
            - 'unique_solved': Count of distinct solved problems
            - 'problems_attempted': Count of distinct problems attempted
            - 'failed_attempts': Total non-OK submissions
            - 'avg_solved_rating': Average rating of solved problems in this tag
            - 'failure_ratio': Ratio of failed attempts to (solved + 1)
            - 'solve_rate': Fraction of attempted problems solved
            - 'rating_deficit': Max(0, user_rating - avg_solved_rating)
            - 'weakness_score': Composite weakness score (higher = weaker)
    """
    output_columns = [
        "tag",
        "unique_solved",
        "problems_attempted",
        "failed_attempts",
        "avg_solved_rating",
        "failure_ratio",
        "solve_rate",
        "rating_deficit",
        "weakness_score",
    ]

    if submissions_df is None or submissions_df.empty:
        return pd.DataFrame(columns=output_columns)

    if ignored_tags is None:
        ignored_tags = DEFAULT_TRIVIAL_TAGS

    # Ensure required columns exist
    required_cols = {"problem_id", "tags", "verdict"}
    if not required_cols.issubset(submissions_df.columns):
        return pd.DataFrame(columns=output_columns)

    # Explode tags so each row represents a single (submission, tag) pair
    exploded = submissions_df.explode("tags").copy()
    exploded = exploded[exploded["tags"].notna()].copy()
    exploded["tags"] = exploded["tags"].astype(str).str.strip().str.lower()

    # Filter out empty or trivial tags
    exploded = exploded[
        (exploded["tags"] != "") & (~exploded["tags"].isin(ignored_tags))
    ]

    if exploded.empty:
        return pd.DataFrame(columns=output_columns)

    # First, summarize performance per (problem_id, tag)
    # A problem is solved if at least one submission has verdict == 'OK'
    tag_metrics: List[Dict[str, Any]] = []
    grouped_by_tag = exploded.groupby("tags")

    effective_user_rating = max(user_rating, 800)

    for tag, tag_group in grouped_by_tag:
        # Group submissions by problem_id for this tag
        problem_groups = tag_group.groupby("problem_id")
        problems_attempted = len(problem_groups)

        if problems_attempted < min_attempts:
            continue

        unique_solved = 0
        failed_attempts = 0
        solved_problem_ratings: List[float] = []

        for prob_id, p_subs in problem_groups:
            # Check if problem was solved
            is_solved = (p_subs["verdict"] == "OK").any()
            # Non-OK submissions count as failed attempts (whether struggled or never solved)
            failed_count = (p_subs["verdict"] != "OK").sum()
            failed_attempts += int(failed_count)

            if is_solved:
                unique_solved += 1
                # Grab problem rating if available
                ratings = p_subs["rating"].dropna()
                if not ratings.empty:
                    val = ratings.iloc[0]
                    if pd.notna(val) and float(val) > 0:
                        solved_problem_ratings.append(float(val))

        # Calculate average solved rating
        if solved_problem_ratings:
            avg_solved_rating = round(float(np.mean(solved_problem_ratings)), 1)
        else:
            avg_solved_rating = 0.0

        # Calculate solve rate and failure ratio
        solve_rate = round(unique_solved / problems_attempted, 3) if problems_attempted > 0 else 0.0
        failure_ratio = round(failed_attempts / (unique_solved + 1.0), 2)

        # Rating deficit relative to user's rating
        if unique_solved > 0 and avg_solved_rating > 0:
            rating_deficit = max(0.0, float(effective_user_rating - avg_solved_rating))
        else:
            # If user has not solved any problem with this tag, full rating deficit applied
            rating_deficit = float(effective_user_rating)

        # Weakness Score Calculation:
        # 1. Failure-to-solve component: failure_ratio scaled by unsolved problem fraction
        unsolved_fraction = 1.0 - solve_rate
        failure_component = failure_ratio * 10.0 * (0.5 + 0.5 * unsolved_fraction)

        # 2. Rating deficit component: scaled per 50 rating points
        rating_component = (rating_deficit / 50.0) * (0.7 + 0.3 * unsolved_fraction)

        # 3. Volume confidence: prevent a 1-attempt fluke from outranking frequent failures
        confidence = min(1.0, np.log2(problems_attempted + 1.0) / 1.5)

        composite_score = (failure_component + rating_component) * confidence
        weakness_score = round(float(composite_score), 2)

        tag_metrics.append({
            "tag": tag,
            "unique_solved": unique_solved,
            "problems_attempted": problems_attempted,
            "failed_attempts": failed_attempts,
            "avg_solved_rating": avg_solved_rating,
            "failure_ratio": failure_ratio,
            "solve_rate": solve_rate,
            "rating_deficit": round(rating_deficit, 1),
            "weakness_score": weakness_score,
        })

    if not tag_metrics:
        return pd.DataFrame(columns=output_columns)

    df_metrics = pd.DataFrame(tag_metrics)

    # Sort from weakest to strongest (descending weakness_score, then descending failed_attempts)
    df_metrics = df_metrics.sort_values(
        by=["weakness_score", "failed_attempts", "solve_rate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return df_metrics[output_columns]


def get_priority_weak_tags(
    weakness_df: pd.DataFrame,
    top_k: int = 3,
    dag: Optional[Dict[str, List[str]]] = None,
    boost_factor: float = 0.35,
    return_details: bool = False,
) -> Union[List[str], pd.DataFrame]:
    """Identify the Top weak tags with DAG-based prerequisite boosting.

    If a foundational prerequisite tag (e.g. 'graphs', 'dp', 'binary search')
    and one or more of its dependent advanced tags (e.g. 'shortest paths', 'trees',
    'bitmasks') both appear weak, the foundational prerequisite tag receives a priority boost.
    This ensures students master foundational concepts before tackling advanced variants.

    Args:
        weakness_df: Sorted DataFrame from compute_tag_metrics containing 'tag' and 'weakness_score'.
        top_k: Number of prioritized weak tags to return (default: 3).
        dag: Dictionary mapping parent prerequisite -> list of advanced children (defaults to TOPIC_DAG).
        boost_factor: Fractional score increase per weak dependent child (default: 0.35 = +35%).
        return_details: If True, returns a DataFrame showing boosted scores and boost reasons;
            if False, returns list of Top-k tag strings.

    Returns:
        List[str] or pd.DataFrame: Top weak tags ordered by boosted priority.
    """
    if weakness_df is None or weakness_df.empty:
        return pd.DataFrame() if return_details else []

    if dag is None:
        dag = TOPIC_DAG

    # Work on a copy with numeric weakness_score
    df = weakness_df.copy()
    if "weakness_score" not in df.columns or "tag" not in df.columns:
        return pd.DataFrame() if return_details else []

    # Map tags to their base metrics
    tag_score_map = dict(zip(df["tag"], df["weakness_score"]))
    weak_tag_set = set(df["tag"].tolist())

    boosted_scores: Dict[str, float] = {}
    boost_reasons: Dict[str, str] = {}
    weak_children_map: Dict[str, List[str]] = {}

    for tag in df["tag"]:
        base_score = float(tag_score_map.get(tag, 0.0))
        # Check if this tag is a foundational prerequisite in the DAG
        children = dag.get(tag, [])

        # Find which child tags also appear in the user's weak topics
        weak_children = [child for child in children if child in weak_tag_set]

        if weak_children and base_score > 0:
            # Boost foundational parent tag proportionally to number of struggling children
            multiplier = 1.0 + (boost_factor * len(weak_children))
            boosted_score = round(base_score * multiplier, 2)
            reason = (
                f"Prerequisite boost (+{int(boost_factor * 100 * len(weak_children))}%) "
                f"for weak dependent topics: {', '.join(weak_children)}"
            )
            boosted_scores[tag] = boosted_score
            boost_reasons[tag] = reason
            weak_children_map[tag] = weak_children
        else:
            boosted_scores[tag] = base_score
            boost_reasons[tag] = "Base weakness score"
            weak_children_map[tag] = []

    # Attach boosted results
    df["boosted_score"] = df["tag"].map(boosted_scores).fillna(df["weakness_score"])
    df["boost_reason"] = df["tag"].map(boost_reasons).fillna("Base weakness score")
    df["weak_children"] = df["tag"].map(weak_children_map)

    # Sort primarily by boosted_score descending
    df_sorted = df.sort_values(
        by=["boosted_score", "weakness_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top_df = df_sorted.head(top_k).copy()

    if return_details:
        return top_df

    return top_df["tag"].tolist()


def _construct_cf_problem_url(problem_id: str) -> str:
    """Helper to build Codeforces problem URL from problem_id (e.g. 1915G)."""
    pid = str(problem_id).strip()
    match = re.match(r"^(\d+)([A-Za-z0-9_]+)$", pid)
    if match:
        contest_id, index = match.group(1), match.group(2)
        return f"https://codeforces.com/contest/{contest_id}/problem/{index}"
    return f"https://codeforces.com/problemset/problem/{pid}"


def get_stuck_problems(submissions_df: pd.DataFrame) -> pd.DataFrame:
    """Identify all unique problems attempted with zero OK verdicts, sorted by attempt count.

    Args:
        submissions_df: DataFrame of user submissions from cf_api.get_user_submissions.

    Returns:
        pd.DataFrame: Unsolved problems sorted by attempt_count descending:
            - 'problem_id': str (e.g. '1915G')
            - 'name': str
            - 'rating': float/int
            - 'tags': list of str
            - 'attempt_count': int (Total failed submissions on this problem)
            - 'most_frequent_verdict': str (e.g. 'WRONG_ANSWER', 'TIME_LIMIT_EXCEEDED')
            - 'url': str (Direct link to problem on Codeforces)
    """
    columns = [
        "problem_id",
        "name",
        "rating",
        "tags",
        "attempt_count",
        "most_frequent_verdict",
        "url",
    ]

    if submissions_df is None or submissions_df.empty:
        return pd.DataFrame(columns=columns)

    stuck_records: List[Dict[str, Any]] = []
    grouped = submissions_df.groupby("problem_id")

    for pid, group in grouped:
        # Check if user has zero OK verdicts
        has_ok = (group["verdict"] == "OK").any()
        if not has_ok and len(group) >= 1:
            name = str(group["name"].iloc[0]) if "name" in group.columns else ""
            rating_series = group["rating"].dropna() if "rating" in group.columns else pd.Series(dtype=float)
            rating = rating_series.iloc[0] if not rating_series.empty else None

            tags = []
            if "tags" in group.columns:
                for t_list in group["tags"]:
                    if isinstance(t_list, list) and t_list:
                        tags = t_list
                        break

            attempt_count = len(group)
            # Most frequent non-OK verdict
            verdict_counts = group["verdict"].value_counts()
            most_freq_verdict = str(verdict_counts.index[0]) if not verdict_counts.empty else "UNKNOWN"

            url = _construct_cf_problem_url(pid)

            stuck_records.append({
                "problem_id": str(pid),
                "name": name,
                "rating": rating,
                "tags": tags,
                "attempt_count": int(attempt_count),
                "most_frequent_verdict": most_freq_verdict,
                "url": url,
            })

    if not stuck_records:
        return pd.DataFrame(columns=columns)

    df_stuck = pd.DataFrame(stuck_records)
    # Sort primarily by attempt_count descending, then rating descending
    df_stuck["rating_num"] = pd.to_numeric(df_stuck["rating"], errors="coerce").fillna(0)
    df_stuck = df_stuck.sort_values(
        by=["attempt_count", "rating_num"],
        ascending=[False, False],
    ).drop(columns=["rating_num"]).reset_index(drop=True)

    return df_stuck[columns]


def get_verdict_and_wall_stats(
    submissions_df: pd.DataFrame,
    user_rating: int = 800,
) -> Dict[str, Any]:
    """Compute verdict breakdowns for top weak tags and detect the user's Rating Wall.

    1. Tag vs. Verdict breakdown (WRONG_ANSWER, TIME_LIMIT_EXCEEDED, RUNTIME_ERROR, MEMORY_LIMIT_EXCEEDED)
       for the user's top 5 weakest tags.
    2. Rating-wise Attempted vs. Solved counts and solve_rate = solved / attempted to detect
       the Rating Wall (lowest rating with >= 3 attempted where solve_rate < 0.5).

    Args:
        submissions_df: DataFrame of user submissions from cf_api.get_user_submissions.
        user_rating: Current rating of the user.

    Returns:
        Dict[str, Any] containing:
            - 'tag_verdicts_df': DataFrame with verdict breakdown and 1-line diagnoses per weak tag
            - 'rating_stats_df': DataFrame of [rating, attempted, solved, solve_rate]
            - 'rating_wall': int or None (Lowest rating bottleneck where solve_rate < 0.5 and attempts >= 3)
            - 'rating_wall_stats': Dict with wall metrics or None
            - 'automated_diagnoses': List[str] of automated diagnoses
            - 'primary_diagnosis': str (Summary diagnostic insight)
    """
    empty_tag_verdicts = pd.DataFrame(columns=[
        "tag", "WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "RUNTIME_ERROR", "MEMORY_LIMIT_EXCEEDED",
        "total_failures", "dominant_verdict", "diagnosis"
    ])
    empty_rating_stats = pd.DataFrame(columns=["rating", "attempted", "solved", "solve_rate"])

    if submissions_df is None or submissions_df.empty:
        return {
            "tag_verdicts_df": empty_tag_verdicts,
            "rating_stats_df": empty_rating_stats,
            "rating_wall": None,
            "rating_wall_stats": None,
            "automated_diagnoses": [],
            "primary_diagnosis": "No submission history available to diagnose.",
        }

    # 1. Top 5 weakest tags
    weakness_df = compute_tag_metrics(submissions_df, user_rating=user_rating)
    top_5_tags = weakness_df.head(5)["tag"].tolist() if not weakness_df.empty else []

    # Tag vs. Verdict breakdown
    tag_verdict_rows: List[Dict[str, Any]] = []
    diagnoses: List[str] = []

    if top_5_tags:
        # Explode tags
        exploded = submissions_df.explode("tags").copy()
        exploded = exploded[exploded["tags"].notna()].copy()
        exploded["tags"] = exploded["tags"].astype(str).str.strip().str.lower()

        # Only evaluate failed submissions
        failed_exploded = exploded[exploded["verdict"] != "OK"]

        for tag in top_5_tags:
            tag_failures = failed_exploded[failed_exploded["tags"] == tag]
            counts = tag_failures["verdict"].value_counts().to_dict()

            wa_count = int(counts.get("WRONG_ANSWER", 0))
            tle_count = int(counts.get("TIME_LIMIT_EXCEEDED", 0))
            re_count = int(counts.get("RUNTIME_ERROR", 0))
            mle_count = int(counts.get("MEMORY_LIMIT_EXCEEDED", 0))
            total_fail = wa_count + tle_count + re_count + mle_count

            # Determine dominant failure mode and generate automated diagnosis
            diag = ""
            dominant = "None"
            if total_fail > 0:
                verdict_map = {
                    "WRONG_ANSWER": wa_count,
                    "TIME_LIMIT_EXCEEDED": tle_count,
                    "RUNTIME_ERROR": re_count,
                    "MEMORY_LIMIT_EXCEEDED": mle_count,
                }
                dominant = max(verdict_map, key=verdict_map.get)
                if dominant == "TIME_LIMIT_EXCEEDED":
                    diag = f"Mostly TLE on {tag.title()} ({tle_count}/{total_fail}) -> focus on time complexity optimization & faster algorithms"
                elif dominant == "RUNTIME_ERROR":
                    diag = f"Mostly RE on {tag.title()} ({re_count}/{total_fail}) -> check array bounds, null pointers & recursion depth"
                elif dominant == "MEMORY_LIMIT_EXCEEDED":
                    diag = f"Mostly MLE on {tag.title()} ({mle_count}/{total_fail}) -> optimize auxiliary memory & reduce table dimensions"
                else:
                    diag = f"Mostly WA on {tag.title()} ({wa_count}/{total_fail}) -> focus on edge cases, mathematical proofs & corner cases"

                diagnoses.append(diag)
            else:
                diag = f"No recorded failures on {tag.title()}."

            tag_verdict_rows.append({
                "tag": tag,
                "WRONG_ANSWER": wa_count,
                "TIME_LIMIT_EXCEEDED": tle_count,
                "RUNTIME_ERROR": re_count,
                "MEMORY_LIMIT_EXCEEDED": mle_count,
                "total_failures": total_fail,
                "dominant_verdict": dominant,
                "diagnosis": diag,
            })

    tag_verdicts_df = pd.DataFrame(tag_verdict_rows) if tag_verdict_rows else empty_tag_verdicts

    # 2. Rating-wise Attempted vs. Solved counts
    # Group distinct problems by rating
    df_valid = submissions_df.copy()
    df_valid["rating"] = pd.to_numeric(df_valid["rating"], errors="coerce")
    df_valid = df_valid[df_valid["rating"].notna() & (df_valid["rating"] > 0)]

    rating_records = []
    if not df_valid.empty:
        # Group by problem_id first to determine if each problem was ever solved
        problem_summary = df_valid.groupby("problem_id").agg({
            "rating": "first",
            "verdict": lambda v: (v == "OK").any(),
        }).rename(columns={"verdict": "is_solved"}).reset_index()

        problem_summary["rating"] = problem_summary["rating"].astype(int)

        for rating, r_group in problem_summary.groupby("rating"):
            attempted = len(r_group)
            solved = int(r_group["is_solved"].sum())
            solve_rate = round(solved / attempted, 3) if attempted > 0 else 0.0

            rating_records.append({
                "rating": int(rating),
                "attempted": int(attempted),
                "solved": int(solved),
                "solve_rate": float(solve_rate),
            })

    rating_stats_df = pd.DataFrame(rating_records) if rating_records else empty_rating_stats
    if not rating_stats_df.empty:
        rating_stats_df = rating_stats_df.sort_values(by="rating").reset_index(drop=True)

    # 3. Detect Rating Wall:
    # Lowest rating where solve_rate < 0.5 with at least 3 attempted problems
    rating_wall = None
    rating_wall_stats = None

    if not rating_stats_df.empty:
        bottlenecks = rating_stats_df[
            (rating_stats_df["attempted"] >= 3) & (rating_stats_df["solve_rate"] < 0.5)
        ]
        if not bottlenecks.empty:
            first_wall = bottlenecks.iloc[0]
            rating_wall = int(first_wall["rating"])
            rating_wall_stats = {
                "rating": rating_wall,
                "attempted": int(first_wall["attempted"]),
                "solved": int(first_wall["solved"]),
                "solve_rate": float(first_wall["solve_rate"]),
            }

    # Primary diagnostic summary
    if rating_wall_stats:
        primary_diag = (
            f"Your bottleneck starts at {rating_wall} rating — "
            f"{int(rating_wall_stats['solve_rate'] * 100)}% solve rate "
            f"({rating_wall_stats['solved']}/{rating_wall_stats['attempted']} problems solved)."
        )
    elif not rating_stats_df.empty:
        primary_diag = "Consistent solid solve rate across attempted rating brackets; no premature rating wall detected."
    else:
        primary_diag = "Not enough rated problem data to identify a specific rating wall."

    return {
        "tag_verdicts_df": tag_verdicts_df,
        "rating_stats_df": rating_stats_df,
        "rating_wall": rating_wall,
        "rating_wall_stats": rating_wall_stats,
        "automated_diagnoses": diagnoses,
        "primary_diagnosis": primary_diag,
    }

