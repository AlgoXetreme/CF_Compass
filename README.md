# 🎯 Codeforces Weak-Topic Recommender

A modular Python and Streamlit application that analyzes a competitive programmer's submission history on [Codeforces](https://codeforces.com), identifies their weakest algorithmic topics, and recommends targeted unsolved problems within their rating range to practice and improve.

---

## 📁 Project Structure

```text
codeforces-weak-topic-recommender/
│
├── cf_api.py                 # Core Codeforces API client module (user info, submissions, problemset cache)
├── analyzer.py               # Weak-topic analyzer & DAG prerequisite engine
├── recommender.py            # Practice problem recommendation logic
├── app.py                    # Interactive Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation & usage guide
├── .gitignore                # Git ignore rules for Python & data caches
│
├── data/
│   ├── .gitkeep              # Data folder placeholder
│   └── problems_cache.json   # Auto-generated 24-hour JSON cache of Codeforces problemset
│
└── tests/
    ├── __init__.py
    ├── test_cf_api.py        # Unit tests for cf_api functions with mocks
    ├── test_analyzer.py      # Unit tests for analyzer & DAG prerequisite engine
    └── test_recommender.py   # Unit tests for recommendation ranking
```

---

## 🚀 Quickstart

### 1. Installation

Clone or open the project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, enter any Codeforces handle (e.g. `tourist`, `Benq`, or your own username), and view:
- **Profile Summary**: Current rating, peak rating, rank, and avatar (handles unrated users gracefully by defaulting rating to 800).
- **Submissions Analytics**: Donut chart of verdicts and practice frequency by topic tag using Plotly.
- **Weak Topics Detection**: Ranked list of topics with the lowest solve rates and highest failure counts.
- **Personalized Recommendations**: Unsolved rated problems tailored to your current rating with direct Codeforces problem links.

---

## 🛠️ API Reference: `cf_api.py`

The core API wrapper `cf_api.py` includes robust error handling, network timeouts, and JSON caching.

### 1. `get_user_info(handle: str) -> dict`
Fetches user profile information from Codeforces.
- **Endpoint**: `https://codeforces.com/api/user.info?handles={handle}`
- **Unrated Handling**: Automatically defaults unrated users to a rating of **800** and rank to `"unrated"`.
- **Returns**:
  ```python
  {
      "handle": "tourist",
      "rating": 3979,
      "maxRating": 4009,
      "rank": "legendary grandmaster",
      "avatar": "https://userpic.codeforces.org/..."
  }
  ```

### 2. `get_user_submissions(handle: str, count: Optional[int] = None) -> pd.DataFrame`
Fetches a user's full submission history.
- **Endpoint**: `https://codeforces.com/api/user.status?handle={handle}`
- **Returns**: A structured `pandas.DataFrame` with the following columns:
  - `problem_id`: Formatted problem identifier (e.g. `"1915G"`).
  - `name`: Problem name.
  - `rating`: Difficulty rating (or `NaN` if unrated).
  - `tags`: Python list of strings containing topic tags (e.g. `["dijkstra", "graphs"]`).
  - `verdict`: Submission verdict (e.g. `"OK"`, `"WRONG_ANSWER"`, `"TIME_LIMIT_EXCEEDED"`).
  - `creationTimeSeconds`: Integer epoch timestamp.

### 3. `get_problemset(cache_path="data/problems_cache.json", force_refresh=False) -> pd.DataFrame`
Fetches and caches the full Codeforces problemset.
- **Endpoint**: `https://codeforces.com/api/problemset.problems`
- **Cache Strategy**:
  - Checks if `cache_path` exists and was modified within the last **24 hours**.
  - If valid, loads directly from disk to prevent hitting API rate limits.
  - If missing or older than 24 hours, fetches fresh data from Codeforces, merges problem stats (`solvedCount` from `problemStatistics`), writes to JSON, and returns.
- **Returns**: A clean `pandas.DataFrame` filtered to **rated problems only**, containing:
  - `problem_id`: e.g. `"1915G"`
  - `contestId`: Contest ID integer
  - `index`: Problem index letter (e.g. `"G"`)
  - `name`: Problem title
  - `rating`: Integer rating
  - `tags`: List of topic tags
  - `solvedCount`: Total number of accepted submissions
  - `url`: Direct link to the problem (`https://codeforces.com/contest/{contestId}/problem/{index}`)

### 4. `get_upcoming_contests(gym=False) -> List[dict]`
Fetches and formats all upcoming Codeforces contests scheduled in the future (`phase == "BEFORE"`).
- **Endpoint**: `https://codeforces.com/api/contest.list?gym=false`
- **Sorting**: Ordered strictly ascending by `startTimeSeconds` (soonest first).
- **Enriched Metadata**:
  - `division`: Parsed canonical division (`"Div. 4"`, `"Div. 3"`, `"Div. 2"`, `"Div. 1"`, `"Div. 1 + Div. 2"`, `"Educational"`, or `"Other"`).
  - `start_time_ist`: Localized date and time string in Indian Standard Time (IST) (e.g. `"22 Nov 2026, 08:05 PM IST"`).
  - `countdown`: Human-readable live countdown string (e.g. `"1d 4h 30m"`).
  - `duration_formatted`: Human-readable contest length (e.g. `"2 hours"`, `"2h 15m"`).
  - `url`: Direct contest registration link (`https://codeforces.com/contests/{id}`).
  - `gcal_url`: 1-click Google Calendar reminder creation URL formatted with UTC timestamps.

---

## 🧠 Algorithmic Analyzer & DAG Engine: `analyzer.py`

The `analyzer.py` module evaluates topic-level weaknesses and structures learning paths through topic dependency graphs:

### 1. `compute_tag_metrics(submissions_df, user_rating) -> pd.DataFrame`
Groups all submissions by tag, filtering out trivial tags (`*special`, `implementation`). Calculates:
- `unique_solved`: Number of distinct problems solved (`verdict == 'OK'`).
- `problems_attempted`: Unique problems attempted.
- `failed_attempts`: Total non-OK submissions on problems never solved or struggled on.
- `avg_solved_rating`: Average difficulty of problems solved in this tag.
- `failure_ratio`: $\frac{\text{failed\_attempts}}{\text{unique\_solved} + 1}$.
- `rating_deficit`: $\max(0, \text{user\_rating} - \text{avg\_solved\_rating})$.
- `weakness_score`: Composite score factoring failure intensity, rating deficit, and sample confidence. Returns tags sorted from weakest to strongest.

### 2. `TOPIC_DAG` Dictionary
A Directed Acyclic Graph mapping foundational prerequisite topics to their advanced variants:
```python
TOPIC_DAG = {
    "dp": ["trees", "bitmasks", "graphs", "probabilities"],
    "binary search": ["greedy", "two pointers", "ternary search"],
    "graphs": ["shortest paths", "dfs and similar", "dsu", "flows", "graph matchings"],
    "math": ["number theory", "combinatorics", "matrices", "geometry"],
    "data structures": ["dsu", "trees", "bitmasks"],
    "strings": ["string suffix structures", "expression parsing"],
}
```

### 3. `get_priority_weak_tags(weakness_df, top_k=3, boost_factor=0.35) -> List[str]`
Returns the Top 3 weak tags. If both a foundational prerequisite tag (e.g. `graphs`) and any of its child tags (e.g. `shortest paths`, `dfs and similar`) appear weak, the prerequisite tag receives a priority boost (+35% per weak child) to prioritize foundational learning before advanced problem sets.

### 4. `get_stuck_problems(submissions_df) -> pd.DataFrame`
Finds all unique attempted problems with zero `OK` verdicts, sorted by `attempt_count` descending. Includes problem name, rating, tags, attempt count, most frequent verdict (e.g. `WRONG_ANSWER`, `TIME_LIMIT_EXCEEDED`), and direct Codeforces problem URLs.

### 5. `get_verdict_and_wall_stats(submissions_df, user_rating=800) -> dict`
- **Tag vs. Verdict Breakdown**: Aggregates `WRONG_ANSWER`, `TIME_LIMIT_EXCEEDED`, `RUNTIME_ERROR`, and `MEMORY_LIMIT_EXCEEDED` across the Top 5 weakest tags with automated 1-line diagnoses.
- **Rating Wall Detection**: Evaluates attempted vs. solved problems per difficulty tier to identify the exact **Rating Wall** (lowest rating with $\ge 3$ attempted problems where $\text{solve\_rate} < 0.5$).

---

## 🪜 TF-IDF + Cosine Similarity Upsolving Ladder: `recommender.py`

The `recommender.py` module uses `scikit-learn` to formulate progressive practice sets:

### `generate_upsolving_ladder(problemset_df, solved_ids, weak_tags_df, user_rating) -> List[dict]`
1. **Filtering**: Excludes already solved problems and problems without a valid difficulty rating.
2. **Tag Preprocessing**: Converts multi-word tags into unified underscore-separated tokens (e.g., `"binary search"` $\rightarrow$ `"binary_search"`, `"shortest paths"` $\rightarrow$ `"shortest_paths"`).
3. **TF-IDF Vectorization**: Fits `TfidfVectorizer(token_pattern=r"\S+")` across the candidate problem tag corpus.
4. **Weighted Query Vector**: Constructs a query string from the user's **Top 4 weak tags**, repeating tokens proportionally to their weakness/boosted score.
5. **Cosine Similarity & Popularity Boost**: Computes tag cosine similarity, combined with a normalized $\log(1 + \text{solvedCount})$ quality boost.
6. **5-Problem Progressive Ladder**: Selects 5 curated problems across rating brackets:
   $$[R, R+100, R+100, R+200, R+300]$$
   (where $R$ is user rating rounded to the nearest 100, minimum 800), enforcing sub-topic variety so the ladder covers diverse weak areas.

### `recommend_cp31(cp31_df, problemset_df, solved_ids, weak_tags_df, user_rating) -> pd.DataFrame`
Joins curated `cp31_df` with `problemset_df` on `problem_id`, filters out problems in `solved_ids`, matches against the user's top weak tags within rating window $[R, R+300]$, and returns the Top 5 unsolved CP-31 Sheet problems.

### `recommend_cses(cses_df, weak_tags_df, completed_cses_ids) -> pd.DataFrame`
Maps the user's Top 3 Codeforces weak tags to CSES categories via `CF_TO_CSES_MAP`, filters out completed tasks stored in `data/cses_progress.json`, and returns 5 foundational CSES problems ordered from **Easy to Hard**.

### `get_cp31_unique_tags(cp31_df, problemset_df) -> List[str]`
Extracts and returns an alphabetically sorted list of unique algorithmic tags present across the merged CP-31 dataset.

### `get_cp31_topic_ladder(cp31_df, problemset_df, submissions_df, selected_tag, min_rating=800, max_rating=1900) -> Tuple[pd.DataFrame, dict]`
Builds a complete topic practice ladder for any chosen algorithmic tag from CP-31:
- **Filtering**: Matches `selected_tag` in problem tags and restricts difficulties to `[min_rating, max_rating]`.
- **Status Cross-Check**: Cross-references `submissions_df` to assign:
  - `✓ Solved`: User has at least one accepted (`OK`) verdict.
  - `⚠️ Stuck`: User attempted the problem ($\ge 1$ submission) but has 0 `OK` verdicts.
  - `○ Todo`: Unattempted by the user.
- **Strict Ordering**: Sorts problems strictly by `rating` ascending (and `solvedCount` descending as a popularity tie-breaker).
- **Summary Metrics**: Computes `total_count`, `solved_count`, `stuck_count`, `todo_count`, and `solve_rate` for dynamic progress tracking in the UI.

---

## 🖥️ Streamlit Interactive Dashboard: `app.py`

The dashboard integrates all analytics and recommendation engines into five tabs:
1. **🪜 Upsolving Ladder**: 5-problem progressive TF-IDF ladder with match percentage, difficulty badges, and DAG prerequisite hierarchy expander.
2. **📚 Curated Sheets (CP-31 & CSES)**:
   - **CP-31 Recommendations**: Unsolved CP-31 problems tailored to weak tags within $[R, R+300]$.
   - **CSES Foundations**: Foundational tasks ordered Easy to Hard with persistent completion tracking.
   - **📈 CP-31 Topic Ladder**: Interactive tag selector (defaults to user's #1 weak tag), rating range slider (800–1900), "Hide Already Solved" toggle, completion progress bar, and collapsible rating-grouped cards with direct Codeforces solve/retry links.
3. **🔍 Stuck Point Diagnostics**:
   - **Rating Wall**: Detected wall rating callout + Attempted vs. Solved grouped bar chart.
   - **Failure Modes**: WA, TLE, RE, MLE breakdown stacked chart with actionable diagnoses.
   - **Abandoned Problems Table**: Top 5 most-attempted unsolved problems with retry links.
4. **📊 Topic Analytics**: Horizontal weak tag ranking chart + Solved problems by rating comfort zone chart.
5. **🏆 Upcoming Contests (All Divisions)**:
   - **Compact Sidebar Banner**: Highlights the very next upcoming contest with countdown, division pill, IST time, and quick registration & calendar buttons.
   - **Division Multiselect**: Quick filtering across `Div. 4`, `Div. 3`, `Div. 2`, `Div. 1`, `Div. 1 + Div. 2`, `Educational`, and `Other`.
   - **"Best for Your Rating" Smart Badging**: Automatically identifies rounds best matched to user rating (< 1400 $\rightarrow$ Div. 4/3/2; 1400–1899 $\rightarrow$ Div. 3/2/Educational; 1900+ $\rightarrow$ Div. 1/2).
   - **1-Click Actions**: Direct Codeforces registration link and pre-filled Google Calendar event reminder links.

---

## 🧪 Running Unit Tests

Run the test suite using Python's built-in `unittest` runner:

```bash
python -m unittest discover tests -v
```

All network calls are mocked with simulated Codeforces responses, testing edge cases such as unrated users, non-existent handles, timeouts, and cache invalidation.

---

## ☁️ Deploying to Render

This repository is pre-configured with `render.yaml`, `Procfile`, and `.streamlit/config.toml` for seamless 1-click cloud deployment on [Render](https://render.com).

### Method 1: Automatic Blueprint (Recommended)
1. Push this repository to your GitHub or GitLab account:
   ```bash
   git remote add origin https://github.com/<your-username>/codeforces-weak-topic-recommender.git
   git branch -M main
   git push -u origin main
   ```
2. In the Render Dashboard, click **New +** $\rightarrow$ **Blueprint**.
3. Connect your repository. Render will automatically read `render.yaml` and configure the service.
4. Click **Apply**. Render will install dependencies and start your live dashboard.

### Method 2: Manual Web Service
1. In the Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
2. Select your connected repository.
3. Configure the settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Instance Type**: `Free`
4. Click **Create Web Service**. Your live URL (e.g. `https://codeforces-recommender.onrender.com`) will be generated.
