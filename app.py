import io
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

EXCEL_URL = "https://www.football-data.co.uk/mmz4281/2526/all-euro-data-2025-2026.xlsx"


# =========================
# DATA LOADING & AUTOMATION
# =========================

@st.cache_data(ttl=6 * 60 * 60)  # cache for 6 hours
def download_and_build_dataset(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    excel_bytes = resp.content

    sheets = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
    frames = []

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue

        df = df.copy()

        # Keep track of original sheet
        df["SourceSheet"] = sheet_name

        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]

        frames.append(df)

    if not frames:
        raise ValueError("No non-empty sheets found in Excel file")

    data = pd.concat(frames, ignore_index=True)

    # Try to create a unified "League" column
    if "Div" in data.columns:
        data["League"] = data["Div"]
    elif "League" in data.columns:
        data["League"] = data["League"]
    else:
        data["League"] = data["SourceSheet"]

    # Ensure date column if present
    date_col = None
    for candidate in ["Date", "date", "MatchDate"]:
        if candidate in data.columns:
            date_col = candidate
            break

    if date_col is not None:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data.rename(columns={date_col: "Date"}, inplace=True)
    else:
        data["Date"] = pd.NaT

    # Standard column aliases (fallback if needed)
    # Football-data standard: HomeTeam, AwayTeam, FTHG, FTAG, FTR
    # If the file uses different labels, adapt here.
    rename_map = {}

    # Example fallback (adjust if necessary)
    if "Home" in data.columns and "HomeTeam" not in data.columns:
        rename_map["Home"] = "HomeTeam"
    if "Away" in data.columns and "AwayTeam" not in data.columns:
        rename_map["Away"] = "AwayTeam"

    if rename_map:
        data.rename(columns=rename_map, inplace=True)

    return data


def compute_league_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-league stats: matches, goals, win/draw/lose %, etc.
    Requires columns: League, HomeTeam, AwayTeam, FTHG, FTAG, FTR.
    """

    required_cols = {"League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    # Infer FTR if missing
    if "FTR" not in df.columns:
        def infer_result(row):
            if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
                return None
            if row["FTHG"] > row["FTAG"]:
                return "H"
            elif row["FTHG"] < row["FTAG"]:
                return "A"
            else:
                return "D"
        df = df.copy()
        df["FTR"] = df.apply(infer_result, axis=1)

    # Work on numeric goals only
    tmp = df.dropna(subset=["FTHG", "FTAG"]).copy()
    tmp["FTHG"] = pd.to_numeric(tmp["FTHG"], errors="coerce")
    tmp["FTAG"] = pd.to_numeric(tmp["FTAG"], errors="coerce")

    tmp["TotalGoals"] = tmp["FTHG"] + tmp["FTAG"]
    tmp["HomeWin"] = (tmp["FTR"] == "H").astype(int)
    tmp["Draw"] = (tmp["FTR"] == "D").astype(int)
    tmp["AwayWin"] = (tmp["FTR"] == "A").astype(int)
    tmp["BTTS"] = ((tmp["FTHG"] > 0) & (tmp["FTAG"] > 0)).astype(int)
    tmp["Over2_5"] = (tmp["TotalGoals"] > 2.5).astype(int)

    grouped = tmp.groupby("League").agg(
        Matches=("League", "size"),
        AvgGoals=("TotalGoals", "mean"),
        AvgHomeGoals=("FTHG", "mean"),
        AvgAwayGoals=("FTAG", "mean"),
        HomeWinPct=("HomeWin", "mean"),
        DrawPct=("Draw", "mean"),
        AwayWinPct=("AwayWin", "mean"),
        BTTS_Pct=("BTTS", "mean"),
        Over2_5_Pct=("Over2_5", "mean")
    )

    # Convert ratios to percentages
    for col in ["HomeWinPct", "DrawPct", "AwayWinPct", "BTTS_Pct", "Over2_5_Pct"]:
        grouped[col] = grouped[col] * 100

    return grouped.reset_index()


def compute_team_table_for_league(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """
    Standings-like table for a given league.
    """

    sub = df[df["League"] == league].copy()

    required_cols = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required_cols.issubset(sub.columns):
        return pd.DataFrame()

    # Ensure FTR
    if "FTR" not in sub.columns:
        def infer_result(row):
            if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
                return None
            if row["FTHG"] > row["FTAG"]:
                return "H"
            elif row["FTHG"] < row["FTAG"]:
                return "A"
            else:
                return "D"
        sub["FTR"] = sub.apply(infer_result, axis=1)

    sub["FTHG"] = pd.to_numeric(sub["FTHG"], errors="coerce")
    sub["FTAG"] = pd.to_numeric(sub["FTAG"], errors="coerce")

    teams = pd.unique(sub[["HomeTeam", "AwayTeam"]].values.ravel("K"))
    teams = [t for t in teams if pd.notna(t)]

    records = []

    for team in teams:
        # Home matches
        home = sub[sub["HomeTeam"] == team]
        # Away matches
        away = sub[sub["AwayTeam"] == team]

        played = len(home) + len(away)

        if played == 0:
            continue

        # Goals for/against
        gf = home["FTHG"].sum(skipna=True) + away["FTAG"].sum(skipna=True)
        ga = home["FTAG"].sum(skipna=True) + away["FTHG"].sum(skipna=True)

        # Results
        w = (
            (home["FTR"] == "H").sum()
            + (away["FTR"] == "A").sum()
        )
        d = (
            (home["FTR"] == "D").sum()
            + (away["FTR"] == "D").sum()
        )
        l = (
            (home["FTR"] == "A").sum()
            + (away["FTR"] == "H").sum()
        )

        pts = 3 * w + d
        gd = gf - ga
        ppg = pts / played if played > 0 else 0

        records.append(
            {
                "Team": team,
                "Played": played,
                "W": w,
                "D": d,
                "L": l,
                "GF": gf,
                "GA": ga,
                "GD": gd,
                "Pts": pts,
                "PPG": round(ppg, 3),
            }
        )

    table = pd.DataFrame(records)
    if table.empty:
        return table

    table.sort_values(
        by=["Pts", "GD", "GF"],
        ascending=[False, False, False],
        inplace=True
    )
    table.reset_index(drop=True, inplace=True)
    table.index += 1  # 1-based ranking

    return table


def compute_team_stats(df: pd.DataFrame, league: str, team: str) -> dict:
    sub = df[df["League"] == league].copy()
    if sub.empty:
        return {}

    if "FTR" not in sub.columns:
        def infer_result(row):
            if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
                return None
            if row["FTHG"] > row["FTAG"]:
                return "H"
            elif row["FTHG"] < row["FTAG"]:
                return "A"
            else:
                return "D"
        sub["FTR"] = sub.apply(infer_result, axis=1)

    sub["FTHG"] = pd.to_numeric(sub["FTHG"], errors="coerce")
    sub["FTAG"] = pd.to_numeric(sub["FTAG"], errors="coerce")

    home = sub[sub["HomeTeam"] == team]
    away = sub[sub["AwayTeam"] == team]
    all_matches = pd.concat([home, away], ignore_index=True)
    all_matches = all_matches.dropna(subset=["FTHG", "FTAG"])

    if all_matches.empty:
        return {}

    # Overall stats
    played = len(all_matches)
    gf = home["FTHG"].sum() + away["FTAG"].sum()
    ga = home["FTAG"].sum() + away["FTHG"].sum()
    gd = gf - ga

    wins = (home["FTR"] == "H").sum() + (away["FTR"] == "A").sum()
    draws = (home["FTR"] == "D").sum() + (away["FTR"] == "D").sum()
    losses = (home["FTR"] == "A").sum() + (away["FTR"] == "H").sum()

    pts = 3 * wins + draws
    ppg = pts / played if played > 0 else 0

    all_matches["TotalGoals"] = all_matches["FTHG"] + all_matches["FTAG"]
    avg_total_goals = all_matches["TotalGoals"].mean()

    # Simple form (last 5)
    def result_symbol(row):
        if row["HomeTeam"] == team:
            if row["FTR"] == "H":
                return "W"
            elif row["FTR"] == "D":
                return "D"
            else:
                return "L"
        else:
            if row["FTR"] == "A":
                return "W"
            elif row["FTR"] == "D":
                return "D"
            else:
                return "L"

    all_matches_sorted = all_matches.sort_values("Date", ascending=False)
    all_matches_sorted["ResSymbol"] = all_matches_sorted.apply(result_symbol, axis=1)
    form = "".join(all_matches_sorted["ResSymbol"].head(5).tolist())

    return {
        "Played": played,
        "Wins": wins,
        "Draws": draws,
        "Losses": losses,
        "GF": gf,
        "GA": ga,
        "GD": gd,
        "Points": pts,
        "PPG": round(ppg, 3),
        "AvgTotalGoals": round(avg_total_goals, 3),
        "FormLast5": form,
    }


# ============
# STREAMLIT UI
# ============

def main():
    st.set_page_config(
        page_title="All Euro 2025-26 Explorer",
        layout="wide",
    )

    st.title("All Euro 2025–2026 – League & Team Stats")

    with st.sidebar:
        st.header("Data source")
        st.write("Football-Data all-euro Excel 2025–2026")
        st.markdown(f"[Open source file]({EXCEL_URL})")

        refresh = st.button("Force refresh data (ignore cache)")

    if refresh:
        download_and_build_dataset.clear()
        st.experimental_rerun()

    try:
        data = download_and_build_dataset(EXCEL_URL)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Quick info
    leagues = sorted(data["League"].dropna().unique())
    n_matches = len(data)
    n_leagues = len(leagues)

    date_min = data["Date"].min()
    date_max = data["Date"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total matches", f"{n_matches:,}")
    col2.metric("Leagues", n_leagues)

    date_from_display = date_min.date().isoformat() if pd.notna(date_min) else "N/A"
    date_to_display = date_max.date().isoformat() if pd.notna(date_max) else "N/A"

    col3.metric("Date from", date_from_display)
    col4.metric("Date to", date_to_display)

    st.markdown("---")

    # Tabs: Leagues overview / League details (teams & table)
    tab_overview, tab_league = st.tabs(["Leagues overview", "League & teams"])

    with tab_overview:
        st.subheader("Leagues summary")
        league_stats = compute_league_stats(data)
        if league_stats.empty:
            st.warning("Required columns for league stats not found.")
        else:
            # Sort by avg goals by default
            league_stats_sorted = league_stats.sort_values("AvgGoals", ascending=False)
            st.dataframe(
                league_stats_sorted,
                use_container_width=True,
                height=500,
            )

    with tab_league:
        st.subheader("League and team details")

        if not leagues:
            st.warning("No leagues detected.")
            return

        sel_league = st.selectbox("Select league", leagues, index=0)

        league_df = data[data["League"] == sel_league].copy()
        if league_df.empty:
            st.warning("No data for selected league.")
            return

        # Standings-like table
        st.markdown(f"### Standings – {sel_league}")
        league_table = compute_team_table_for_league(data, sel_league)
        if league_table.empty:
            st.warning("Cannot build standings table for this league.")
        else:
            st.dataframe(
                league_table,
                use_container_width=True,
                height=400,
            )

        # Team stats
        teams_in_league = sorted(
            pd.unique(league_df[["HomeTeam", "AwayTeam"]].values.ravel("K"))
        )
        teams_in_league = [t for t in teams_in_league if pd.notna(t)]

        st.markdown(f"### Team stats – {sel_league}")
        if not teams_in_league:
            st.warning("No teams detected for this league.")
        else:
            sel_team = st.selectbox("Select team", teams_in_league, index=0)
            tstats = compute_team_stats(data, sel_league, sel_team)

            if not tstats:
                st.warning("No stats available for selected team.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Played", tstats["Played"])
                c2.metric("Points", tstats["Points"])
                c3.metric("PPG", tstats["PPG"])
                c4.metric("Goal diff", tstats["GD"])

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Wins", tstats["Wins"])
                c6.metric("Draws", tstats["Draws"])
                c7.metric("Losses", tstats["Losses"])
                c8.metric("Avg total goals (matches)", tstats["AvgTotalGoals"])

                st.markdown(f"**Last 5 results:** {tstats['FormLast5']} (W/D/L sequence)")

                # Show raw match list for this team
                st.markdown("#### Matches for selected team")
                team_matches = league_df[
                    (league_df["HomeTeam"] == sel_team)
                    | (league_df["AwayTeam"] == sel_team)
                ].copy()

                # Sort by date desc if available
                if "Date" in team_matches.columns:
                    team_matches = team_matches.sort_values("Date", ascending=False)

                cols_to_show = []
                for col in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]:
                    if col in team_matches.columns:
                        cols_to_show.append(col)
                # plus odds if present
                for col in ["B365H", "B365D", "B365A"]:
                    if col in team_matches.columns:
                        cols_to_show.append(col)

                if cols_to_show:
                    st.dataframe(
                        team_matches[cols_to_show],
                        use_container_width=True,
                        height=400,
                    )
                else:
                    st.dataframe(
                        team_matches,
                        use_container_width=True,
                        height=400,
                    )


if __name__ == "__main__":
    main()
