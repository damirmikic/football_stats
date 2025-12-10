import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

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
        df["SourceSheet"] = sheet_name
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)

    if not frames:
        raise ValueError("No non-empty sheets found in Excel file")

    data = pd.concat(frames, ignore_index=True)

    # League column
    if "Div" in data.columns:
        data["League"] = data["Div"]
    elif "League" in data.columns:
        data["League"] = data["League"]
    else:
        data["League"] = data["SourceSheet"]

    # Date column
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

    # Possible fallback renames
    rename_map = {}
    if "Home" in data.columns and "HomeTeam" not in data.columns:
        rename_map["Home"] = "HomeTeam"
    if "Away" in data.columns and "AwayTeam" not in data.columns:
        rename_map["Away"] = "AwayTeam"

    if rename_map:
        data.rename(columns=rename_map, inplace=True)

    return data


# =====================
# LEAGUE-LEVEL METRICS
# =====================

def compute_league_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-league stats including:
    - Avg goals, win/draw/lose %
    - BTTS, Over 2.5/3.5/4.5
    - 1st half Over 1.5, 1st half BTTS, goals in both halves
    """

    required_cols = {"League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()

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
        df["FTR"] = df.apply(infer_result, axis=1)

    tmp = df.dropna(subset=["FTHG", "FTAG"]).copy()
    tmp["FTHG"] = pd.to_numeric(tmp["FTHG"], errors="coerce")
    tmp["FTAG"] = pd.to_numeric(tmp["FTAG"], errors="coerce")

    tmp["TotalGoals"] = tmp["FTHG"] + tmp["FTAG"]
    tmp["HomeWin"] = (tmp["FTR"] == "H").astype(int)
    tmp["Draw"] = (tmp["FTR"] == "D").astype(int)
    tmp["AwayWin"] = (tmp["FTR"] == "A").astype(int)
    tmp["BTTS"] = ((tmp["FTHG"] > 0) & (tmp["FTAG"] > 0)).astype(int)
    tmp["Over2_5"] = (tmp["TotalGoals"] > 2.5).astype(int)
    tmp["Over3_5"] = (tmp["TotalGoals"] > 3.5).astype(int)
    tmp["Over4_5"] = (tmp["TotalGoals"] > 4.5).astype(int)

    has_ht = {"HTHG", "HTAG"}.issubset(tmp.columns)

    if has_ht:
        tmp["HTHG"] = pd.to_numeric(tmp["HTHG"], errors="coerce")
        tmp["HTAG"] = pd.to_numeric(tmp["HTAG"], errors="coerce")

        tmp["HTGoals"] = tmp["HTHG"] + tmp["HTAG"]
        tmp["FH_Over15"] = (tmp["HTGoals"] > 1.5).astype(int)
        tmp["FH_BTTS"] = ((tmp["HTHG"] > 0) & (tmp["HTAG"] > 0)).astype(int)

        tmp["SHGoals"] = tmp["TotalGoals"] - tmp["HTGoals"]
        tmp["GoalsBothHalves"] = (
            (tmp["HTGoals"] > 0) & (tmp["SHGoals"] > 0)
        ).astype(int)
    else:
        tmp["FH_Over15"] = pd.NA
        tmp["FH_BTTS"] = pd.NA
        tmp["GoalsBothHalves"] = pd.NA

    grouped = tmp.groupby("League").agg(
        Matches=("League", "size"),
        AvgGoals=("TotalGoals", "mean"),
        AvgHomeGoals=("FTHG", "mean"),
        AvgAwayGoals=("FTAG", "mean"),
        HomeWinPct=("HomeWin", "mean"),
        DrawPct=("Draw", "mean"),
        AwayWinPct=("AwayWin", "mean"),
        BTTS_Pct=("BTTS", "mean"),
        Over2_5_Pct=("Over2_5", "mean"),
        Over3_5_Pct=("Over3_5", "mean"),
        Over4_5_Pct=("Over4_5", "mean"),
        FH_Over15_Pct=("FH_Over15", "mean"),
        FH_BTTS_Pct=("FH_BTTS", "mean"),
        GoalsBothHalves_Pct=("GoalsBothHalves", "mean"),
    )

    pct_cols = [
        "HomeWinPct",
        "DrawPct",
        "AwayWinPct",
        "BTTS_Pct",
        "Over2_5_Pct",
        "Over3_5_Pct",
        "Over4_5_Pct",
        "FH_Over15_Pct",
        "FH_BTTS_Pct",
        "GoalsBothHalves_Pct",
    ]

    for col in pct_cols:
        grouped[col] = grouped[col] * 100

    return grouped.reset_index()


# =====================
# TEAM TABLE / STANDINGS
# =====================

def compute_team_table_for_league(df: pd.DataFrame, league: str) -> pd.DataFrame:
    sub = df[df["League"] == league].copy()

    required_cols = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required_cols.issubset(sub.columns):
        return pd.DataFrame()

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
        home = sub[sub["HomeTeam"] == team]
        away = sub[sub["AwayTeam"] == team]

        played = len(home) + len(away)
        if played == 0:
            continue

        gf = home["FTHG"].sum(skipna=True) + away["FTAG"].sum(skipna=True)
        ga = home["FTAG"].sum(skipna=True) + away["FTHG"].sum(skipna=True)

        w = ((home["FTR"] == "H").sum() + (away["FTR"] == "A").sum())
        d = ((home["FTR"] == "D").sum() + (away["FTR"] == "D").sum())
        l = ((home["FTR"] == "A").sum() + (away["FTR"] == "H").sum())

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
        inplace=True,
    )
    table.reset_index(drop=True, inplace=True)
    table.index += 1

    return table


# =====================
# TEAM-LEVEL METRICS
# =====================

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
    all_matches["BTTS"] = ((all_matches["FTHG"] > 0) & (all_matches["FTAG"] > 0)).astype(int)
    all_matches["O2_5"] = (all_matches["TotalGoals"] > 2.5).astype(int)
    all_matches["O3_5"] = (all_matches["TotalGoals"] > 3.5).astype(int)
    all_matches["O4_5"] = (all_matches["TotalGoals"] > 4.5).astype(int)

    avg_total_goals = all_matches["TotalGoals"].mean()
    btts_pct = all_matches["BTTS"].mean() * 100
    o25_pct = all_matches["O2_5"].mean() * 100
    o35_pct = all_matches["O3_5"].mean() * 100
    o45_pct = all_matches["O4_5"].mean() * 100

    fh_over15_pct = None
    fh_btts_pct = None
    goals_both_halves_pct = None

    if {"HTHG", "HTAG"}.issubset(all_matches.columns):
        all_matches["HTHG"] = pd.to_numeric(all_matches["HTHG"], errors="coerce")
        all_matches["HTAG"] = pd.to_numeric(all_matches["HTAG"], errors="coerce")

        ht_valid = all_matches.dropna(subset=["HTHG", "HTAG"]).copy()
        if not ht_valid.empty:
            ht_valid["HTGoals"] = ht_valid["HTHG"] + ht_valid["HTAG"]
            ht_valid["FH_Over15"] = (ht_valid["HTGoals"] > 1.5).astype(int)
            ht_valid["FH_BTTS"] = ((ht_valid["HTHG"] > 0) & (ht_valid["HTAG"] > 0)).astype(int)

            ht_valid["SHGoals"] = ht_valid["TotalGoals"] - ht_valid["HTGoals"]
            ht_valid["GoalsBothHalves"] = (
                (ht_valid["HTGoals"] > 0) & (ht_valid["SHGoals"] > 0)
            ).astype(int)

            fh_over15_pct = ht_valid["FH_Over15"].mean() * 100
            fh_btts_pct = ht_valid["FH_BTTS"].mean() * 100
            goals_both_halves_pct = ht_valid["GoalsBothHalves"].mean() * 100

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
        "BTTS_Pct": round(btts_pct, 1),
        "FH_Over15_Pct": round(fh_over15_pct, 1) if fh_over15_pct is not None else None,
        "FH_BTTS_Pct": round(fh_btts_pct, 1) if fh_btts_pct is not None else None,
        "GoalsBothHalves_Pct": round(goals_both_halves_pct, 1) if goals_both_halves_pct is not None else None,
        "Over2_5_Pct": round(o25_pct, 1),
        "Over3_5_Pct": round(o35_pct, 1),
        "Over4_5_Pct": round(o45_pct, 1),
    }


# ============
# STREAMLIT UI
# ============

def main():
    st.set_page_config(
        page_title="All Euro 2025-26 Stats",
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

    leagues = sorted(data["League"].dropna().unique())
    n_matches = len(data)
    n_leagues = len(leagues)

    date_min = data["Date"].min()
    date_max = data["Date"].max()

    date_from_display = date_min.date().isoformat() if pd.notna(date_min) else "N/A"
    date_to_display = date_max.date().isoformat() if pd.notna(date_max) else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total matches", f"{n_matches:,}")
    col2.metric("Leagues", n_leagues)
    col3.metric("Date from", date_from_display)
    col4.metric("Date to", date_to_display)

    st.markdown("---")

    tab_overview, tab_league = st.tabs(["Leagues overview", "League & teams"])

    with tab_overview:
        st.subheader("Leagues summary")
        league_stats = compute_league_stats(data)
        if league_stats.empty:
            st.warning("Required columns for league stats not found.")
        else:
            league_stats_sorted = league_stats.sort_values("AvgGoals", ascending=False)
            st.dataframe(
                league_stats_sorted,
                use_container_width=True,
                height=600,
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

        teams_in_league = sorted(
            pd.unique(league_df[["HomeTeam", "AwayTeam"]].values.ravel("K"))
        )
        teams_in_league = [t for t in teams_in_league if pd.notna(t)]

        st.markdown(f"### Team stats – {sel_league}")
        if not teams_in_league:
            st.warning("No teams detected for this league.")
            return

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
            c8.metric("Avg total goals", tstats["AvgTotalGoals"])

            c9, c10, c11, c12 = st.columns(4)
            c9.metric("BTTS %", f"{tstats['BTTS_Pct']}%")

            fh_over15_display = (
                f"{tstats['FH_Over15_Pct']}%"
                if tstats["FH_Over15_Pct"] is not None
                else "N/A"
            )
            fh_btts_display = (
                f"{tstats['FH_BTTS_Pct']}%"
                if tstats["FH_BTTS_Pct"] is not None
                else "N/A"
            )
            gbh_display = (
                f"{tstats['GoalsBothHalves_Pct']}%"
                if tstats["GoalsBothHalves_Pct"] is not None
                else "N/A"
            )

            c10.metric("1H > 1.5 %", fh_over15_display)
            c11.metric("1H BTTS %", fh_btts_display)
            c12.metric("Goals both halves %", gbh_display)

            c13, c14, c15 = st.columns(3)
            c13.metric("Over 2.5 %", f"{tstats['Over2_5_Pct']}%")
            c14.metric("Over 3.5 %", f"{tstats['Over3_5_Pct']}%")
            c15.metric("Over 4.5 %", f"{tstats['Over4_5_Pct']}%")

            st.markdown(f"**Last 5 results:** {tstats['FormLast5']} (W/D/L sequence)")

            st.markdown("#### Matches for selected team")
            team_matches = league_df[
                (league_df["HomeTeam"] == sel_team)
                | (league_df["AwayTeam"] == sel_team)
            ].copy()

            if "Date" in team_matches.columns:
                team_matches = team_matches.sort_values("Date", ascending=False)

            cols_to_show = []
            for col in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]:
                if col in team_matches.columns:
                    cols_to_show.append(col)
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
