import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
st.set_page_config(
    page_title="IPL Match Statistics Analyzer",
    page_icon="🏏",
    layout="wide"
)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("IPL_Cleaned.csv", low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv("IPL.csv", low_memory=False)
        except FileNotFoundError:
            st.error(
                "IPL_Cleaned.csv not found. Keep IPL_Cleaned.csv "
                "or IPL.csv in the same folder as app.py."
            )
            st.stop()
    df.columns = df.columns.str.strip()
    numeric_columns = [
        "runs_batter",
        "runs_total",
        "runs_bowler",
        "is_wicket"
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)
    if "season" in df.columns:
        df["season"] = df["season"].astype(str).str.strip()
    return df
df = load_data()
def find_column(possible_names):
    columns = {
        col.lower().strip(): col
        for col in df.columns
    }
    for name in possible_names:
        if name.lower().strip() in columns:
            return columns[name.lower().strip()]
    return None
runs_col = find_column([
    "runs_batter",
    "batter_runs",
    "runs_off_bat"
])
total_runs_col = find_column([
    "runs_total",
    "total_runs"
])
runs_bowler_col = find_column([
    "runs_bowler",
    "bowler_runs",
    "runs_conceded"
])
wicket_col = find_column([
    "is_wicket",
    "bowler_wicket",
    "wickets"])
batter_col = find_column([
    "batter",
    "batsman"
])
bowler_col = find_column([
    "bowler"
])
batting_team_col = find_column([
    "batting_team"
])
bowling_team_col = find_column([
    "bowling_team"
])
match_col = find_column([
    "match_id",
    "id"
])
season_col = find_column([
    "season"
])
date_col = find_column([
    "date"
])
st.sidebar.title("🏏 IPL Analyzer")
page = st.sidebar.radio(
    "",
    [
        "🏠 Overview",
        "Batting Analysis",
        "Bowling Analysis",
        "Team Analysis",
        "Season Analysis",
        "IPL Records",
        "Final Score Prediction"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "IPL Match Statistics Analyzer\n\n"
    "Data Analytics + Visualization + ML"
)
if page == "🏠 Overview":
    st.title("🏏 IPL Match Statistics Analyzer")
    st.markdown(
        "### Interactive IPL Data Analytics Dashboard"
    )
    st.divider()
    total_records = len(df)
    if match_col:
        total_matches = df[match_col].nunique()
    else:
        total_matches = 0
    if season_col:
        total_seasons = df[season_col].nunique()
    else:
        total_seasons = 0
    if batting_team_col:
        total_teams = df[batting_team_col].nunique()
    else:
        total_teams = 0
    if batter_col:
        total_players = df[batter_col].nunique()
    else:
        total_players = 0
    if total_runs_col:
        total_runs = int(df[total_runs_col].sum())
    else:
        total_runs = 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Matches",
        f"{total_matches:,}"
    )
    c2.metric(
        "Seasons",
        f"{total_seasons:,}"
    )
    c3.metric(
        "Teams",
        f"{total_teams:,}"
    )
    c4.metric(
        "Players",
        f"{total_players:,}"
    )
    c5.metric(
        "Total Runs",
        f"{total_runs:,}"
    )
    st.divider()
    if season_col and match_col:
        st.subheader(
            "📈 Matches Played Across Seasons"
        )
        season_matches = (
            df.groupby(season_col)[match_col]
            .nunique()
            .reset_index(name="matches")
        )
        season_matches[season_col] = (
            season_matches[season_col]
            .astype(str)
        )
        fig, ax = plt.subplots(
            figsize=(12, 4)
        )
        ax.plot(
            season_matches[season_col],
            season_matches["matches"],
            marker="o",
            linewidth=2
        )
        ax.set_xlabel("Season")
        ax.set_ylabel("Matches")
        ax.set_title("IPL Matches by Season")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    st.success(
        "Use the navigation menu on the left to explore "
    )
elif page == "Batting Analysis":
    st.title("Batting Analysis 🏏")
    if batter_col and runs_col:
        top_batters = (
            df.groupby(batter_col)[runs_col]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
            .sort_values()
        )
        st.subheader(
            "Top 10 Run Scorers"
        )
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )
        ax.barh(
            top_batters.index.astype(str),
            top_batters.values
        )
        ax.set_xlabel("Runs")
        ax.set_ylabel("Player")
        ax.set_title(
            "Top 10 IPL Run Scorers"
        )
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader(
            "📋 Top Run Scorers Table"
        )
        batting_table = (
            top_batters
            .sort_values(ascending=False)
            .reset_index()
        )
        batting_table.columns = [
            "Player",
            "Runs"
        ]
        st.dataframe(
            batting_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Required batting columns were not found."
        )
elif page == "Bowling Analysis":
    st.title("Bowling Analysis 🎾")    
    if bowler_col and wicket_col:
        df[wicket_col] = pd.to_numeric(
            df[wicket_col],
            errors="coerce"
        ).fillna(0)
        top_bowlers = (
            df.groupby(bowler_col)[wicket_col]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
            .sort_values()
        )
        st.subheader(
            "Top 10 Wicket Takers"
        )
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )
        ax.barh(
            top_bowlers.index.astype(str),
            top_bowlers.values
        )
        ax.set_xlabel("Wickets")
        ax.set_ylabel("Bowler")
        ax.set_title(
            "Top 10 IPL Wicket Takers"
        )
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader(
            "📋 Bowling Leaderboard"
        )
        bowling_table = (
            top_bowlers
            .sort_values(ascending=False)
            .reset_index()
        )
        bowling_table.columns = [
            "Bowler",
            "Wickets"
        ]
        st.dataframe(
            bowling_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Required bowling/wicket columns were not found."
        )
elif page == "Team Analysis":
    st.title("Team Analysis 🏆")
    if batting_team_col and match_col:
        team_matches = (
            df.groupby(batting_team_col)[match_col]
            .nunique()
            .sort_values(
                ascending=False
            )
        )
        st.subheader(
            "Matches Played by Teams"
        )
        fig, ax = plt.subplots(
            figsize=(11, 6)
        )
        ax.bar(
            team_matches.index.astype(str),
            team_matches.values
        )
        ax.set_xlabel("Team")
        ax.set_ylabel("Matches")
        ax.set_title(
            "Team Participation in IPL"
        )
        plt.xticks(rotation=75)
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader(
            "📋 Team Statistics"
        )
        team_table = (
            team_matches
            .reset_index()
        )
        team_table.columns = [
            "Team",
            "Matches"
        ]
        st.dataframe(
            team_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Team or match columns were not found."
        )
elif page == "Season Analysis":
    st.title("Season Analysis 📊")
    if season_col and total_runs_col:
        season_runs = (
            df.groupby(season_col)[total_runs_col]
            .sum()
            .reset_index()
        )
        season_runs.columns = [
            "Season",
            "Runs"
                ]
        season_runs["Season"] = (
            season_runs["Season"]
            .astype(str)
        )
        st.subheader(
            "Total Runs by Season"
        )
        fig, ax = plt.subplots(
            figsize=(12, 5)
        )
        ax.plot(
            season_runs["Season"],
            season_runs["Runs"],
            marker="o",
            linewidth=2
        )
        ax.set_xlabel("Season")
        ax.set_ylabel("Total Runs")
        ax.set_title(
            "IPL Run Trend Across Seasons"
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader(
            "📋 Season Statistics"
        )
        st.dataframe(
            season_runs.sort_values(
                "Runs",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Season/run columns were not found."
        )
elif page == "IPL Records":
    st.title("IPL Records 📈")
    if batter_col and runs_col and match_col:
        player_scores = (
            df.groupby(
                [match_col, batter_col]
            )[runs_col]
            .sum()
            .reset_index()
        )
        highest_score = (
            player_scores
            .sort_values(
                runs_col,
                ascending=False
            )
            .head(10)
        )
        st.subheader(
            "Highest Individual Match Scores"
        )
        record_table = highest_score.copy()
        record_table.columns = [
            "Match ID",
            "Player",
            "Runs"
        ]
        st.dataframe(
            record_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Required columns for individual scores "
            "were not found."
        )
    st.markdown("---")
    if (
        total_runs_col
        and match_col
        and batting_team_col
    ):
        team_scores = (
            df.groupby(
                [
                    match_col,
                    batting_team_col
                ]
            )[total_runs_col]
            .sum()
            .reset_index()
        )
        highest_team_scores = (
            team_scores
            .sort_values(
                total_runs_col,
                ascending=False
            )
            .head(10)
        )
        st.subheader(
            "Highest Team Scores"
        )
        team_record = (
            highest_team_scores.copy()
        )
        team_record.columns = [
            "Match ID",
            "Team",
            "Runs"
        ]
        st.dataframe(
            team_record,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Required columns for team scores "
            "were not found."
        )
    st.markdown("---")
    st.subheader(
        "Best Bowling Figures"
    )
    if (
        match_col
        and bowler_col
        and bowling_team_col
        and wicket_col
    ):
        df[wicket_col] = pd.to_numeric(
            df[wicket_col],
            errors="coerce"
        ).fillna(0)
        if runs_bowler_col:
            df[runs_bowler_col] = pd.to_numeric(
                df[runs_bowler_col],
                errors="coerce"
            ).fillna(0)
            bowling_records = (
                df.groupby(
                    [
                        match_col,
                        bowler_col,
                        bowling_team_col
                    ]
                )
                .agg(
                    wickets=(
                        wicket_col,
                        "sum"
                    ),
                    runs_conceded=(
                        runs_bowler_col,
                        "sum"
                    )
                )
                .reset_index()
            )
        else:
            if total_runs_col:
                bowling_records = (
                    df.groupby(
                        [
                            match_col,
                            bowler_col,
                            bowling_team_col
                        ]
                    )
                    .agg(
                        wickets=(
                            wicket_col,
                            "sum"
                        ),
                        runs_conceded=(
                            total_runs_col,
                            "sum"
                        )
                    )
                    .reset_index()
                )
            else:
                bowling_records = (
                    df.groupby(
                        [
                            match_col,
                            bowler_col,
                            bowling_team_col
                        ]
                    )
                    .agg(
                        wickets=(
                            wicket_col,
                            "sum"
                        )
                    )
                    .reset_index()
                )
                bowling_records[
                    "runs_conceded"
                ] = 0
        bowling_records = (
            bowling_records[
                bowling_records["wickets"] > 0
            ]
        )
        best_bowling = (
            bowling_records
            .sort_values(
                [
                    "wickets",
                    "runs_conceded"
                ],
                ascending=[
                    False,
                    True
                ]
            )
            .head(10)
            .copy()
        )
        best_bowling.columns = [
            "Match ID",
            "Bowler",
            "Bowling Team",
            "Wickets",
            "Runs Conceded"
        ]
        st.dataframe(
            best_bowling,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Required bowling columns were not found."
        )
elif page == "Final Score Prediction":
    st.title("Predict Final IPL Score 😎")
    st.markdown(
        "Use the current match situation to estimate the final score of the batting team."
    )
    required_ml_columns = [
        match_col,
        batting_team_col,
        bowling_team_col,
        total_runs_col
    ]
    if not all(required_ml_columns):
        st.error(
            "ML prediction requires match_id, batting_team, "
            "bowling_team and runs_total."
        )
        st.stop()
    innings_col = find_column(["innings", "inning"])
    valid_ball_col = find_column(["valid_ball", "legal_ball"])
    ml_df = df.copy()
    ml_df[total_runs_col] = pd.to_numeric(
        ml_df[total_runs_col], errors="coerce"
    ).fillna(0)
    player_out_col = find_column(["player_out"])
    wicket_kind_col = find_column(["wicket_kind"])
    if wicket_col:
        ml_df["_wicket_event"] = pd.to_numeric(
            ml_df[wicket_col], errors="coerce"
        ).fillna(0).clip(0, 1)
    elif player_out_col:
        ml_df["_wicket_event"] = ml_df[player_out_col].notna().astype(int)
    elif wicket_kind_col:
        ml_df["_wicket_event"] = ml_df[wicket_kind_col].notna().astype(int)
    else:
        ml_df["_wicket_event"] = 0
    if valid_ball_col:
        ml_df["_legal_ball"] = pd.to_numeric(
            ml_df[valid_ball_col], errors="coerce"
        ).fillna(0).clip(0, 1)
    else:
        extra_type_col = find_column(["extra_type"])
        if extra_type_col:
            ml_df["_legal_ball"] = (~ml_df[extra_type_col].isin(
                ["wides", "noballs", "no_balls", "wide", "noball"]
            )).astype(int)
        else:
            ml_df["_legal_ball"] = 1
    ml_df["_row_order"] = range(len(ml_df))
    group_cols = [match_col, batting_team_col, bowling_team_col]
    if innings_col:
        group_cols.insert(1, innings_col)
    final_cols = group_cols + [total_runs_col]
    innings_totals = (
        ml_df.groupby(group_cols, dropna=False)[total_runs_col]
        .sum()
        .reset_index(name="final_score")
    )
    ml_df = ml_df.sort_values("_row_order").copy()
    ml_df["current_score"] = (
        ml_df.groupby(group_cols, dropna=False)[total_runs_col]
        .cumsum()
    )
    ml_df["wickets_lost"] = (
        ml_df.groupby(group_cols, dropna=False)["_wicket_event"]
        .cumsum()
        .clip(0, 10)
    )
    ml_df["legal_balls"] = (
        ml_df.groupby(group_cols, dropna=False)["_legal_ball"]
        .cumsum()
    )
    ml_df = ml_df.merge(
        innings_totals,
        on=group_cols,
        how="left"
    )
    snapshots = ml_df[
        (ml_df["legal_balls"] >= 6) &
        (ml_df["legal_balls"] <= 120) &
        (ml_df["legal_balls"] % 3 == 0)
    ].copy()
    last_states = (
        ml_df[ml_df["legal_balls"] >= 6]
        .groupby(group_cols, dropna=False)["_row_order"]
        .idxmax()
    )
    if len(last_states) > 0:
        last_states = ml_df.loc[last_states].copy()
        snapshots = pd.concat(
            [snapshots, last_states],
            ignore_index=True
        )
    snapshots = snapshots.drop_duplicates(
        subset=group_cols + ["legal_balls"]
    )
    snapshots["current_run_rate"] = (
        snapshots["current_score"] /
        (snapshots["legal_balls"] / 6.0)
    ).replace([float("inf"), -float("inf")], 0).fillna(0)
    snapshots["remaining_balls"] = (
        120 - snapshots["legal_balls"]
    ).clip(lower=0)
    snapshots = snapshots.dropna(
        subset=[
            batting_team_col,
            bowling_team_col,
            "current_score",
            "wickets_lost",
            "legal_balls",
            "final_score"
        ]
    )
    snapshots = snapshots[
        (snapshots["wickets_lost"] <= 10) &
        (snapshots["current_score"] <= snapshots["final_score"]) &
        (snapshots["final_score"] > 0)
    ].copy()
    if len(snapshots) > 30000:
        snapshots = snapshots.sample(
            n=30000,
            random_state=42
        )
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    feature_cols = [
        batting_team_col,
        bowling_team_col,
        "current_score",
        "wickets_lost",
        "legal_balls",
        "current_run_rate",
        "remaining_balls"
    ]
    X = snapshots[feature_cols].copy()
    y = snapshots["final_score"].astype(float)
    categorical_features = [
        batting_team_col,
        bowling_team_col
    ]
    numeric_features = [
        "current_score",
        "wickets_lost",
        "legal_balls",
        "current_run_rate",
        "remaining_balls"
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "teams",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numbers",
                "passthrough",
                numeric_features
            )
        ]
    )
    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ])
    @st.cache_resource(show_spinner="Training IPL score prediction model...")
    def train_prediction_model(training_data, target_data):
        X_train, X_test, y_train, y_test = train_test_split(
            training_data,
            target_data,
            test_size=0.20,
            random_state=42
        )

        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)

        model_mae = mean_absolute_error(
            y_test,
            test_predictions
        )
        model_r2 = r2_score(
            y_test,
            test_predictions
        )
        return model, model_mae, model_r2

    if len(X) >= 20:
        model, mae, r2 = train_prediction_model(X, y)
    else:
        st.error("Not enough innings data to train the prediction model.")
        st.stop()
    st.subheader("Match Information")

    all_teams = sorted(
        pd.concat([
            df[batting_team_col],
            df[bowling_team_col]
        ])
        .dropna()
        .astype(str)
        .unique()
    )
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox(
            "Batting Team",
            all_teams,
            key="prediction_batting_team"
        )
    with col2:
        bowling_team = st.selectbox(
            "Bowling Team",
            all_teams,
            key="prediction_bowling_team"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        completed_overs = st.number_input(
            "Completed Overs",
            min_value=0,
            max_value=20,
            value=10,
            step=1,
            help="Enter completed full overs only."
        )
    with col2:
        balls_in_over = st.number_input(
            "Balls in Current Over",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            help="0 = over completed. Maximum is 5 because 6 balls complete an over."
        )
    with col3:
        current_score = st.number_input(
            "Current Score",
            min_value=0,
            max_value=400,
            value=80,
            step=1
        )
    wickets_lost = st.number_input(
        "Wickets Lost",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )
    legal_balls = int(completed_overs) * 6 + int(balls_in_over)
    if legal_balls > 120:
        st.error("A T20 innings cannot have more than 20 overs (120 legal balls).")
        st.stop()
    if int(completed_overs) == 20 and int(balls_in_over) > 0:
        st.error("After 20 overs, balls in the current over must be 0.")
        st.stop()
    if legal_balls == 0:
        current_run_rate = 0.0
    else:
        current_run_rate = current_score / (legal_balls / 6.0)
    remaining_balls = max(0, 120 - legal_balls)
    if st.button(
        "Predict Final Score",
        use_container_width=True
    ):
        input_data = pd.DataFrame({
            batting_team_col: [batting_team],
            bowling_team_col: [bowling_team],
            "current_score": [current_score],
            "wickets_lost": [wickets_lost],
            "legal_balls": [legal_balls],
            "current_run_rate": [current_run_rate],
            "remaining_balls": [remaining_balls]
        })
        if legal_balls >= 120 or wickets_lost >= 10:
            final_prediction = float(current_score)
        else:
            ml_prediction = float(model.predict(input_data)[0])
            if legal_balls > 0:
                projected_score = (
                    current_score +
                    current_run_rate * (remaining_balls / 6.0)
                )
            else:
                projected_score = ml_prediction
            final_prediction = (
                0.75 * ml_prediction +
                0.25 * projected_score
            )
            final_prediction = max(
                float(current_score),
                final_prediction
            )
            final_prediction = min(
                350.0,
                final_prediction
            )
        final_prediction = round(final_prediction)
        st.success(
            f"Predicted Final Score: **{final_prediction} runs**"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Current Score",
                f"{current_score} runs"
            )
        with c2:
            st.metric(
                "Current Run Rate",
                f"{current_run_rate:.2f}"
            )
        with c3:
            st.metric(
                "Predicted Score",
                f"{final_prediction}"
            )
        lower = max(
            int(current_score),
            int(final_prediction - 10)
        )
        upper = min(
            350,
            int(final_prediction + 10)
        )
        st.info(
            f"Expected scoring range: **{lower} – {upper} runs**"
        )
        st.caption(
            f"Match state: {completed_overs} overs + {balls_in_over} balls "
            f"= {legal_balls} legal balls completed, "
            f"with {remaining_balls} legal balls remaining."
        )
    st.divider()
    st.subheader("🧠 Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Mean Absolute Error",
            f"{mae:.2f} runs"
        )
    with c2:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )
st.sidebar.markdown("---")
st.sidebar.caption("IPL Match Statistics Analyzer")
st.sidebar.caption("Python • Pandas • Matplotlib • Streamlit")
