import streamlit as st
import sys
import os
import altair as alt
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from model.cp_sat_model import run_model_and_get_results


st.title("HARMONOGRAM DYŻURÓW")

status, schedule_df, stats_df, solver_stats_df = run_model_and_get_results()


st.write("Status:", status)
st.dataframe(schedule_df)
st.dataframe(stats_df)

# === Godziny pracy na osobę ===
st.subheader("Liczba godzin pracy na lekarza")

df_hours = stats_df[["Doctor", "TotalHours", "MaxHours"]]
bars = (
    alt.Chart(df_hours)
    .mark_bar(color="#4C78A8")
    .encode(
        x=alt.X("Doctor", sort="-y"),
        y=alt.Y("TotalHours", title="Godziny pracy"),
        tooltip=["Doctor", "TotalHours", "MaxHours"]
    )
)
limits = (
    alt.Chart(df_hours)
    .mark_rule(color="pink", strokeWidth=2)
    .encode(
        x="Doctor",
        y="MaxHours",
        tooltip=["Doctor", "MaxHours"]
    )
)
chart = (bars + limits).properties(height=350)
st.altair_chart(chart, use_container_width=True)


# ===== LICZBA DYŻURÓW NOCNYCH =====
print("Liczba dyżurów nocnych (nocne + 24h)")
df_night_shifts = stats_df[["Doctor", "NightShifts"]]
chart_shifts = (
    alt.Chart(df_night_shifts)
    .mark_bar(color="#4C78A8")
    .encode(
        x=alt.X("Doctor", sort="-y"),
        y=alt.Y("NightShifts", title="Liczba zmian nocnych"),
        tooltip=["Doctor", "NightShifts"]
    )
)
st.altair_chart(chart_shifts, use_container_width=True)

# ===== LICZBA ZMIAN 24H =====
print("Liczba zmian 24-godzinnych")
df_24_shifts = stats_df[["Doctor", "TwentyFourCount"]]
chart_24_shifts = (
    alt.Chart(df_24_shifts)
    .mark_bar(color="#4C78A8")
    .encode(
        x=alt.X("Doctor", sort="-y"),
        y=alt.Y("TwentyFourCount", title="Liczba zmian 24-godzinnych"),
        tooltip=["Doctor", "TwentyFourCount"]
    )
)
st.altair_chart(chart_24_shifts, use_container_width=True)

# ===== OBCIĄŻENIE ODDZIAŁ / ICU / PORADNIA =====
st.subheader("D. Obciążenie pracą (oddział / OIOM / poradnia)")

df_load = stats_df[["Doctor", "WardCount", "ICUCount", "ClinicCount"]]

df_melt = df_load.melt(id_vars=["Doctor"],
                       value_vars=["WardCount", "ICUCount", "ClinicCount"],
                       var_name="Department",
                       value_name="Count")
color_scale = alt.Scale(
    domain=["WardCount", "ICUCount", "ClinicCount"],
    range=["#df7680", "#008a80", "#68295c"]
)
stacked = (
    alt.Chart(df_melt)
    .mark_bar()
    .encode(
        x=alt.X("Doctor", sort=None),
        y="Count",
        color=alt.Color("Department", scale=color_scale, title="Oddział"),
        tooltip=["Doctor", "Department", "Count"]
    )
    .properties(height=350)
)

st.altair_chart(stacked, use_container_width=True)

# ===== SPEŁNIONE PREFERENCJE =====
st.subheader("Preferencje lekarzy — spełnione like / naruszone dislike")

df_pref = stats_df[["Doctor", "like_satisfied", "dislike_violated"]]

df_pref_melt = df_pref.melt(
    id_vars=["Doctor"],
    value_vars=["like_satisfied", "dislike_violated"],
    var_name="PreferenceType",
    value_name="Count"
)

pref_chart = (
    alt.Chart(df_pref_melt)
    .mark_bar()
    .encode(
        x=alt.X("Doctor:N", sort=None),
        y=alt.Y("Count:Q"),
        color=alt.Color("PreferenceType:N",
                        scale=alt.Scale(
                            domain=["like_satisfied", "dislike_violated"],
                            range=["#4CAF50", "#E53935"]  # zielony / czerwony
                        )),
        column=alt.Column("PreferenceType:N", header=alt.Header(title="")),
        tooltip=["Doctor", "PreferenceType", "Count"]
    )
    .properties(height=300)
)

st.altair_chart(pref_chart, use_container_width=True)


# ===== FAIRNESS =====
st.header("Fairness")

st.subheader("Dyżury nocne – rozkład na lekarzy")

max_nights = int(solver_stats_df["max_nights"].iloc[0])
min_nights = int(solver_stats_df["min_nights"].iloc[0])
spread = int(solver_stats_df["spread"].iloc[0])


st.write(
    f"Zakres dyżurów nocnych: **{min_nights}–{max_nights}**, "
    f"różnica (spread) = **{spread}**"
)

base = alt.Chart(stats_df).encode(
    x=alt.X("Doctor:N", sort=None)
)

bars = base.mark_bar().encode(
    y=alt.Y("NightShifts:Q", title="Liczba dyżurów nocnych"),
    tooltip=["Doctor", "NightShifts", "TwentyFourCount"]
)

rules_df = pd.DataFrame({
    "y": [min_nights, max_nights],
    "label": ["min_nights", "max_nights"]
})

rules = (
    alt.Chart(rules_df)
    .mark_rule(strokeDash=[4, 4])
    .encode(
        y="y:Q",
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=["min_nights", "max_nights"],
                range=["#4CAF50", "#E53935"]
            ),
            title="Granice"
        )
    )
)

fairness_chart = (bars + rules).properties(height=300)

st.altair_chart(fairness_chart, use_container_width=True)

# ===== POTENCJALNIE PRZEPRACOWANI =====
st.subheader("Lekarze potencjalnie przepracowani")

cols = ["Doctor", "TotalHours", "MaxHours", "NightShifts", "TwentyFourCount"]
problematic = stats_df.copy()
problematic["OverLimit90"] = (
    (problematic["MaxHours"] > 48) &
    (problematic["TotalHours"] >= 0.9 * problematic["MaxHours"])
)
problematic["Many24h"] = problematic["TwentyFourCount"] >= 2
problematic["ManyNights"] = problematic["NightShifts"] >= 3

problematic["AnyIssue"] = (
    problematic["OverLimit90"] |
    problematic["Many24h"] |
    problematic["ManyNights"]
)

bar = (
    alt.Chart(problematic)
    .mark_bar()
    .encode(
        x=alt.X("Doctor:N", sort="-y"),
        y=alt.Y("TotalHours:Q", title="Liczba godzin w tygodniu"),
        color=alt.condition(
            "datum.AnyIssue",
            alt.value("crimson"),
            alt.value("steelblue"),
        ),
        tooltip=[
            "Doctor",
            "TotalHours",
            "MaxHours",
            "NightShifts",
            "TwentyFourCount",
            "OverLimit90",
            "Many24h",
            "ManyNights",
        ],
    )
    .properties(height=350)
)

st.altair_chart(bar, use_container_width=True)

risk_df = problematic.melt(
    id_vars=["Doctor"],
    value_vars=["OverLimit90", "Many24h", "ManyNights"],
    var_name="RiskType",
    value_name="Flag",
)

risk_df = risk_df[risk_df["Flag"]]

heatmap = (
    alt.Chart(risk_df)
    .mark_rect()
    .encode(
        y=alt.Y("Doctor:N", sort=None),
        x=alt.X("RiskType:N", title="Typ problemu"),
        color=alt.value("orange"),
        tooltip=["Doctor", "RiskType"],
    )
    .properties(height=300)
)

st.subheader("Mapa problemów (kto ma jaki typ obciążenia)")
st.altair_chart(heatmap, use_container_width=True)

st.subheader("Tabela – szczegóły dla lekarzy z co najmniej jednym problemem")

problematic_filtered = problematic[problematic["AnyIssue"]]

st.dataframe(
    problematic_filtered[
        ["Doctor", "TotalHours", "MaxHours",
         "NightShifts", "TwentyFourCount",
         "OverLimit90", "Many24h", "ManyNights"]
    ].sort_values(by="TotalHours", ascending=False)
)
