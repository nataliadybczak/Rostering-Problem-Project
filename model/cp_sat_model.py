from ortools.sat.python import cp_model
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# === HELPERS =====================================================

def build_doctor_stats(
    D, S, id_to_name, id_to_role, x, hours,
    night_shifts, twentyfour_shifts, shifts, shift_idx, max_hours, solver
):
    """Buduje DataFrame ze statystykami dla każdego lekarza"""
    rows = []
    for d in D:
        rows.append({
            "Doctor": id_to_name[d],
            "Role": id_to_role[d],
            "TotalHours": sum(solver.Value(x[(d, s)]) * hours[s] for s in S),
            "MaxHours": max_hours[d],
            "NightShifts": sum(solver.Value(x[(d, s)]) for s in night_shifts),
            "TwentyFourCount": sum(solver.Value(x[(d, s)]) for s in twentyfour_shifts),
            "WardCount": sum(
                solver.Value(x[(d, s)])
                for s in S if shifts.loc[shift_idx[s], "dept"] == "WARD"
            ),
            "ICUCount": sum(
                solver.Value(x[(d, s)])
                for s in S if shifts.loc[shift_idx[s], "dept"] == "ICU"
            ),
            "ClinicCount": sum(
                solver.Value(x[(d, s)])
                for s in S if shifts.loc[shift_idx[s], "dept"] == "CLINIC"
            ),
        })
    return pd.DataFrame(rows)


def add_preference_stats(stats_df, pref, x, solver, id_to_name, code_to_id):
    """Dodaje do stats_df kolumny like_satisfied / dislike_violated dla każdego lekarza"""
    stats_df["like_satisfied"] = 0
    stats_df["dislike_violated"] = 0

    for _, row in pref.iterrows():
        d = row["doctor_id"]
        s = code_to_id[row["code"]]
        pref_type = row["preference"]

        if solver.Value(x[(d, s)]) == 1:
            col = "like_satisfied" if pref_type == "like" else "dislike_violated"
            stats_df.loc[stats_df["Doctor"] == id_to_name[d], col] += 1

    return stats_df


def build_solver_stats(solver, max_nights, min_nights, spread, status):
    """Zwraca DataFrame z globalnymi statystykami"""
    solver_stats = {
        "objective_value": solver.ObjectiveValue()
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "status": solver.StatusName(status),
        "conflicts": solver.NumConflicts(),
        "branches": solver.NumBranches(),
        "wall_time": solver.WallTime(),
        "max_nights": solver.Value(max_nights),
        "min_nights": solver.Value(min_nights),
        "spread": solver.Value(spread),
    }
    return pd.DataFrame([solver_stats])


# === MAIN FUNCTION ===
def run_model_and_get_results():
    doctors = pd.read_csv(DATA_DIR / "doctors2.csv")
    shifts = pd.read_csv(DATA_DIR / "shifts.csv")
    unavail_day = pd.read_csv(DATA_DIR / "unavailabilities_day.csv")
    unavail_shift = pd.read_csv(DATA_DIR / "unavailabilities_shift.csv")
    pref = pd.read_csv(DATA_DIR / "preferences.csv")

    D = list(doctors["id"])
    S = list(shifts["id"])

    doctor_idx = {doc_id: idx for idx, doc_id in enumerate(D)}
    shift_idx = {shift_id: idx for idx, shift_id in enumerate(S)}

    id_to_name = {row["id"]: row["name"] for _, row in doctors.iterrows()}
    id_to_role = {row["id"]: row["role"] for _, row in doctors.iterrows()}

    doctors["skill_list"] = doctors["skills"].apply(lambda skill: skill.split(";") if isinstance(skill, str) else [])

    model = cp_model.CpModel()

    """ VARIABLES """
    x = {}
    for d in D:
        for s in S:
            x[(d, s)] = model.NewBoolVar(f"x_{d}_{s}")

    """ LISTS AND DICTS """
    shift_day = {shift["id"]: shift["day"] for _, shift in shifts.iterrows()}
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_index = {d: i for i, d in enumerate(days)}

    shift_start = {shift["id"]: shift["start_hour"] for _, shift in shifts.iterrows()}
    shift_end = {shift["id"]: shift["end_hour"] for _, shift in shifts.iterrows()}

    # czas liczony jako czas od początku tygodnia
    abs_start = {s: day_index[shift_day[s]] * 24 + shift_start[s] for s in S}
    abs_end = {s: day_index[shift_day[s]] * 24 + shift_end[s] for s in S}

    hours = {shift["id"]: shift["hours"] for _, shift in shifts.iterrows()}
    max_hours = {doctor["id"]: doctor["max_hours"] for _, doctor in doctors.iterrows()}

    specialists = doctors[doctors["role"].isin(["specialist", "icu_specialist"])]["id"]
    needs_mentor = doctors[doctors["needs_mentor"]==1]["id"]

    twentyfour_allowed = {doctor["id"]: doctor["twentyfour_allowed"] for _, doctor in doctors.iterrows()}
    twentyfour_shifts = [s for s in S if shifts.loc[shift_idx[s], "hours"] == 24]

    night_shifts = [s for s in S if "_N_" in shifts.loc[shift_idx[s], "code"] or shifts.loc[shift_idx[s], "hours"] == 24]
    night_shifts_by_day = {day: [s for s in night_shifts if shift_day[s] == day] for day in days }

    day_shifts = {day: [s for s in S if shift_day[s] == day and hours[s] < 24] for day in days}
    day_24h = {day: [s for s in S if shift_day[s] == day and hours[s] == 24] for day in days}

    days_to_shifts = {day: [s for s in S if shift_day[s] == day] for day in days}
    code_to_id = {row["code"]: row["id"] for _, row in shifts.iterrows()}

    """ FUNCTIONS """
    def rest_violation(sh1, sh2, min_rest=11) -> bool:
        if abs_start[sh2] <= abs_end[sh1]:
            return False
        return (abs_start[sh2] - abs_end[sh1]) < min_rest

    """ HARD CONSTRAINTS """
    """1. Lekarz musi posiadać odpowiednie uprawnienia, aby mógł być przypisany do danej zmiany (taska) """
    for _, shift in shifts.iterrows():
        s = shift["id"]
        req = shift["required_skill"]

        for _, doctor in doctors.iterrows():
            d = doctor["id"]
            if req not in doctor["skill_list"]:
                model.Add(x[(d,s)] == 0)

    """2. Na każdej zmianie jest co najmniej wymagana liczba lekarzy."""
    for _, shift in shifts.iterrows():
        s = shift["id"]
        min_staff = shift["min_staff"]

        model.Add(
            sum(x[d, s] for d in D) >= min_staff
        )

    """3. Maksymalnie 1 zmiana w ciągu doby """
    for d in D:
        for day in days:
            shifts_per_day = [s for s in S if shift_day[s] == day]
            model.Add(
                # sum(x[(d, s)] for s in shifts_per_day) <= 1
                sum(x[(d, s)] for s in day_24h[day]) +
                sum(x[(d, s)] for s in day_shifts[day]) <= 1
            )

    """4. Co najmniej 11 godzin nieprzerwanego odpoczynku po zmianie """
    for d in D:
        for s1 in S:
            for s2 in S:
                if s1 == s2:
                    continue
                if rest_violation(s1, s2):
                    model.Add(x[(d, s1)] + x[(d, s2)] <= 1)


    """5. Zachowanie limitu tygodniowego godzin pracy (w zależności od lekarza) """
    for _, doctor in doctors.iterrows():
        d = doctor["id"]
        model.Add(
            sum(x[(d, s)] * hours[s] for s in S) <= max_hours[d]
        )

    """6. Opiekun dla stażysty (i niekórych rezydentów) """
    for s in S:
        for d in needs_mentor:
            model.Add(
                x[(d,s)] <= sum(x[(spec,s)] for spec in specialists)
            )

    """7. Maksymalnie 2 dyżury nocne pod rząd """
    for d in D:
        for i in range(len(days)-2):
            window_days = days[i:i+3]
            shifts_in_window = [shift for day in window_days for shift in night_shifts_by_day[day]]
            if shifts_in_window:
                model.Add(sum(x[(d, s)] for s in shifts_in_window) <= 2)

    """8. Dzień wolny po zmianie nocnej """
    for d in D:
        for i in range(len(days)-1):
            current_day = days[i]
            next_day = days[i+1]
            current_night_shifts = [s for s in night_shifts if shift_day[s] == current_day]
            next_day_shifts = [s for s in S if shift_day[s] == next_day]

            if current_night_shifts and next_day_shifts:
                model.Add(
                    sum(x[(d, s)] for s in current_night_shifts) + sum(x[(d, s)] for s in next_day_shifts) <= 1
                )


    """9. Co najmniej 35 godzin nieprzerwanego odpoczynku w każdym tygodniu """
    for d in D:
        shifts_per_day = {day: [s for s in S if shift_day[s] == day] for day in days}
        works_vars = []
        for day in days:
            works_var = model.NewBoolVar(f"works_{d}_{day}")
            works_vars.append(works_var)

            model.Add(sum(x[(d, s)] for s in shifts_per_day[day]) >= works_var)
            model.Add(sum(x[(d, s)] for s in shifts_per_day[day]) <= 1000 * works_var)
        model.Add(sum(works_vars) <= 6)


    '''10. Nie każdy może mieć 24-godzinny dyżur '''
    for d in D:
        if twentyfour_allowed[d] == 0:
            for s in twentyfour_shifts:
                model.Add(x[(d,s)] == 0)


    '''11. Uwzględnienie niedostępności (np: urlopy) '''
    # Cały dzień
    for _, row in unavail_day.iterrows():
        d = row["doctor_id"]
        day = row["day"]

        for s in days_to_shifts[day]:
            model.Add(x[d,s] == 0)

    # Konkretna zmiana
    for _, row in unavail_shift.iterrows():
        d = row["doctor_id"]
        code = row["code"]
        s = code_to_id[code]
        model.Add(x[d,s] == 0)


    """ SOFT CONSTRAINTS """
    pref_terms = []

    # === WORKLOAD PER DOCTOR (HOURS) ===
    worked_hours = {}
    for d in D:
        # od 0 do limitu godzin danego lekarza
        h_var = model.NewIntVar(0, max_hours[d], f"worked_hours_{d}")
        model.Add(h_var == sum(x[(d, s)] * hours[s] for s in S))
        worked_hours[d] = h_var


    """1. Preferencje """
    for _, row in pref.iterrows():
        d = row["doctor_id"]
        shift_code = row["code"]
        preference = row["preference"]

        s = code_to_id[shift_code]

        if preference == "like":
            pref_terms.append(-1 * x[(d, s)])

        if preference == "dislike":
            pref_terms.append(1 * x[(d, s)])


    """2. Fairness - jak najbardziej równomierne obłożenie trudnymi dyżurami """
    """a. Zmiany nocne """
    night_count = {}
    for d in D:
        count = model.NewIntVar(0, 100, f"night_count_{d}")
        model.Add(count == sum(x[(d, s)] for s in night_shifts))
        night_count[d] = count

    max_nights = model.NewIntVar(0, 100, "max_nights")
    min_nights = model.NewIntVar(0, 100, "min_nights")

    model.AddMaxEquality(max_nights, list(night_count.values()))
    model.AddMinEquality(min_nights, list(night_count.values()))

    spread = model.NewIntVar(0, 100, "night_spread")
    model.Add(spread == max_nights - min_nights)


    """b. Zmiany weekendowe - TODO: w perspektywnie miesiąca (na razie patrzymy na pojedynczy tydzień) """

    """3. Jak najbardziej równy procent wypracowanych godzin wzglęem limitu """
    workload_ratio = {}
    for d in D:
        ratio = model.NewIntVar(0, 2000, f"workload_ratio_{d}")

        model.Add(ratio * max_hours[d] <= worked_hours[d] * 1000 + 50)
        model.Add(ratio * max_hours[d] >= worked_hours[d] * 1000 - 50)

        workload_ratio[d] = ratio

    max_ratio = model.NewIntVar(0, 2000, "max_ratio")
    min_ratio = model.NewIntVar(0, 2000, "min_ratio")

    model.AddMaxEquality(max_ratio, list(workload_ratio.values()))
    model.AddMinEquality(min_ratio, list(workload_ratio.values()))

    ratio_spread = model.NewIntVar(0, 2000, "ratio_spread")
    model.Add(ratio_spread == max_ratio - min_ratio)

    """ OBJECTIVE FUNCTION """
    w_pref = 3
    w_night = 6
    w_ratio = 8

    objective_terms = []

    objective_terms += [w_pref * term for term in pref_terms]
    objective_terms.append(w_night * spread)
    objective_terms.append(w_ratio * ratio_spread)

    model.Minimize(sum(objective_terms))

    """ SOLVER """
    solver = cp_model.CpSolver()

    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)


    """ SCHEDULE DATAFRAME """
    rows = []
    shifts_sorted = shifts.sort_values(by=["day", "start_hour"])

    for day in days:
        for _, sh in shifts_sorted[shifts_sorted["day"] == day].iterrows():
            s_id = sh["id"]
            assigned = [d for d in D if solver.Value(x[(d, s_id)]) == 1]

            if assigned:
                for d in assigned:
                    rows.append({
                        "Day": day,
                        "ShiftCode": sh["code"],
                        "Dept": sh["dept"],
                        "StartHour": sh["start_hour"],
                        "EndHour": sh["end_hour"],
                        "Hours": sh["hours"],
                        "Doctor": id_to_name[d],
                        "Role": id_to_role[d],
                    })
            else:
                rows.append({
                    "Day": day,
                    "ShiftCode": sh["code"],
                    "Dept": sh["dept"],
                    "StartHour": sh["start_hour"],
                    "EndHour": sh["end_hour"],
                    "Hours": sh["hours"],
                    "Doctor": None,
                    "Role": None,
                })

    schedule_df = pd.DataFrame(rows)

    """ STATISTICS """
    stats_df = build_doctor_stats(
        D, S, id_to_name, id_to_role, x, hours,
        night_shifts, twentyfour_shifts, shifts, shift_idx, max_hours, solver
    )

    stats_df = add_preference_stats(stats_df, pref, x, solver, id_to_name, code_to_id)

    solver_stats_df = build_solver_stats(solver, max_nights, min_nights, spread, status)

    """ FUNCTIONS """
    def print_schedule():
        print("\n=== HARMONOGRAM TYGODNIOWY ===\n")

        data = []

        shifts_sorted = shifts.sort_values(by=["day", "start_hour"])

        for day in days:
            print(f"\n--- {day} ---")
            day_shifts = shifts_sorted[shifts_sorted["day"] == day]

            for _, sh in day_shifts.iterrows():
                s_id = sh["id"]
                code = sh["code"]
                dept = sh["dept"]
                start = sh["start_hour"]
                end = sh["end_hour"]
                hours_s = sh["hours"]

                assigned_docs = [
                    d for d in D if solver.Value(x[(d, s_id)]) == 1
                ]

                if not assigned_docs:
                    assigned_str = "(brak obsady!)"
                else:
                    assigned_str = ", ".join(
                        f"{id_to_name[d]} ({id_to_role[d]})" for d in assigned_docs
                    )

                print(f"{code:15s} [{dept:6s}] {start:02d}:00–{end % 24:02d}:00 ({hours_s}h) -> {assigned_str}")

        print("\\n=== RAPORT SZCZEGÓŁOWY ===\\n")
        # ===== A. LICZBA GODZIN NA OSOBĘ =====
        print("\\n--- A. Liczba godzin pracy na lekarza ---")
        total_hours_worked = {}
        for d in D:
            total = sum(solver.Value(x[(d, s)]) * hours[s] for s in S)
            total_hours_worked[d] = total
            print(f"{id_to_name[d]:15s}: {total} h / limit {max_hours[d]}")

        # ===== B. LICZBA DYŻURÓW NOCNYCH =====
        print("\\n--- B. Liczba dyżurów nocnych (nocne + 24h) ---")
        night_counts = {}
        for d in D:
            cnt = sum(solver.Value(x[(d, s)]) for s in night_shifts)
            night_counts[d] = cnt
            print(f"{id_to_name[d]:15s}: {cnt} nocnych")

        # ===== C. LICZBA ZMIAN 24H =====
        print("\\n--- C. Liczba zmian 24-godzinnych ---")
        twf_counts = {}
        for d in D:
            cnt = sum(solver.Value(x[(d, s)]) for s in twentyfour_shifts)
            twf_counts[d] = cnt
            print(f"{id_to_name[d]:15s}: {cnt} × 24h")

        # ===== D. OBCIĄŻENIE ODDZIAŁ / ICU / PORADNIA =====
        print("\\n--- D. Obciążenie per oddział ---")
        for d in D:
            ward = sum(solver.Value(x[(d, s)]) for s in S if shifts.loc[shift_idx[s], "dept"] == "WARD")
            icu = sum(solver.Value(x[(d, s)]) for s in S if shifts.loc[shift_idx[s], "dept"] == "ICU")
            clinic = sum(solver.Value(x[(d, s)]) for s in S if shifts.loc[shift_idx[s], "dept"] == "CLINIC")

            print(f"{id_to_name[d]:15s}: Ward={ward}, ICU={icu}, Clinic={clinic}")

        # ===== E. SPEŁNIONE PREFERENCJE =====
        print("\\n--- E. Preferencje (like/dislike) ---")
        like_count = 0
        dislike_count = 0

        for _, row in pref.iterrows():
            d = row["doctor_id"]
            s = code_to_id[row["code"]]
            pref_type = row["preference"]

            if solver.Value(x[(d, s)]) == 1:
                if pref_type == "like":
                    like_count += 1
                else:
                    dislike_count += 1

        print(f"Spełnione like     : {like_count}")
        print(f"Naruszone dislike : {dislike_count}")

        # ===== F. FAIRNESS METRICS =====
        print("\\n--- F. Fairness (nocne) ---")
        print(f"max_nights = {solver.Value(max_nights)}")
        print(f"min_nights = {solver.Value(min_nights)}")
        print(f"spread     = {solver.Value(spread)}")

        # ===== G. POTENCJALNIE PRZEPRACOWANI =====
        print("\\n--- G. Lekarze potencjalnie przepracowani ---")

        for d in D:
            warnings = []

            # blisko limitu godzin
            if max_hours[d] > 48 and total_hours_worked[d] >= 0.9 * max_hours[d]:
                warnings.append("Blisko limitu godzin")

            # dużo zmian 24h
            if twf_counts[d] >= 2:
                warnings.append("Dużo zmian 24h")

            # dużo nocnych
            if night_counts[d] >= 3:
                warnings.append("Dużo zmian nocnych")

            # same nocne
            if night_counts[d] > 0 and night_counts[d] == sum(
                    solver.Value(x[(d, s)]) for s in S
            ):
                warnings.append("Same zmiany nocne")

            if warnings:
                print(f"{id_to_name[d]:15s}: " + ", ".join(warnings))
            else:
                print(f"{id_to_name[d]:15s}: OK")

    """ RESULTS """
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Objective value:", solver.ObjectiveValue())

        print_schedule()

        print("\n=== STATYSTYKI SOLVERA ===")
        print(f"Conflicts  : {solver.NumConflicts()}")
        print(f"Branches   : {solver.NumBranches()}")
        print(f"Wall time  : {solver.WallTime():.3f} s")
    else:
        print("Brak wykonalnego rozwiązania dla obecnych ograniczeń.")

    return status, schedule_df, stats_df, solver_stats_df

if __name__ == "__main__":
    status, schedule_df, stats_df, solver_stats_df = run_model_and_get_results()
    print("Status:", status)
    schedule_df.to_csv("schedule_output.csv", index=False, encoding="utf-8")