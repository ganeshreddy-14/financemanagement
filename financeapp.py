# smartfinance_app.py
# SmartFinance — Professional, grid‑based Streamlit app for normal family users
# Features
# - No CSV required: users add expenses via a clean form; data is stored locally
# - Pro UI with CSS/HTML: grid dashboard, cards, progress bars, alerts
# - Visuals: spending by category, monthly trends, comparisons (matplotlib)
# - Budgets per category with utilization bars + warnings
# - To‑Do & Goals (link task amounts to goals on completion)
# - ML 1: Auto‑categorize from notes/description (TF‑IDF + LogisticRegression) => reports accuracy
# - ML 2: Forecast next month total spend (RandomForestRegressor) => MAE / RMSE backtest
# - Export/Import data, Reset
#
# Run:
#   pip install streamlit pandas numpy scikit-learn matplotlib python-dateutil joblib
#   streamlit run smartfinance_app.py

from __future__ import annotations
import json
from pathlib import Path
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# ----------------------------
# App Config & Styling
# ----------------------------
st.set_page_config(page_title="SmartFinance", page_icon="💰", layout="wide")

CUSTOM_CSS = r"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  :root{
    --bg:#f7f8fb; --card:#ffffff; --text:#0f172a; --muted:#64748B; --brand:#2563eb; --ok:#16a34a; --warn:#f59e0b; --bad:#dc2626;
  }
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
  .page { background: var(--bg); }
  .hero { background: linear-gradient(135deg,#10b981,#2563eb); color:#fff; padding:22px 26px; border-radius:20px; margin:8px 0 18px 0; box-shadow:0 16px 40px rgba(37,99,235,.25); }
  .hero h1{ margin:0; font-weight:800; font-size:28px; }
  .hero p{ margin:6px 0 0; opacity:.95; }
  .grid { display:grid; grid-template-columns: repeat(12, 1fr); gap:14px; }
  .card { background:var(--card); border:1px solid rgba(2,6,23,.06); border-radius:16px; padding:14px; box-shadow:0 8px 24px rgba(2,6,23,.06); }
  .kpi { display:flex; flex-direction:column; gap:6px; }
  .kpi .label{ color:var(--muted); font-size:12px; }
  .kpi .value{ font-size:24px; font-weight:800; }
  .chip{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }
  .chip.ok{ background:#dcfce7; color:#166534; }
  .chip.warn{ background:#fef9c3; color:#713f12; }
  .chip.bad{ background:#fee2e2; color:#7f1d1d; }
  .progress{ height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
  .progress > div{ height:100%; background:var(--brand); }
  .section-title{ margin:4px 0 6px; font-weight:800; }
  .btn-row{ display:flex; gap:8px; align-items:center; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>💰 SmartFinance</h1>
      <p>Simple budget tracker for families — add expenses, set budgets, get insights and predictions. No CSV required.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Paths & Defaults
# ----------------------------
DATA_PATH = Path("smartfinance_user_data.csv")
BUDGETS_PATH = Path("smartfinance_budgets.json")
TASKS_PATH = Path("smartfinance_tasks.json")
GOALS_PATH = Path("smartfinance_goals.json")
AUTO_CAT_MODEL_PATH = Path("auto_categorizer.joblib")

DEFAULT_CATEGORIES = [
    "Groceries", "Rent", "Utilities", "Transport", "Dining", "Entertainment",
    "Healthcare", "Education", "EMI/Loans", "Insurance", "Shopping", "Other"
]

# ----------------------------
# Helpers
# ----------------------------

def money(v: float) -> str:
    try: return f"₹{v:,.0f}"
    except: return str(v)

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])  # saved with Date
        return df
    # start with an empty dataset
    return pd.DataFrame(columns=["Date","Amount","Category","Notes"])  # Date as string initially

@st.cache_data(show_spinner=False)
def load_budgets() -> Dict[str, float]:
    if BUDGETS_PATH.exists():
        return json.loads(BUDGETS_PATH.read_text())
    # default reasonable monthly budgets
    return {c: 0.0 for c in DEFAULT_CATEGORIES}

@st.cache_data(show_spinner=False)
def load_tasks() -> List[dict]:
    if TASKS_PATH.exists():
        return json.loads(TASKS_PATH.read_text())
    return []

@st.cache_data(show_spinner=False)
def load_goals() -> List[dict]:
    if GOALS_PATH.exists():
        return json.loads(GOALS_PATH.read_text())
    return []


def save_data(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)
    load_data.clear()  # invalidate cache


def save_budgets(budgets: Dict[str, float]):
    BUDGETS_PATH.write_text(json.dumps(budgets, indent=2))
    load_budgets.clear()


def save_tasks(tasks: List[dict]):
    TASKS_PATH.write_text(json.dumps(tasks, indent=2))
    load_tasks.clear()


def save_goals(goals: List[dict]):
    GOALS_PATH.write_text(json.dumps(goals, indent=2))
    load_goals.clear()


def ensure_date(col):
    return pd.to_datetime(col, errors='coerce')


def month_key(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m")


# ----------------------------
# Data
# ----------------------------
df = load_data()
if not df.empty:
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = ensure_date(df["Date"])  # parse

budgets = load_budgets()
tasks = load_tasks()
goals = load_goals()

# ----------------------------
# Sidebar Quick Actions
# ----------------------------
st.sidebar.header("⚙️ Settings & Data")
with st.sidebar:
    if st.button("Export data as CSV"):
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="smartfinance_data.csv", mime="text/csv")
    st.markdown("---")
    if st.button("Reset all data", type="secondary"):
        if DATA_PATH.exists(): DATA_PATH.unlink()
        if BUDGETS_PATH.exists(): BUDGETS_PATH.unlink()
        if TASKS_PATH.exists(): TASKS_PATH.unlink()
        if GOALS_PATH.exists(): GOALS_PATH.unlink()
        if AUTO_CAT_MODEL_PATH.exists(): AUTO_CAT_MODEL_PATH.unlink()
        load_data.clear(); load_budgets.clear(); load_tasks.clear(); load_goals.clear()
        st.success("All data cleared. Please reload the app.")

# ----------------------------
# Tabs
# ----------------------------
T1, T2, T3, T4, T5 = st.tabs(["📊 Dashboard", "➕ Add Expense", "💵 Budgets", "🔮 Predictions", "✅ To‑Do & Goals"])

# ----------------------------
# Tab: Dashboard
# ----------------------------
with T1:
    # KPI grid
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="grid">', unsafe_allow_html=True)
        # compute KPIs for current month
        today = pd.Timestamp.today()
        cur_month = today.strftime("%Y-%m")
        if not df.empty:
            df["Month"] = df["Date"].dt.to_period('M').astype(str)
            mdf = df[df["Month"] == cur_month]
            total_spent = float(mdf["Amount"].sum()) if not mdf.empty else 0.0
            top_cat = mdf.groupby("Category")["Amount"].sum().sort_values(ascending=False)
            top_cat_name = (top_cat.index[0] if len(top_cat)>0 else "—")
            # budgets
            total_budget = sum(v for v in budgets.values() if v)
            util_pct = (total_spent/total_budget*100) if total_budget>0 else 0
        else:
            total_spent = 0.0; top_cat_name = "—"; util_pct = 0

        # KPI cards as 12‑column grid (3x)
        k1 = f"""
        <div class='card kpi' style='grid-column: span 4;'>
          <div class='label'>This Month Spent</div>
          <div class='value'>{money(total_spent)}</div>
        </div>"""
        k2 = f"""
        <div class='card kpi' style='grid-column: span 4;'>
          <div class='label'>Top Category</div>
          <div class='value'>{top_cat_name}</div>
        </div>"""
        chip = "ok" if util_pct < 70 else ("warn" if util_pct < 100 else "bad")
        k3 = f"""
        <div class='card kpi' style='grid-column: span 4;'>
          <div class='label'>Budget Utilization</div>
          <div class='value'>{util_pct:.0f}% <span class='chip {chip}'>{'On track' if chip=='ok' else ('Watch' if chip=='warn' else 'Exceeded')}</span></div>
        </div>"""
        st.markdown(k1+k2+k3, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Chart grid 2x2
    with st.container():
        st.markdown('<div class="grid">', unsafe_allow_html=True)

        # Pie: category share (current month)
        st.markdown('<div class="card" style="grid-column: span 6;">', unsafe_allow_html=True)
        st.subheader("Category share — this month")
        if not df.empty and not mdf.empty:
            cat_sum = mdf.groupby("Category")["Amount"].sum()
            fig = plt.figure(figsize=(4.8,4.0))
            plt.pie(cat_sum.values, labels=cat_sum.index, autopct='%1.0f%%')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Add some expenses to see category shares.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Line: month trends (last 12)
        st.markdown('<div class="card" style="grid-column: span 6;">', unsafe_allow_html=True)
        st.subheader("Monthly trend — total spend")
        if not df.empty:
            tmp = df.copy()
            tmp["Month"] = tmp["Date"].dt.to_period('M').astype(str)
            trend = tmp.groupby("Month")["Amount"].sum().sort_index().tail(12)
            fig2, ax2 = plt.subplots(figsize=(5.6,3.8))
            ax2.plot(trend.index, trend.values, marker='o')
            ax2.set_xlabel("Month"); ax2.set_ylabel("Total spend"); plt.xticks(rotation=30, ha='right')
            st.pyplot(fig2)
        else:
            st.info("Trend will appear after you add expenses.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Bar: category compare (this vs last month)
        st.markdown('<div class="card" style="grid-column: span 6;">', unsafe_allow_html=True)
        st.subheader("Category compare — this vs last month")
        if not df.empty:
            tmp = df.copy(); tmp["Month"] = tmp["Date"].dt.to_period('M').astype(str)
            this_m = tmp[tmp["Month"]==cur_month].groupby("Category")["Amount"].sum()
            last_m_key = (pd.Timestamp(today.year, today.month, 1)-relativedelta(months=1)).strftime("%Y-%m")
            last_m = tmp[tmp["Month"]==last_m_key].groupby("Category")["Amount"].sum()
            cats = sorted(set(this_m.index).union(last_m.index))
            this_vals = [this_m.get(c,0.0) for c in cats]
            last_vals = [last_m.get(c,0.0) for c in cats]
            x = np.arange(len(cats))
            fig3, ax3 = plt.subplots(figsize=(5.6,3.8))
            ax3.bar(x-0.2, last_vals, width=0.4, label='Last')
            ax3.bar(x+0.2, this_vals, width=0.4, label='This')
            ax3.set_xticks(x); ax3.set_xticklabels(cats, rotation=30, ha='right'); ax3.legend()
            st.pyplot(fig3)
        else:
            st.info("Add two months of data to compare.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Info card: tips/prediction teaser
        st.markdown('<div class="card" style="grid-column: span 6;">', unsafe_allow_html=True)
        st.subheader("Smart tip ✨")
        if total_spent == 0:
            st.write("Start by logging daily expenses. Small consistency builds powerful insights.")
        else:
            st.write("Consider reviewing your top category and setting a tighter budget if needed.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Tab: Add Expense
# ----------------------------
with T2:
    st.markdown('<div class="section-title">Add a new expense</div>', unsafe_allow_html=True)
    with st.form("add_expense", clear_on_submit=True):
        c1,c2,c3 = st.columns([1,1,2])
        amount = c1.number_input("Amount (₹)", min_value=0.0, value=0.0, step=50.0)
        d = c2.date_input("Date", value=date.today())
        # auto‑suggest category using rules from notes
        notes = c3.text_input("Notes / Description", placeholder="e.g., Big Bazaar groceries, Uber ride, Netflix…")
        # initial category
        suggested = ""
        if notes:
            text = notes.lower()
            if any(k in text for k in ["swiggy","zomato","restaurant","cafe","dine","pizza","burger"]): suggested = "Dining"
            elif any(k in text for k in ["uber","ola","metro","bus","fuel","petrol","diesel"]): suggested = "Transport"
            elif any(k in text for k in ["rent","landlord"]): suggested = "Rent"
            elif any(k in text for k in ["electric","water","gas","internet","wifi","broadband"]): suggested = "Utilities"
            elif any(k in text for k in ["hospital","pharmacy","medic","doctor"]): suggested = "Healthcare"
            elif any(k in text for k in ["amazon","flipkart","shopping","mall"]): suggested = "Shopping"
            elif any(k in text for k in ["school","tuition","course","fees"]): suggested = "Education"
            elif any(k in text for k in ["movie","netflix","spotify","gaming"]): suggested = "Entertainment"
            elif any(k in text for k in ["grocery","groceries","vegetable","kirana","big bazaar","dmart"]): suggested = "Groceries"
        category = st.selectbox("Category", options=DEFAULT_CATEGORIES, index=(DEFAULT_CATEGORIES.index(suggested) if suggested in DEFAULT_CATEGORIES else 0))
        add = st.form_submit_button("Save Expense")
        if add:
            new_row = {"Date": pd.to_datetime(d), "Amount": float(amount), "Category": category, "Notes": notes}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_data(df)
            st.success("Expense saved.")

    st.markdown("---")
    st.subheader("Optional: Train Auto‑Categorizer (ML)")
    st.caption("Uses TF‑IDF + LogisticRegression on your notes to predict categories. Shows accuracy on a test split.")
    if len(df) >= 30 and df["Notes"].notna().sum() >= 20:
        train_btn = st.button("Train/Update categorizer")
        if train_btn:
            data = df.dropna(subset=["Notes","Category"]).copy()
            X = data["Notes"].astype(str)
            y = data["Category"].astype(str)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
                ("clf", LogisticRegression(max_iter=2000))
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Categorizer accuracy: {acc:.2f}")
            st.text(classification_report(y_test, preds))
            joblib.dump(pipe, AUTO_CAT_MODEL_PATH)
            st.caption("Model saved for future auto‑suggestions.")
    else:
        st.info("Add at least ~30 expenses with notes to train the categorizer and see accuracy.")

# ----------------------------
# Tab: Budgets
# ----------------------------
with T3:
    st.markdown('<div class="section-title">Monthly budgets</div>', unsafe_allow_html=True)
    # edit budgets in a grid
    cols = st.columns(3)
    updated = False
    for i, cat in enumerate(DEFAULT_CATEGORIES):
        with cols[i%3]:
            cur = float(budgets.get(cat, 0.0) or 0.0)
            val = st.number_input(f"{cat}", min_value=0.0, value=cur, step=500.0, key=f"bdg_{cat}")
            if val != cur:
                budgets[cat] = float(val)
                updated = True
    if updated:
        save_budgets(budgets)
        st.success("Budgets updated.")

    st.markdown("---")
    st.subheader("Utilization this month")
    if not df.empty:
        df["Month"] = df["Date"].dt.to_period('M').astype(str)
        cur = pd.Timestamp.today().strftime("%Y-%m")
        mdf = df[df["Month"]==cur]
        for cat in DEFAULT_CATEGORIES:
            cat_spend = float(mdf[mdf["Category"]==cat]["Amount"].sum())
            b = float(budgets.get(cat,0.0) or 0.0)
            pct = (cat_spend/b*100) if b>0 else 0
            tone = 'ok' if pct<70 else ('warn' if pct<100 else 'bad')
            st.markdown(f"**{cat}** — {money(cat_spend)} / {money(b)}  <span class='chip {tone}'>{pct:.0f}%</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='progress'><div style='width:{min(pct,100)}%'></div></div>", unsafe_allow_html=True)
    else:
        st.info("Add expenses to see utilization.")

# ----------------------------
# Tab: Predictions (Forecast next month total)
# ----------------------------
with T4:
    st.markdown('<div class="section-title">Forecast next month — total spend</div>', unsafe_allow_html=True)
    if len(df) >= 12:
        tmp = df.copy()
        tmp["Month"] = tmp["Date"].dt.to_period('M').astype(str)
        # create monthly totals per category
        pt = tmp.pivot_table(index="Month", columns="Category", values="Amount", aggfunc="sum").fillna(0.0)
        pt.sort_index(inplace=True)
        # features: last 3 months rolling means for each category
        feats = pt.rolling(window=3, min_periods=1).mean().shift(1)  # avoid leakage
        target = pt.sum(axis=1)  # total spend
        data = feats.copy(); data["y"] = target
        data = data.dropna()  # after shift
        if len(data) >= 8:
            X = data.drop(columns=["y"]) ; y = data["y"]
            # time series split backtest
            tscv = TimeSeriesSplit(n_splits=4)
            maes, rmses = [], []
            preds_all, idx_all = [], []
            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                num_cols = list(X.columns)
                pre = ColumnTransformer([("num", StandardScaler(with_mean=False), num_cols)])
                model = RandomForestRegressor(n_estimators=300, random_state=42)
                pipe = Pipeline([("prep", pre), ("rf", model)])
                pipe.fit(X_tr, y_tr)
                y_hat = pipe.predict(X_te)
                preds_all.extend(list(y_hat)); idx_all.extend(list(X_te.index))
                maes.append(mean_absolute_error(y_te, y_hat))
                rmses.append(mean_squared_error(y_te, y_hat, squared=False))
            mae, rmse = float(np.mean(maes)), float(np.mean(rmses))
            st.success(f"Backtest — MAE: {money(mae)}   |   RMSE: {money(rmse)}")

            # Train on all & predict next month
            pipe.fit(X, y)
            next_month = (pd.Period(pt.index[-1], freq='M') + 1).strftime('%Y-%m')
            last_feats = feats.iloc[[-1]]
            next_pred = float(pipe.predict(last_feats)[0])
            st.markdown(f"**Predicted total spend for {next_month}: {money(next_pred)}**")

            # Plot actual vs backtest preds
            ser_pred = pd.Series(preds_all, index=idx_all).sort_index()
            common = y.loc[ser_pred.index]
            fig4, ax4 = plt.subplots(figsize=(6.2,3.8))
            ax4.plot(common.index, common.values, marker='o', label='Actual')
            ax4.plot(ser_pred.index, ser_pred.values, marker='o', label='Predicted')
            ax4.set_xlabel('Month'); ax4.set_ylabel('Total spend'); plt.xticks(rotation=30, ha='right'); ax4.legend()
            st.pyplot(fig4)
        else:
            st.info("Need at least ~8 months of data after feature prep to run a meaningful backtest.")
    else:
        st.info("Add at least 12 months of expenses for forecasting.")

# ----------------------------
# Tab: To‑Do & Goals
# ----------------------------
with T5:
    cA, cB = st.columns([1,1])
    with cA:
        st.subheader("To‑Do (Finance)")
        with st.form("todo_form", clear_on_submit=True):
            t1, t2 = st.columns([2,1])
            task = t1.text_input("Task", placeholder="Pay electricity bill")
            amount = t2.number_input("Amount (₹)", min_value=0.0, value=0.0, step=100.0)
            d1, d2, d3 = st.columns([1,1,1])
            due = d1.date_input("Due", value=date.today()+timedelta(days=3))
            pr = d2.selectbox("Priority", ["Low","Medium","High"], index=1)
            goal_link = d3.selectbox("Link Goal", options=["-"]+[g.get('name') for g in goals])
            add_t = st.form_submit_button("Add Task")
            if add_t and task:
                tasks.append({
                    "task": task, "amount": float(amount), "due": str(due), "priority": pr,
                    "goal": (None if goal_link=='-' else goal_link), "done": False,
                    "created_at": datetime.now().isoformat(timespec='seconds')
                })
                save_tasks(tasks)
                st.success("Task added.")
        # Render tasks
        if tasks:
            prio_rank = {"High":0, "Medium":1, "Low":2}
            tasks_sorted = sorted(tasks, key=lambda x: (x.get("done", False), prio_rank.get(x.get("priority","Medium"),1), x.get("due","9999-12-31")))
            for i, t in enumerate(tasks_sorted):
                days_left = (pd.to_datetime(t['due']).date() - date.today()).days
                badge = 'ok' if t.get('done') else ('bad' if days_left<0 else ('warn' if days_left<=2 else ''))
                col1, col2, col3, col4, col5 = st.columns([0.08, 0.42, 0.18, 0.14, 0.18])
                with col1:
                    done = st.checkbox("", value=t.get("done", False), key=f"chk_{i}")
                with col2:
                    st.markdown(f"**{t['task']}**\n\n<span style='color:#64748B;font-size:12px;'>Due {t['due']} • {t['priority']}</span>", unsafe_allow_html=True)
                with col3:
                    st.write(money(t.get('amount',0)))
                with col4:
                    st.markdown(f"<span class='chip {badge}'> {'Done' if done else (str(days_left)+'d left') }</span>", unsafe_allow_html=True)
                with col5:
                    if st.button("Delete", key=f"del_{i}"):
                        real_i = tasks.index(t)
                        tasks.pop(real_i)
                        save_tasks(tasks)
                        st.experimental_rerun()
                if done != t.get("done", False):
                    real_i = tasks.index(t)
                    tasks[real_i]["done"] = done
                    # if linked to goal and has amount, add to goal savings on completion
                    if done and t.get("goal") and t.get("amount",0)>0:
                        for gi, g in enumerate(goals):
                            if g.get("name") == t["goal"]:
                                goals[gi]["saved"] = float(goals[gi].get("saved",0))+float(t["amount"]) 
                                save_goals(goals)
                                break
                    save_tasks(tasks)
                    st.experimental_rerun()
        else:
            st.info("No tasks yet.")

    with cB:
        st.subheader("Goals")
        with st.form("goal_form", clear_on_submit=True):
            g1, g2, g3 = st.columns([2,1,1])
            name = g1.text_input("Goal name", placeholder="Emergency Fund")
            target = g2.number_input("Target (₹)", min_value=0.0, value=10000.0, step=500.0)
            deadline = g3.date_input("Deadline", value=date.today()+relativedelta(months=6))
            add_g = st.form_submit_button("Add Goal")
            if add_g and name:
                goals.append({"name":name, "target":float(target), "saved":0.0, "deadline": str(deadline)})
                save_goals(goals)
                st.success("Goal added.")
        # Render goals
        if goals:
            for i, g in enumerate(goals):
                pct = (g.get('saved',0)/g.get('target',1)*100) if g.get('target',0)>0 else 0
                tone = 'ok' if pct>=100 else ('warn' if pct>=70 else '')
                st.markdown(f"**{g['name']}**  <span class='chip {tone}'>{pct:.0f}%</span>", unsafe_allow_html=True)
                st.markdown(f"{money(g.get('saved',0))} / {money(g.get('target',0))} • by {g.get('deadline','—')}")
                st.markdown(f"<div class='progress'><div style='width:{min(pct,100)}%'></div></div>", unsafe_allow_html=True)
                c1, c2 = st.columns([1,1])
                add_amt = c1.number_input(f"Add savings to {g['name']}", min_value=0.0, value=0.0, step=500.0, key=f"addsv_{i}")
                if c2.button("Update", key=f"upg_{i}"):
                    goals[i]['saved'] = float(goals[i].get('saved',0))+float(add_amt)
                    save_goals(goals)
                    st.experimental_rerun()
                if st.button("Delete Goal", key=f"delg_{i}"):
                    goals.pop(i)
                    save_goals(goals)
                    st.experimental_rerun()
        else:
            st.info("No goals yet.")

# ----------------------------
# Footer
# ----------------------------
st.caption("SmartFinance • Built with ❤️ for families • Streamlit • pandas • scikit‑learn • matplotlib")
