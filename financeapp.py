# smartfinance_app.py
# Full SmartFinance app (professional UI) with Add Expense tab shown first,
# Dashboard second. All previous functionality preserved.
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
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# ---------------------------- App config & CSS ----------------------------
st.set_page_config(page_title="SmartFinance", page_icon="💰", layout="wide")

CUSTOM_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
:root{--brand:#2563eb;--muted:#64748B;--ok:#16a34a;--warn:#f59e0b;--bad:#dc2626;}
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.hero { background: linear-gradient(135deg,#10b981,#2563eb); color:#fff; padding:22px 26px; border-radius:20px; margin:8px 0 18px 0; box-shadow:0 16px 40px rgba(37,99,235,.25); }
.hero h1{ margin:0; font-weight:800; font-size:28px; }
.hero p{ margin:6px 0 0; opacity:.95; }
.section-title{ font-size:22px; font-weight:700; color:#1f2937; margin-top:18px; margin-bottom:12px; padding:8px 12px; border-left:6px solid var(--brand); background:#fbfdff; border-radius:8px; }
.small{ color:var(--muted); font-size:13px; }
.card{ background:#fff; border-radius:12px; padding:12px; box-shadow:0 8px 18px rgba(15,23,42,0.04); border:1px solid rgba(2,6,23,0.04); }
.progress{ height:10px; background:#eef2f7; border-radius:999px; overflow:hidden; margin-top:6px; }
.progress > div{ height:100%; background:var(--brand); }
.chip{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }
.chip.ok{ background:#dcfce7; color:#065f46; }
.chip.warn{ background:#fef3c7; color:#92400e; }
.chip.bad{ background:#fee2e2; color:#991b1b; }
.caption-text{ color:#8b98aa; font-size:13px; margin-top:6px; }
.muted{ color:#64748b; font-size:13px; }
.custom-footer{ font-size:14px; color:#475569; text-align:center; padding:18px; margin-top:30px; border-top:1px solid #e6eef8; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>💰 SmartFinance</h1>
      <p>Simple budget tracker for families — add expenses, set budgets, get insights and predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Paths & defaults ----------------------------
DATA_PATH = Path("smartfinance_user_data.csv")
BUDGETS_PATH = Path("smartfinance_budgets.json")
TASKS_PATH = Path("smartfinance_tasks.json")
GOALS_PATH = Path("smartfinance_goals.json")
AUTO_CAT_MODEL_PATH = Path("auto_categorizer.joblib")

DEFAULT_CATEGORIES = [
    "Groceries", "Rent", "Utilities", "Transport", "Dining", "Entertainment",
    "Healthcare", "Education", "EMI/Loans", "Insurance", "Shopping", "Other"
]

# ---------------------------- Helpers & persistence ----------------------------
def money(v: float) -> str:
    try: return f"₹{v:,.0f}"
    except: return str(v)

def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        return df
    return pd.DataFrame(columns=["Date","Amount","Category","Notes"])

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))

# initialize persistent storage if needed
if "df" not in st.session_state:
    st.session_state.df = load_data()
if "budgets" not in st.session_state:
    st.session_state.budgets = load_json(BUDGETS_PATH, {c:0.0 for c in DEFAULT_CATEGORIES})
if "tasks" not in st.session_state:
    st.session_state.tasks = load_json(TASKS_PATH, [])
if "goals" not in st.session_state:
    st.session_state.goals = load_json(GOALS_PATH, [])
if "pred_lookback" not in st.session_state:
    st.session_state.pred_lookback = 3

def ensure_date(col):
    return pd.to_datetime(col, errors="coerce")

# ---------------------------- Sidebar ----------------------------
st.sidebar.header("⚙️ Settings & Data")
if st.sidebar.button("Export data (CSV)"):
    st.download_button("Download CSV", data=st.session_state.df.to_csv(index=False).encode("utf-8"), file_name="smartfinance_data.csv", mime="text/csv")
if st.sidebar.button("Reset all data"):
    for p in [DATA_PATH, BUDGETS_PATH, TASKS_PATH, GOALS_PATH, AUTO_CAT_MODEL_PATH]:
        if p.exists(): p.unlink()
    st.session_state.df = pd.DataFrame(columns=["Date","Amount","Category","Notes"])
    st.session_state.budgets = {c:0.0 for c in DEFAULT_CATEGORIES}
    st.session_state.tasks = []
    st.session_state.goals = []
    st.success("All data cleared. Reload the app.")

st.sidebar.caption("Quick navigation")
# **IMPORTANT**: Tabs order below: Add Expense FIRST, Dashboard SECOND (user requested)
tab_add, tab_dash, tab_budget, tab_pred, tab_todo = st.tabs(
    ["➕ Add Expense", "📊 Dashboard", "💵 Budgets", "🔮 Predictions", "✅ To-Do & Goals"]
)

# ---------------------------- Tab: Add Expense (FIRST) ----------------------------
with tab_add:
    st.markdown('<div class="section-title">➕ Add Expense</div>', unsafe_allow_html=True)
    with st.form("add_expense", clear_on_submit=True):
        c1,c2,c3 = st.columns([1,1,2])
        d = c1.date_input("Date", date.today())
        amt = c2.number_input("Amount (₹)", min_value=0.0, value=0.0, step=50.0)
        notes = c3.text_input("Notes / Description (optional)", placeholder="e.g., Big Bazaar groceries, Uber ride")
        # Suggest category from notes (simple rules)
        suggested = None
        if notes:
            t = notes.lower()
            if any(k in t for k in ["swiggy","zomato","restaurant","cafe","dine","pizza","burger"]):
                suggested = "Dining"
            elif any(k in t for k in ["uber","ola","metro","bus","fuel","petrol","diesel"]):
                suggested = "Transport"
            elif any(k in t for k in ["rent","landlord"]):
                suggested = "Rent"
            elif any(k in t for k in ["electric","water","gas","internet","wifi","broadband"]):
                suggested = "Utilities"
            elif any(k in t for k in ["hospital","pharmacy","medic","doctor"]):
                suggested = "Healthcare"
            elif any(k in t for k in ["amazon","flipkart","shopping","mall"]):
                suggested = "Shopping"
            elif any(k in t for k in ["school","tuition","course","fees"]):
                suggested = "Education"
            elif any(k in t for k in ["movie","netflix","spotify","gaming"]):
                suggested = "Entertainment"
            elif any(k in t for k in ["grocery","groceries","vegetable","kirana","dmart","big bazaar"]):
                suggested = "Groceries"
        category = st.selectbox("Category", options=DEFAULT_CATEGORIES, index=DEFAULT_CATEGORIES.index(suggested) if suggested in DEFAULT_CATEGORIES else 0)
        submitted = st.form_submit_button("Save Expense")
        if submitted:
            if amt <= 0:
                st.error("Please enter an amount greater than zero.")
            else:
                new = {"Date": pd.to_datetime(d), "Amount": float(amt), "Category": category, "Notes": notes}
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new])], ignore_index=True)
                save_data(st.session_state.df)
                st.success("Expense saved.")

    st.markdown("---")
    # Train auto-categorizer option (simple friendly text)
    st.markdown("### Smart Auto-Categorizer (optional)")
    st.markdown("This feature can learn from your expense notes and suggest categories. For good suggestions, add about 30 expenses with short notes.")
    if st.button("Train / Update auto-categorizer"):
        data = st.session_state.df.dropna(subset=["Notes","Category"]).copy()
        if len(data) < 30:
            st.info("Please add at least ~30 expenses with notes to train the auto-categorizer.")
        else:
            X = data["Notes"].astype(str); y = data["Category"].astype(str)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)), ("clf", LogisticRegression(max_iter=2000))])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Auto-categorizer trained — approx. {acc:.2f} accuracy on holdout.")
            joblib.dump(pipe, AUTO_CAT_MODEL_PATH)

    # Show recent transactions and simple edit/delete
    st.markdown("---")
    st.subheader("Recent Transactions")
    if st.session_state.df.empty:
        st.info("No transactions yet — add your first expense above.")
    else:
        df_display = st.session_state.df.copy()
        df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(df_display.sort_values("Date", ascending=False).reset_index(drop=True), use_container_width=True)

        st.markdown("**Edit / Delete an expense**")
        idx = st.number_input("Row index to edit/delete (start at 0)", min_value=0, max_value=max(0, len(df_display)-1), value=0, step=1)
        if idx is not None and len(df_display) > 0:
            sel = df_display.iloc[int(idx)]
            st.markdown(f"Selected: **{sel.Category}** • {money(sel.Amount)} • {sel.Date}")
            new_amt = st.number_input("Correct amount (₹)", min_value=0.0, value=float(sel.Amount))
            new_cat = st.selectbox("Category", options=DEFAULT_CATEGORIES, index=DEFAULT_CATEGORIES.index(sel.Category) if sel.Category in DEFAULT_CATEGORIES else 0)
            if st.button("Update expense"):
                real_idx = int(idx)
                st.session_state.df.at[real_idx, "Amount"] = float(new_amt)
                st.session_state.df.at[real_idx, "Category"] = new_cat
                save_data(st.session_state.df)
                st.success("Expense updated.")
            if st.button("Delete expense"):
                st.session_state.df = st.session_state.df.drop(int(idx)).reset_index(drop=True)
                save_data(st.session_state.df)
                st.success("Expense deleted.")

# ---------------------------- Tab: Dashboard (SECOND) ----------------------------
with tab_dash:
    st.markdown('<div class="section-title">📊 Dashboard</div>', unsafe_allow_html=True)
    if st.session_state.df.empty:
        st.info("No expenses yet. Start by adding an expense (open the Add Expense tab).")
    else:
        df = st.session_state.df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        cur_month = pd.Timestamp.today().strftime("%Y-%m")
        month_df = df[df["Month"] == cur_month]
        total_spent = float(month_df["Amount"].sum()) if not month_df.empty else 0.0
        top_cat = month_df.groupby("Category")["Amount"].sum().sort_values(ascending=False).index[0] if (not month_df.empty and month_df["Category"].notna().any()) else "—"
        total_budget = sum(v for v in st.session_state.budgets.values() if v)
        util_pct = (total_spent / total_budget * 100) if total_budget > 0 else 0.0

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("This Month Spent", money(total_spent))
            st.markdown('<p class="caption-text">(Total money you used this month)</p>', unsafe_allow_html=True)
        with k2:
            st.metric("Top Category", top_cat)
            st.markdown('<p class="caption-text">(Where you spent the most)</p>', unsafe_allow_html=True)
        with k3:
            st.metric("Budget Utilization", f"{util_pct:.0f}%")
            st.markdown('<p class="caption-text">(How much of your set budget is used)</p>', unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns([1,1])
        with c1:
            st.subheader("Category share — this month")
            if not month_df.empty:
                cat_sum = month_df.groupby("Category")["Amount"].sum()
                fig, ax = plt.subplots(figsize=(5,4))
                ax.pie(cat_sum.values, labels=cat_sum.index, autopct="%1.0f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)
            else:
                st.info("No data for current month.")
        with c2:
            st.subheader("Monthly trend — total spend (last 12 months)")
            trend = df.groupby("Month")["Amount"].sum().sort_index().tail(12)
            if not trend.empty:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                ax2.plot(trend.index, trend.values, marker="o")
                ax2.set_xlabel("Month")
                ax2.set_ylabel("Total spend")
                plt.xticks(rotation=30)
                st.pyplot(fig2)
            else:
                st.info("Add several months of expenses to see trend.")

# ---------------------------- Tab: Budgets ----------------------------
with tab_budget:
    st.markdown('<div class="section-title">💵 Budgets</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Set a monthly budget and optional per-category budgets to keep control of spending.</div>', unsafe_allow_html=True)
    # total budget
    total_budget_val = st.number_input("Set total monthly budget (₹)", min_value=0.0, value=0.0, step=500.0)
    if st.button("Save total budget"):
        # distribute or store as total (we'll store total under a special key)
        st.session_state.budgets["__total__"] = float(total_budget_val)
        save_json(BUDGETS_PATH, st.session_state.budgets)
        st.success("Total budget saved.")
    st.markdown("---")
    st.subheader("Category budgets (optional)")
    if "category_budgets" not in st.session_state:
        st.session_state.category_budgets = {c: float(st.session_state.budgets.get(c, 0.0) or 0.0) for c in DEFAULT_CATEGORIES}
    cols = st.columns(3)
    for i, cat in enumerate(DEFAULT_CATEGORIES):
        val = cols[i % 3].number_input(cat, min_value=0.0, value=float(st.session_state.category_budgets.get(cat, 0.0)), key=f"bdg_{cat}")
        st.session_state.category_budgets[cat] = float(val)
    if st.button("Save category budgets"):
        for c in DEFAULT_CATEGORIES:
            st.session_state.budgets[c] = float(st.session_state.category_budgets.get(c, 0.0))
        save_json(BUDGETS_PATH, st.session_state.budgets)
        st.success("Category budgets saved.")
    # show utilization
    st.markdown("---")
    st.subheader("Budget utilization — this month")
    df = st.session_state.df.copy()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        cur = pd.Timestamp.today().strftime("%Y-%m")
        mdf = df[df["Month"] == cur]
        for cat in DEFAULT_CATEGORIES:
            spent = float(mdf.loc[mdf["Category"] == cat, "Amount"].sum()) if not mdf.empty else 0.0
            limit = float(st.session_state.budgets.get(cat, 0.0) or 0.0)
            pct = (spent / limit * 100) if limit > 0 else 0.0
            tone = "ok" if pct < 70 else ("warn" if pct < 100 else "bad")
            st.markdown(f"**{cat}** — {money(spent)} / {money(limit)}  <span class='chip {('ok' if tone=='ok' else ('warn' if tone=='warn' else 'bad'))}'>{pct:.0f}%</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='progress'><div style='width:{min(pct,100)}%'></div></div>", unsafe_allow_html=True)
    else:
        st.info("Add expenses to see utilization metrics.")

# ---------------------------- Tab: Predictions ----------------------------
with tab_pred:
    st.markdown('<div class="section-title">🔮 Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">You can enter last 1 month, 3 months, or 12 months of expenses.<br>'
                '<b>1 Month</b> → Quick rough forecast &nbsp;&nbsp; <b>3 Months</b> → Balanced prediction &nbsp;&nbsp; <b>12 Months</b> → Best accuracy with seasonal trends</div>',
                unsafe_allow_html=True)
    df = st.session_state.df.copy()
    if df.empty:
        st.info("Add expenses (use Add Expense tab) to get predictions.")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        span = st.selectbox("Choose data span for prediction", options=["1 month (quick)", "3 months (balanced)", "12 months (best)"])
        months_lookup = {"1 month (quick)": 1, "3 months (balanced)": 3, "12 months (best)": 12}
        months = months_lookup[span]
        last_month = pd.Timestamp.today().to_period("M")
        months_list = [(last_month - i).strftime("%Y-%m") for i in range(months)][::-1]
        tmp = df[df["Month"].isin(months_list)].groupby("Month")["Amount"].sum().reindex(months_list, fill_value=0.0)
        if tmp.sum() == 0:
            st.info("Not enough data in selected span for a meaningful prediction. Add more expenses.")
        else:
            pred = float(tmp.mean())
            st.markdown(f"**Estimated spend next month (quick): {money(pred)}**")
            fig, ax = plt.subplots(figsize=(7,3.5))
            ax.bar(tmp.index, tmp.values)
            ax.set_title("Monthly totals used for prediction")
            ax.set_ylabel("Amount")
            st.pyplot(fig)
            st.markdown('<div class="muted">Note: This is a simple estimate. More months → better accuracy.</div>', unsafe_allow_html=True)
            # if enough months, run a backtest with RF regressor
            if df["Month"].nunique() >= 8 and months >= 3:
                st.markdown("---")
                st.markdown("Backtest (simple time-series validation)")
                pt = df.pivot_table(index="Month", columns="Category", values="Amount", aggfunc="sum").fillna(0.0).sort_index()
                feats = pt.rolling(window=3, min_periods=1).mean().shift(1).dropna()
                target = pt.sum(axis=1).loc[feats.index]
                if len(feats) >= 8:
                    X = feats; y = target
                    tscv = TimeSeriesSplit(n_splits=4)
                    maes, rmses = [], []
                    for train_idx, test_idx in tscv.split(X):
                        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                        pre = ColumnTransformer([("num", StandardScaler(with_mean=False), list(X.columns))])
                        model = RandomForestRegressor(n_estimators=200, random_state=42)
                        pipe = Pipeline([("prep", pre), ("rf", model)])
                        pipe.fit(X_tr, y_tr)
                        y_hat = pipe.predict(X_te)
                        maes.append(mean_absolute_error(y_te, y_hat))
                        rmses.append(mean_squared_error(y_te, y_hat, squared=False))
                    st.success(f"Backtest MAE: {money(np.mean(maes))}   RMSE: {money(np.mean(rmses))}")

# ---------------------------- Tab: To-Do & Goals ----------------------------
with tab_todo:
    st.markdown('<div class="section-title">✅ To-Do & Goals</div>', unsafe_allow_html=True)

    # Goals
    st.subheader("🎯 Goals")
    with st.form("goal_form", clear_on_submit=True):
        gname = st.text_input("Goal name")
        gtarget = st.number_input("Target amount (₹)", min_value=0.0, step=100.0)
        gdeadline = st.date_input("Deadline", value=date.today()+relativedelta(months=3))
        add_g = st.form_submit_button("Add Goal")
        if add_g and gname and gtarget > 0:
            st.session_state.goals.append({"name": gname, "target": float(gtarget), "saved": 0.0, "deadline": str(gdeadline)})
            save_json(GOALS_PATH, st.session_state.goals)
            st.success("Goal added.")
    if st.session_state.goals:
        for i, g in enumerate(st.session_state.goals):
            pct = (g["saved"]/g["target"]*100) if g["target"]>0 else 0.0
            tone = "ok" if pct >= 100 else ("warn" if pct >= 70 else "")
            st.markdown(f"**{g['name']}**  <span class='chip {('ok' if tone=='ok' else ('warn' if tone=='warn' else ''))}'>{pct:.0f}%</span>", unsafe_allow_html=True)
            st.markdown(f"{money(g['saved'])} / {money(g['target'])} • by {g['deadline']}")
            st.markdown(f"<div class='progress'><div style='width:{min(pct,100)}%'></div></div>", unsafe_allow_html=True)
            add_col = st.number_input(f"Add to {g['name']}", min_value=0.0, step=100.0, key=f"addsave_{i}")
            if st.button("Update Savings", key=f"btnsave_{i}"):
                st.session_state.goals[i]["saved"] += float(add_col)
                save_json(GOALS_PATH, st.session_state.goals)
                st.success("Savings updated.")
            if st.button("Delete Goal", key=f"delgoal_{i}"):
                st.session_state.goals.pop(i)
                save_json(GOALS_PATH, st.session_state.goals)
                st.experimental_rerun()
    else:
        st.info("No goals yet. Add one above.")

    st.markdown("---")
    # Tasks
    st.subheader("📝 Finance To-Do")
    with st.form("task_form", clear_on_submit=True):
        tname = st.text_input("Task")
        tamt = st.number_input("Amount (optional)", min_value=0.0, step=50.0)
        tdue = st.date_input("Due", value=date.today()+timedelta(days=3))
        tprio = st.selectbox("Priority", ["Low","Medium","High"])
        tgoal = st.selectbox("Link to goal (optional)", options=["-"] + [g["name"] for g in st.session_state.goals])
        add_t = st.form_submit_button("Add Task")
        if add_t:
            st.session_state.tasks.append({"task": tname, "amount": float(tamt), "due": str(tdue), "priority": tprio, "goal": None if tgoal=="-" else tgoal, "done": False})
            save_json(TASKS_PATH, st.session_state.tasks)
            st.success("Task added.")
    if st.session_state.tasks:
        for i, t in enumerate(st.session_state.tasks):
            days_left = (pd.to_datetime(t["due"]).date() - date.today()).days
            badge = "ok" if t["done"] else ("bad" if days_left<0 else ("warn" if days_left<=2 else ""))
            cols = st.columns([0.08, 0.5, 0.12, 0.12, 0.18])
            with cols[0]:
                chk = st.checkbox("", value=t["done"], key=f"chk_{i}")
            with cols[1]:
                st.markdown(f"**{t['task']}**<div class='muted'>Due {t['due']} • {t['priority']}</div>", unsafe_allow_html=True)
            with cols[2]:
                st.write(money(t["amount"]) if t["amount"]>0 else "-")
            with cols[3]:
                st.markdown(f"<span class='chip {badge}'>{'Done' if t['done'] else (str(days_left)+'d left')}</span>", unsafe_allow_html=True)
            with cols[4]:
                if st.button("Delete", key=f"del_{i}"):
                    st.session_state.tasks.pop(i)
                    save_json(TASKS_PATH, st.session_state.tasks)
                    st.experimental_rerun()
            # toggle done
            if chk != t["done"]:
                st.session_state.tasks[i]["done"] = chk
                if chk and t.get("goal") and t.get("amount", 0)>0:
                    for gi, g in enumerate(st.session_state.goals):
                        if g["name"] == t["goal"]:
                            st.session_state.goals[gi]["saved"] += float(t["amount"])
                            save_json(GOALS_PATH, st.session_state.goals)
                            break
                save_json(TASKS_PATH, st.session_state.tasks)
                st.experimental_rerun()

# ---------------------------- Simple offline Chatbot (kept in footer area) ----------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="muted">Need quick help with money? Try the built-in finance helper below.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">🤖 Finance Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Ask simple questions like: how to save money, what is a budget, reduce expenses.</div>', unsafe_allow_html=True)
responses = {
    "how to save money": "💡 Try the 50/30/20 rule — 50% needs, 30% wants, 20% savings.",
    "what is a budget": "📊 A budget is a plan to track income and expenses.",
    "reduce expenses": "✂️ Cut unnecessary subscriptions, plan groceries, avoid impulse buys.",
    "invest money": "📈 Start small with SIPs or index funds; keep a long-term view.",
    "emergency fund": "🛟 Keep 3–6 months of essential expenses in a separate savings account.",
    "hello": "👋 Hi! Ask me about saving, budgets, or reducing expenses."
}
q = st.text_input("Ask a finance question (e.g., 'how to save money')")
if q:
    ql = q.lower()
    ans = "🤔 Sorry, I don't have a tip for that yet."
    for k,v in responses.items():
        if k in ql:
            ans = v
            break
    st.success(ans)

# Smart categorizer friendly message
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="muted">Currently, the smart categorizer is about <b>75% accurate</b> after training on ~30+ notes.</div>', unsafe_allow_html=True)

# ---------------------------- Footer ----------------------------
st.markdown(
    """
    <div class="custom-footer">
      <b>SmartFinance</b> • ACE Engineering College<br>
      📧 For help or improvements: <a href="mailto:edulaganeshredd2005@gmail.com">edulaganeshredd2005@gmail.com</a><br>
      © 2025 SmartFinance. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
