import pandas as pd
import streamlit as st
import portable

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

st.set_page_config(page_title="Study office support list", layout="wide")
st.title("Study office · week 6 support list")
st.caption("A prioritisation aid for supportive outreach—not an automatic decision about a student.")

@st.cache_resource
def load_model():
    return portable.Model("model")

@st.cache_data
def load_data():
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    return history, new

model = load_model()
history, new = load_data()
new["risk"] = model.predict_proba(new)
val = history.query("cohort == 2025").copy()
val["risk"] = model.predict_proba(val)

capacity = st.sidebar.slider("Conversations available", 1, 100, 40)
top_new = new.nlargest(capacity, "risk").copy()
top_new["priority"] = "Contact"

st.subheader("This week's ranked list")
st.caption(f"The {capacity} highest-risk students are marked for an adviser review.")
st.dataframe(top_new[["priority", "student_id", "programme", "international", "fees_owed", "submitted_share", "missed_last3", "weeks_since_login", "risk"]],
             hide_index=True, column_config={"risk": st.column_config.ProgressColumn("Predicted risk", min_value=0, max_value=1, format="%.0%%")})

st.subheader("What this rule did on the 2025 cohort")
val["contacted"] = val["risk"].rank(ascending=False, method="first") <= capacity
tp = int((val["contacted"] & (val["left"] == 1)).sum())
fp = int((val["contacted"] & (val["left"] == 0)).sum())
fn = int((~val["contacted"] & (val["left"] == 1)).sum())
tn = int((~val["contacted"] & (val["left"] == 0)).sum())
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reached in time", tp)
c2.metric("Worried unnecessarily", fp)
c3.metric("Missed", fn)
c4.metric("Left uncontacted", tn)
st.write(f"Of contacted students, **{precision:.0%}** later left (precision). The list reached **{recall:.0%}** of students who later left (recall).")

st.subheader("Check outcomes by group")
groups = val.groupby("international").apply(lambda g: pd.Series({
    "students": len(g), "share who left": g["left"].mean(), "mean predicted risk": g["risk"].mean(),
    "recall": g.loc[g["left"] == 1, "contacted"].mean()
}), include_groups=False)
groups.index = groups.index.map({0: "Domestic", 1: "International"})
st.dataframe(groups.style.format({"share who left": "{:.1%}", "mean predicted risk": "{:.1%}", "recall": "{:.1%}"}))

st.subheader("Conversation value")
help_rate = st.slider("Share of conversations expected to help", 0.0, 1.0, 0.30, 0.05)
benefit = tp * help_rate * 60_000 - capacity * 500 - fp * 2_000
st.write(f"Using the assignment assumptions, this rule's estimated value is **{benefit:,.0f} DKK**. Adjust the assumption to support a policy discussion, not to automate the decision.")
