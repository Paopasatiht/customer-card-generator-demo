# streamlit_app.py
import json, re
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------- helpers ----------------
@st.cache_data(show_spinner=False)
def load_jsonl(path: str):
    p = Path(path)
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def _maybe_parse_json(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val

def index_by_customer(records):
    idx = {}
    for rec in records:
        if isinstance(rec, dict) and len(rec) == 1:
            name, payload = next(iter(rec.items()))
            if isinstance(payload, dict):
                for k in ["persona_agent","mutual_fund_agent","economic_and_news_agent","customer_card"]:
                    payload[k] = _maybe_parse_json(payload.get(k, {})) or {}
            idx[str(name)] = payload
    return idx

def overview_df(indexed):
    rows=[]
    for name, p in indexed.items():
        gi = (p or {}).get("general_information", {})
        rows.append({
            "customer": name,
            "persona_flag": gi.get("persona_flag"),
            "fund": gi.get("external_fund") or gi.get("fundcode"),
            "aum": gi.get("aum"),
            "unrealized": gi.get("unrealized"),
        })
    return pd.DataFrame(rows)

def parse_performance_strings(perf_list):
    """Accepts '2025 (YTD): +3.33%' -> (years, values, labels)"""
    years, values, labels = [], [], []
    for s in perf_list or []:
        m = re.match(r'\s*(\d{4})\s*(?:\(([^)]*)\))?\s*:\s*([+-]?\d+(?:\.\d+)?)\s*%', s)
        if not m:
            continue
        year = int(m.group(1))
        suffix = m.group(2)  # e.g., "YTD"
        val = float(m.group(3))
        years.append(year)
        values.append(val)
        labels.append(f"{year}" + (f" ({suffix})" if suffix else ""))
    return years, values, labels

def numbered_with_reasons_md(items, reasons):
    """
    Render:
    1. item
       (because) reason

    ใช้ 'สองช่องว่าง + \\n' เพื่อบังคับขึ้นบรรทัดใน markdown list
    """
    items = items or []
    reasons = reasons or []
    out = []
    for i, it in enumerate(items, 1):
        if i-1 < len(reasons) and reasons[i-1]:
            out.append(f"{i}. {it}  \n   (because) {reasons[i-1]}")
        else:
            out.append(f"{i}. {it}")
    return "\n\n".join(out) if out else "—"

# ---------------- page config & styles ----------------
st.set_page_config(page_title="Customer Insights", layout="wide")
st.markdown("""
<style>
.card {padding:1rem 1.1rem;border:1px solid #e5e7eb;border-radius:16px;background:#ffffff;}
.kpi {padding:0.8rem 1rem;border:1px solid #eef2f7;border-radius:14px;background:#fafafb;}
.h {font-weight:700;margin-bottom:0.35rem;}
.h2 {font-weight:700;font-size:1.05rem;margin-bottom:0.5rem;}
.muted {color:#667085;}
hr {border:none;border-top:1px solid #eee;margin:1rem 0;}
</style>
""", unsafe_allow_html=True)

# ---------------- sidebar ----------------
st.title("📊 Customer Insights")
path = st.sidebar.text_input("Data file (.jsonl / .ndjson)", "./data/json_database.jsonl")

p = Path(path)
if not p.exists():
    st.warning("File not found. Point to your saved `.jsonl` file.")
    st.stop()

records = load_jsonl(path)
indexed = index_by_customer(records)
names = sorted(indexed.keys())
if not names:
    st.info("No customers found."); st.stop()

# q = st.sidebar.text_input("Filter customers")
# if q:
#     names = [n for n in names if q.lower() in n.lower()]
selected = st.sidebar.selectbox("Select customer", names, index=0)

# ---------------- overview table ----------------
st.subheader("Overview")
st.dataframe(overview_df(indexed), use_container_width=True, hide_index=True)

# ---------------- details ----------------
payload = indexed[selected]
gi = (payload or {}).get("general_information", {})

st.markdown(f"### Details — **{selected}**")

# KPI row (แสดงเฉพาะที่มีค่า ไม่เรนเดอร์กล่องเปล่า)
metrics = [
    ("Persona", gi.get("persona_flag")),
    ("Fund", gi.get("external_fund") or (gi.get("fundcode") if gi.get("fundcode") is not None else None)),
    ("AUM", gi.get("aum")),
    ("Unrealized", gi.get("unrealized")),
]
metrics = [(t,v) for t,v in metrics if v not in (None, "", "-", "None")]

if metrics:
    cols = st.columns(len(metrics))
    for col, (t, v) in zip(cols, metrics):
        col.markdown(f"<div class='kpi'><div class='h'>{t}</div>{v}</div>", unsafe_allow_html=True)

st.write("")  # spacing

# content rows
c1, c2 = st.columns([1.25, 1.75])

persona_agent = payload.get("persona_agent", {})
mutual_agent  = payload.get("mutual_fund_agent", {})
econ_agent    = payload.get("economic_and_news_agent", {})
customer_card = payload.get("customer_card", {})

# LEFT: Portfolio Status + News (ซ่อนการ์ดถ้าไม่มีเนื้อหา)
left_has_any = bool(customer_card.get("portfolio_current_status")) or bool(econ_agent.get("news_that_relate_with_industry")) or bool(econ_agent.get("materials"))

if left_has_any:
    with c1:
        if customer_card.get("portfolio_current_status"):
            st.markdown("<div class='h2'>📌 Portfolio Status</div>", unsafe_allow_html=True)
            st.markdown(customer_card.get("portfolio_current_status"))
        if econ_agent.get("news_that_relate_with_industry"):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='h2'>📰 News Highlights</div>", unsafe_allow_html=True)
            st.markdown(econ_agent.get("news_that_relate_with_industry"))
        mats = econ_agent.get("materials") or []
        if mats:
            st.markdown("<div class='muted' style='margin-top:.4rem;'>Sources:</div>", unsafe_allow_html=True)
            for m in mats:
                st.markdown(f"- {m}")
        st.markdown("</div>", unsafe_allow_html=True)

# RIGHT: talking points / CTA / Fund Snapshot (ซ่อนการ์ดถ้าไม่มีอะไรเลย)
tp  = customer_card.get("talking_point") or mutual_agent.get("talking_point") or []
rtp = customer_card.get("reason_of_each_talking_point") or mutual_agent.get("reason_of_each_talking_point") or []
cta  = customer_card.get("call_to_action") or []
rcta = customer_card.get("reason_of_each_call_to_action") or []
perf_lines = mutual_agent.get("last_3_year_performance_of_the_fund") or []
right_has_any = bool(tp or cta or perf_lines)

if right_has_any:
    with c2:
        if tp:
            st.markdown("<div class='h2'>💬 Talking Points</div>", unsafe_allow_html=True)
            st.markdown(numbered_with_reasons_md(tp, rtp))

        if cta:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='h2'>✅ Call to Action</div>", unsafe_allow_html=True)
            st.markdown(numbered_with_reasons_md(cta, rcta))

        if perf_lines:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='h2'>📈 Fund Snapshot</div>", unsafe_allow_html=True)
            years, values, labels = parse_performance_strings(perf_lines)

            if years:
                # 👇 ซับคอลัมน์: ซ้ายเป็นกราฟ / ขวาเป็นรายละเอียด
                gcol, tcol = st.columns([2, 1], gap="large")

                with gcol:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=years, y=values,
                        mode="lines+markers+text",
                        text=[f"{v:+.2f}%" for v in values],
                        textposition="top center",
                        hovertemplate="<b>%{customdata}</b><br>Return: %{y:.2f}%<extra></extra>",
                        customdata=labels
                    ))
                    fig.add_hline(y=0, line_width=1)

                    # ---- default "zoom out once" (≈ 1.2x) ----
                    # X axis
                    x_min, x_max = min(years), max(years)
                    x_mid  = (x_min + x_max) / 2
                    x_half = max((x_max - x_min) / 2, 0.5) * 1.2   # กันกรณีปีเดียว ให้มีที่ว่าง
                    x_range = [x_mid - x_half, x_mid + x_half]

                    # Y axis
                    y_min, y_max = min(values), max(values)
                    y_mid  = (y_min + y_max) / 2
                    y_half = max((y_max - y_min) / 2, 1.0) * 1.2   # กันกรณีค่าเดียว ให้มีที่ว่าง
                    y_range = [y_mid - y_half, y_mid + y_half]
                    # -----------------------------------------

                    fig.update_layout(
                        height=300,
                        margin=dict(l=40, r=20, t=30, b=40),
                        xaxis=dict(title="Year", tickmode="array", tickvals=years, ticktext=labels, range=x_range),
                        yaxis=dict(title="Return (%)", range=y_range),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tcol:
                    # รายละเอียดขวากราฟ (บรรทัดละบูลเล็ต)
                    ki = mutual_agent.get("key_insight_vs_benchmark")
                    what = mutual_agent.get("what_this_fund_invest")
                    best_idx = int(max(range(len(values)), key=lambda i: values[i])) if values else None
                    worst_idx = int(min(range(len(values)), key=lambda i: values[i])) if values else None

                    detail_lines = []
                    if ki:   detail_lines.append(f"- **Benchmark insight:** {ki}")
                    if what: detail_lines.append(f"- **Invests in:** {what}")
                    if best_idx is not None:  detail_lines.append(f"- **Best:** {labels[best_idx]} at {values[best_idx]:+.2f}%")
                    if worst_idx is not None: detail_lines.append(f"- **Worst:** {labels[worst_idx]} at {values[worst_idx]:+.2f}%")

                    if detail_lines:
                        st.markdown("\n".join(detail_lines))
            else:
                st.caption("No performance data available.")

        st.markdown("</div>", unsafe_allow_html=True)
