# streamlit_app.py
import json, re
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------- helpers ----------------
@st.cache_data(show_spinner=False)
def load_records_any(path: str):
    """
    รองรับ:
      1) JSONL: 1 object ต่อบรรทัด
      2) JSON: เป็น list ของ objects
      3) Stream: อ็อบเจกต์หลายก้อนวางต่อกัน (ไม่อยู่ใน list)
    หมายเหตุ: ถ้ามีเครื่องหมาย " ไม่ได้ escape ภายในสตริง -> JSON จะไม่ถูกต้อง ต้องแก้ที่ไฟล์
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    # 2) ลอง JSON array ตรง ๆ ก่อน
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # เผื่อเป็น dict ใหญ่ (ไม่น่าจะใช่เคสนี้)
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # 1) ลอง JSONL: โหลดทีละบรรทัด
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parsed_lines, ok = [], 0
    for ln in lines:
        try:
            parsed_lines.append(json.loads(ln))
            ok += 1
        except Exception:
            parsed_lines.append(None)
    # ถ้าอย่างน้อยครึ่งหนึ่งพาร์สได้ ถือว่าเป็น JSONL
    if lines and ok / len(lines) >= 0.5:
        return [x for x in parsed_lines if x is not None]

    # 3) สแกนทั้งไฟล์แบบนับวงเล็บปีกกา (stream of objects)
    objs = []
    depth = 0
    in_str = False
    esc = False
    start = None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start:i+1]
                    try:
                        objs.append(json.loads(chunk))
                    except Exception:
                        # ถ้าพาร์สไม่ได้ แสดงว่าไฟล์ JSON ไม่สมบูรณ์ (มักเกิดจาก " ไม่ได้ escape)
                        pass
                    start = None
    if objs:
        return objs

    # ถ้าไม่มีวิธีไหนผ่าน
    raise ValueError("Cannot parse the file. Please ensure valid JSON/JSONL. "
                     "Common issue: inner quotes must be escaped (e.g., \\\"เที่ยวไทยคนละครึ่ง\\\").")

def _maybe_parse_json(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val

def _parse_million_thb(x):
    """รับ '2.345 ล้านบาท' หรือ ตัวเลข -> float หน่วย 'ล้านบาท'"""
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace("ล้านบาท","").replace("บาท","").replace(",","").strip()
    s = re.sub(r"[^0-9\.\-]", "", s)
    try: return float(s)
    except Exception: return None

def _parse_pct(x):
    """รับ '-10.11%' หรือ ตัวเลข -> float เป็นเปอร์เซ็นต์ (ไม่ใช่เศษส่วน)"""
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace("%","").replace(",","").strip()
    s = re.sub(r"[^0-9\.\-]", "", s)
    try: return float(s)
    except Exception: return None

def _fund_display(gi: dict) -> str:
    return gi.get("external_fund") or gi.get("fundcode") or gi.get("PortEngName") or "-"

def build_index_multi_fund(records):
    """
    รวมหลายกองต่อคน -> payload ต่อคนมี holdings[]
    พร้อม KPI รวม (fund_count, AUM รวม, %Unrealized ถ่วงด้วย AUM)
    """
    idx = {}

    for rec in records:
        if not (isinstance(rec, dict) and len(rec) == 1):
            # บางกรณีไฟล์อาจเป็น dict ใหญ่ { "paopow": {...}, "jj": {...} }
            # ให้แตกเป็นรายการย่อย
            if isinstance(rec, dict) and len(rec) > 1:
                for name, payload in rec.items():
                    _ingest(idx, name, payload)
                continue
            continue
        name, payload = next(iter(rec.items()))
        _ingest(idx, name, payload)

    # KPI รวมต่อคน
    for name, data in idx.items():
        h = data["holdings"]
        total_aum = sum([(x.get("aum") or 0.0) for x in h])
        total_cost = sum([(x.get("cost") or 0.0) for x in h])
        if total_aum > 0:
            wa_unl = sum([(x.get("aum") or 0.0) * (x.get("unrealized") or 0.0) for x in h]) / total_aum
        else:
            wa_unl = None

        data["general_information"].update({
            "fund_count": len(h),
            "aum_total": round(total_aum, 6) if total_aum else 0.0,          # หน่วย: ล้านบาท
            "cost_total": round(total_cost, 6) if total_cost else 0.0,        # หน่วย: ล้านบาท
            "unrealized_weighted_pct": round(wa_unl, 6) if wa_unl is not None else None,
        })
    return idx

def _ingest(idx, name, payload):
    if not isinstance(payload, dict):
        return
    for k in ["persona_agent","mutual_fund_agent","economic_and_news_agent","customer_card"]:
        payload[k] = _maybe_parse_json(payload.get(k, {})) or {}

    gi = payload.get("general_information", {}) or {}
    aum = _parse_million_thb(gi.get("aum"))
    cost = _parse_million_thb(gi.get("cost"))
    unl  = _parse_pct(gi.get("unrealized"))

    holding = {
        "fund_display": _fund_display(gi),
        "fundcode": gi.get("fundcode"),
        "fundtype": gi.get("fundtype"),
        "external_fund": gi.get("external_fund"),
        "PortEngName": gi.get("PortEngName"),
        "aum": aum,
        "cost": cost,
        "unrealized": unl,   # %
        "persona_agent": payload.get("persona_agent", {}),
        "mutual_fund_agent": payload.get("mutual_fund_agent", {}),
        "economic_and_news_agent": payload.get("economic_and_news_agent", {}),
        "customer_card": payload.get("customer_card", {}),
        "raw_gi": gi,
    }

    if name not in idx:
        idx[name] = {
            "general_information": {
                "customer_id": gi.get("customer_id"),
                "persona_flag": gi.get("persona_flag"),
            },
            "holdings": [],
        }
    else:
        if not idx[name]["general_information"].get("persona_flag") and gi.get("persona_flag"):
            idx[name]["general_information"]["persona_flag"] = gi.get("persona_flag")

    idx[name]["holdings"].append(holding)

def overview_df(indexed):
    rows=[]
    for name, p in indexed.items():
        gi = (p or {}).get("general_information", {})
        rows.append({
            "customer": name,
            "persona_flag": gi.get("persona_flag"),
            "funds": gi.get("fund_count"),
            "aum_total(ล้านบาท)": gi.get("aum_total"),
            "unrealized(WA, %)": gi.get("unrealized_weighted_pct"),
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
path = st.sidebar.text_input("Data file (.jsonl / .ndjson / .json)", "./data/json_database.jsonl")

p = Path(path)
if not p.exists():
    st.error(f"File not found at: {p.resolve()}")
    st.stop()

# โหลด + รวมหลายกองต่อคน
try:
    records = load_records_any(path)
except Exception as e:
    st.exception(e)
    st.stop()

indexed = build_index_multi_fund(records)
names = sorted(indexed.keys())
if not names:
    st.info("No customers found."); st.stop()

selected = st.sidebar.selectbox("Select customer", names, index=0)

# ---------------- overview table ----------------
st.subheader("Overview")
st.dataframe(overview_df(indexed), use_container_width=True, hide_index=True)

# ---------------- details ----------------
cust_payload = indexed[selected]
gi = (cust_payload or {}).get("general_information", {})
holdings = (cust_payload or {}).get("holdings", []) or []

st.markdown(f"### Details — **{selected}**")

# KPI รวม
metrics = [
    ("Persona", gi.get("persona_flag")),
    ("Funds", gi.get("fund_count")),
    ("AUM (Total, ลบ.)", gi.get("aum_total")),
    ("Unrealized (WA, %)", gi.get("unrealized_weighted_pct")),
]
metrics = [(t,v) for t,v in metrics if v not in (None, "", "-", "None")]
if metrics:
    cols = st.columns(len(metrics))
    for col, (t, v) in zip(cols, metrics):
        col.markdown(f"<div class='kpi'><div class='h'>{t}</div>{v}</div>", unsafe_allow_html=True)

st.write("")  # spacing

# เลือกกองทุนของลูกค้าคนนี้
fund_names = [h.get("fund_display") for h in holdings] or ["—"]
sel_fund = st.selectbox("Fund (this customer)", fund_names, index=0)
hsel = next((h for h in holdings if h.get("fund_display") == sel_fund), (holdings[0] if holdings else {}))

# content rows
c1, c2 = st.columns([1.25, 1.75])

# LEFT: Portfolio Status + News ของ "กองที่เลือก"
sel_card = (hsel or {}).get("customer_card", {}) or {}
sel_econ = (hsel or {}).get("economic_and_news_agent", {}) or {}
left_has_any = bool(sel_card.get("portfolio_current_status")) or bool(sel_econ.get("news_that_relate_with_industry")) or bool(sel_econ.get("materials"))

if left_has_any:
    with c1:
        if sel_card.get("portfolio_current_status"):
            st.markdown("<div class='h2'>📌 Portfolio Status</div>", unsafe_allow_html=True)
            st.markdown(sel_card.get("portfolio_current_status"))
        if sel_econ.get("news_that_relate_with_industry"):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='h2'>📰 News Highlights</div>", unsafe_allow_html=True)
            st.markdown(sel_econ.get("news_that_relate_with_industry"))
        mats = sel_econ.get("materials") or []
        if mats:
            st.markdown("<div class='muted' style='margin-top:.4rem;'>Sources:</div>", unsafe_allow_html=True)
            for m in mats:
                st.markdown(f"- {m}")

# RIGHT: Holdings table + Talking points / CTA / Fund Snapshot
with c2:
    if holdings:
        st.markdown("<div class='h2'>📚 Holdings</div>", unsafe_allow_html=True)
        df_h = pd.DataFrame(holdings).copy()
        tot_aum = gi.get("aum_total") or 0.0
        if "aum" in df_h.columns:
            df_h["weight_%"] = df_h["aum"].fillna(0) / (tot_aum if tot_aum else 1) * 100
        show_cols = ["fund_display","fundtype","aum","cost","unrealized","weight_%"]
        show_cols = [c for c in show_cols if c in df_h.columns]
        st.dataframe(df_h[show_cols], use_container_width=True, hide_index=True)

    mutual_agent  = (hsel or {}).get("mutual_fund_agent", {}) or {}
    customer_card = (hsel or {}).get("customer_card", {}) or {}
    tp  = customer_card.get("talking_point") or mutual_agent.get("talking_point") or []
    rtp = customer_card.get("reason_of_each_talking_point") or mutual_agent.get("reason_of_each_talking_point") or []
    cta  = customer_card.get("call_to_action") or []
    rcta = customer_card.get("reason_of_each_call_to_action") or []

    if tp:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>💬 Talking Points</div>", unsafe_allow_html=True)
        st.markdown(numbered_with_reasons_md(tp, rtp))

    if cta:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>✅ Call to Action</div>", unsafe_allow_html=True)
        st.markdown(numbered_with_reasons_md(cta, rcta))

    perf_lines = mutual_agent.get("last_3_year_performance_of_the_fund") or []
    if perf_lines:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<div class='h2'>📈 Fund Snapshot — {sel_fund}</div>", unsafe_allow_html=True)
        years, values, labels = parse_performance_strings(perf_lines)

        if years:
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
                x_min, x_max = min(years), max(years)
                x_mid  = (x_min + x_max) / 2
                x_half = max((x_max - x_min) / 2, 0.5) * 1.2
                x_range = [x_mid - x_half, x_mid + x_half]
                y_min, y_max = min(values), max(values)
                y_mid  = (y_min + y_max) / 2
                y_half = max((y_max - y_min) / 2, 1.0) * 1.2
                y_range = [y_mid - y_half, y_mid + y_half]
                fig.update_layout(
                    height=300,
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis=dict(title="Year", tickmode="array", tickvals=years, ticktext=labels, range=x_range),
                    yaxis=dict(title="Return (%)", range=y_range),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            with tcol:
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
