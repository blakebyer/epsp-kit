import os
from datetime import datetime
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from evoked.io import load_bulk
from evoked.preprocess import preprocess
from evoked.base import PreprocessParams
import hashlib

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        /* Remove top and bottom margins */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        
        /* Remove default spacing between containers */
        .stVerticalBlock {
            gap: 0px !important;
        }
        .stAppDeployButton { visibility: hidden; display: none; }
        header { visibility: hidden; }
        #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)
#st.title("Blind Trace Review")

if "alias" not in st.session_state:
    st.session_state.alias = ""

with st.container(width="content", horizontal=True):
    if st.session_state.alias == "":
        alias = st.text_input("Reviewer Initials", placeholder="Reviewer Initials", label_visibility="collapsed", width=200)

        if st.button("Go"):
            if alias.strip():
                st.session_state.alias = alias.strip()
                st.rerun()
            else:
                st.warning("Enter reviewer initials.")

        st.stop()

# Settings
os.makedirs("review", exist_ok=True)
safe_alias = st.session_state.alias.replace(" ", "_")
SAVE_PATH = os.path.join("review", f"blind_classifier_{safe_alias}.csv")

REPNUM = 3
INT_LIST = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]

PARAMS = PreprocessParams(
    baseline_window=(0, 0.1),
    artifact_window=(0, 2.25),
    artifact="template",
    smoothing="savgol",
    smoothing_params={
        "size": 7,
        "window_length": 11,
        "polyorder": 3,
        "cutoff": 2000.0,
        "order": 2,
    },
)

files_list = [ 
    # 16 CA1
    '2025_03_04_0002.abf',
    '2025_03_03_0000.abf',
    '2025_03_05_0004.abf',
    '2025_03_06_0002.abf',
    '2025_03_06_0010.abf', 
    '2025_03_02_0000.abf', 
    '2025_03_07_0002.abf', 
    '2025_03_06_0004.abf', 
    '2025_03_05_0000.abf', 
    '2025_03_04_0012.abf',       
    '2025_03_03_0007.abf',
    '2025_03_07_0008.abf',
    '2025_03_04_0000.abf',
    '2025_05_22_0000.abf', 
    '2025_03_05_0002.abf',
    '2025_05_22_0006.abf',

    # # 16 DG
    '2025_05_22_0005.abf',
    '2025_03_06_0001.abf',
    '2025_03_05_0007.abf',
    '2025_03_04_0003.abf',
    '2025_03_04_0005.abf',
    '2025_05_22_0001.abf',
    '2025_03_07_0009.abf',
    '2025_03_07_0005.abf',
    '2025_03_07_0007.abf',
    '2025_03_02_0010.abf',
    '2025_03_04_0009.abf',
    '2025_03_06_0007.abf',
    '2025_03_03_0001.abf',
    '2025_03_05_0009.abf',
    '2025_03_03_0006.abf',
    '2025_03_02_0003.abf'
]

base_path = os.path.join(os.path.dirname(__file__), "data")

# get all valid ABF files
all_files = [os.path.join(base_path, f) for f in files_list if f.endswith('.abf')]

# Load / preprocess
if "all_proc" not in st.session_state:
    all_raw = load_bulk(
        all_files,
        intensities=INT_LIST,
        repnum=REPNUM,
    )

    st.session_state.all_proc = preprocess(all_raw, PARAMS)

all_proc = st.session_state.all_proc
seed = int(hashlib.sha256(safe_alias.encode()).hexdigest(), 16) % (2**32) # deterministic seed
# random blinded trace order
if "trace_order" not in st.session_state:
    trace_order = (
        all_proc
        .select(["id", "intensity"])
        .unique()
        .sample(fraction=1.0, shuffle=True, seed=seed)
        .with_row_index("trace_idx")
    )

    st.session_state.trace_order = trace_order

traces = st.session_state.trace_order

# load saved responses
if "responses" not in st.session_state:
    if os.path.exists(SAVE_PATH):
        df = pl.read_csv(SAVE_PATH)

        df = df.filter(
            pl.col("fiber_volley").is_not_null()
            & pl.col("fepsp").is_not_null()
            & pl.col("population_spike").is_not_null()
            & (pl.col("fiber_volley") != "")
            & (pl.col("fepsp") != "")
            & (pl.col("population_spike") != "")
        )

        st.session_state.responses = {
            int(row["trace_idx"]): {
                "fiber_volley": row["fiber_volley"],
                "fepsp": row["fepsp"],
                "population_spike": row["population_spike"],
            }
            for row in df.iter_rows(named=True)
        }

        saved_idxs = set(st.session_state.responses.keys())

        st.session_state.idx = next(
            (i for i in range(len(traces)) if i not in saved_idxs),
            len(traces) - 1,
        )

    else:
        st.session_state.responses = {}
        st.session_state.idx = 0

# current trace
idx = st.session_state.idx

trace_info = traces.row(idx, named=True)
trace_id = trace_info["id"]
trace_intensity = trace_info["intensity"]

trace_df = (
    all_proc
    .filter(
        (pl.col("id") == trace_id)
        & (pl.col("intensity") == trace_intensity)
    )
    .sort("time")
)

alias = st.session_state.alias

st.caption(f"Trace {idx + 1} / {len(traces)}")

# plotly trace viewer
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=trace_df["time"].to_numpy() * 1000,
        y=trace_df["voltage"].to_numpy(),
        mode="lines",
    )
)

fig.update_layout(
    height=450,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_title="Time (ms)",
    yaxis_title="Voltage (mV)",
    #dragmode="drawline",
    modebar_add=["drawline","eraseshape"]
)

# fig.show(config={
#     "modeBarButtonsToAdd": [
#         "drawline",
#         "eraseshape"
#     ]
# })

st.plotly_chart(fig, width="stretch")

st.caption(
    "Mark **Yes** only if you would report a reliable quantitative voltage for this feature. "
    "Mark **No** if the feature is absent, noisy, ambiguous, or would require guessing."
)

# load previous answer
current = st.session_state.responses.get(idx, {})

# questions
col1, col2, col3 = st.columns(3)

options = ["Unanswered", "Yes", "No"]

def answer_index(voltage):
    return options.index(voltage) if voltage in options else 0

current = st.session_state.responses.get(idx, {})

col1, col2, col3 = st.columns(3)

with col1:
    fv = st.radio(
        "Fiber volley quantifiable?",
        options,
        index=answer_index(current.get("fiber_volley", "Unanswered")),
        horizontal=True,
        key=f"fv_{idx}",
    )

with col2:
    fepsp = st.radio(
        "fEPSP slope quantifiable?",
        options,
        index=answer_index(current.get("fepsp", "Unanswered")),
        horizontal=True,
        key=f"fepsp_{idx}",
    )

with col3:
    ps = st.radio(
        "Population spike quantifiable?",
        options,
        index=answer_index(current.get("population_spike", "Unanswered")),
        horizontal=True,
        key=f"ps_{idx}",
    )


# save function
def save_response():
    if fv == "Unanswered" or fepsp == "Unanswered" or ps == "Unanswered":
        return False

    st.session_state.responses[idx] = {
        "fiber_volley": fv,
        "fepsp": fepsp,
        "population_spike": ps,
    }

    row = {
        "trace_idx": idx,
        "id": trace_id,
        "intensity": trace_intensity,
        "fiber_volley": fv,
        "fepsp": fepsp,
        "population_spike": ps,
        "reviewer_alias": alias,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if os.path.exists(SAVE_PATH):
        df = pl.read_csv(SAVE_PATH)
        df = df.filter(pl.col("trace_idx") != idx)
        df = pl.concat([df, pl.DataFrame([row])])
    else:
        df = pl.DataFrame([row])

    df.sort("trace_idx").write_csv(SAVE_PATH)

    return True

# navigation
nav1, _, nav3 = st.columns(3)

with nav1:
    if st.button("Previous", disabled=idx == 0):
        save_response()
        st.session_state.idx -= 1
        st.rerun()

with nav3:
    label = "Save + Finish" if idx == len(traces) - 1 else "Save + Next"

    if st.button(label):
        if save_response():
            if idx < len(traces) - 1:
                st.session_state.idx += 1
                st.rerun()
            else:
                st.success("Review complete. Final trace saved.")
        else:
            st.warning("Answer all questions before saving.")

# optional debug
#st.write(st.session_state.responses)