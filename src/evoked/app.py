import streamlit as st
import polars as pl
import os
import re
from evoked.io import load_bulk, save_results_xlsx
from evoked.preprocess import preprocess
from evoked.ols import match_feature_ols
from evoked.plotting import plot_trace, plot_io_curve, plot_fit, plot_detected, plot_all_files
from evoked.base import RecordingResult, PreprocessParams
import matplotlib.pyplot as plt
from streamlit_adjustable_columns import adjustable_columns

# # Works exactly like st.columns - but with resize handles!
# col1, col2, col3 = adjustable_columns(3, labels=["📊 Charts", "📋 Data", "⚙️ Settings"])

st.set_page_config(layout="wide")
st.title("evoked")

collect_numbers = lambda x: [int(i) for i in re.split(r"[^0-9]+", x) if i != ""]
collect_window = lambda x: tuple(
    float(i) for i in re.findall(r"\d+(?:\.\d+)?", x)
)

col1, col2 = st.columns([1,3])

with col1:
    st.subheader("Data Preprocessing")
    filetypes = [".abf",".csv", ".tsv"]
    uploaded_files = st.file_uploader("Upload data", accept_multiple_files=True, type=filetypes)
    repnum = st.number_input("No. of Repetitions:", min_value=1, max_value=100, step=1)
    stimuli = collect_numbers(st.text_input("Stimulus stimuli (comma-separated):"))

    baseline_window = collect_window(st.text_input("Baseline window (comma-separated):"))
    artifact_window = collect_window(st.text_input("Artifact window (comma-separated):"))

    go = st.button("Load + Preprocess")
    if go:
        if not uploaded_files:
            st.error("Please upload at least one data file.")
            st.stop()

        if not stimuli:
            st.error("Please enter stimulus stimuli.")
            st.stop()

        params = PreprocessParams(
            baseline_window=baseline_window,
            artifact_window=artifact_window,
            artifact="template",
            smoothing="savgol",
            smoothing_params={
                "size": 7,
                "window_length": 15,
                "polyorder": 2,
                "cutoff": 2000.0,
                "order": 2,
            },
        )

        all_raw = load_bulk(
            uploaded_files,
            stimuli=stimuli,
            repnum=repnum,
        )

        all_prep = preprocess(all_raw, params)

        st.session_state["all_prep"] = all_prep
        st.session_state["stimuli"] = stimuli
        st.session_state["uploaded_file_names"] = [file.name for file in uploaded_files]
        st.session_state["figs"] = plot_all_files(
            all_prep,
            stimuli=stimuli,
            max_per_page=6,
        )
        st.session_state["params"] = params

with col2:
    st.subheader("Analysis")

    if "all_prep" not in st.session_state:
        st.info("Load and preprocess data first.")
        st.stop()

    all_prep = st.session_state["all_prep"]
    params = st.session_state["params"]
    stimuli = st.session_state["stimuli"]
    figs = st.session_state["figs"]

    if "results" not in st.session_state:
        st.session_state["results"] = RecordingResult(preprocess_params=params)

    results = st.session_state["results"]

    tabs = st.tabs([f"Page {i + 1}" for i in range(len(figs))])
    for i, tab in enumerate(tabs):
        with tab:
            st.pyplot(figs[i])

    available_ids = list(all_prep["id"].unique())

    template_ids = st.multiselect(
        "Select template files:",
        options=available_ids,
    )

    tab1, tab2 = st.tabs(["Feature Calculations", "Plotting"])

    col3, col4 = st.columns([1, 1])
    with tab1:
        with col3:
            feature_name = st.text_input("Feature name:")

            template_window = collect_window(
                st.text_input("Template window (comma-separated):")
            )

            search_window = collect_window(
                st.text_input("Search window (comma-separated):")
            )

        with col4:
            template_stimuli = collect_numbers(st.text_input("Template stimuli (comma-separated):"))
            
            r2_threshold = st.number_input("$R^2$ threshold:", min_value=0.0, max_value=1.0)

            slope_transform = st.selectbox("Slope transform:", options=[False, True], accept_new_options=False)

        add_feature = st.button("Add Feature")

        if add_feature:
            if not template_ids:
                st.error("Please select at least one template file.")
                st.stop()

            if not feature_name:
                st.error("Please enter a feature name.")
                st.stop()

            train_prep = all_prep[all_prep["id"].isin(template_ids)]
            test_prep = all_prep[~all_prep["id"].isin(template_ids)]

            results.add(
                feature_name,
                match_feature_ols(
                    train_df=train_prep,
                    test_df=test_prep,
                    template_window=template_window,
                    search_window=search_window,
                    template_stimuli=template_stimuli,
                    r2_threshold=r2_threshold,
                    slope_transform=slope_transform,
                ),
            )

            st.session_state["results"] = results
            st.success(f"Added feature: {feature_name}")
    with tab2:
        st.text("Hello")    
