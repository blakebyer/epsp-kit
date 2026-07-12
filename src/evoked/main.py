import os
from evoked.io import load_bulk, save_results_xlsx
from evoked.preprocess import preprocess
from evoked.ols import match_feature_ols
from evoked.plotting import plot_trace, plot_io_curve, plot_fit, plot_detected, plot_all_files
from evoked.base import RecordingResult, PreprocessParams
import matplotlib.pyplot as plt

def main() -> None:
    ## Determine which files are for training and testing

    base_path = "/mnt/c/Users/bbyer/OneDrive/Documents/UniversityofKentucky/BachstetterLab/evoked/evoked_broken/src/data/newdata_ca1/"

    # get all valid ABF files
    all_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.abf')]

    stimuli_list = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600] 

    params = PreprocessParams(
        baseline_window=(0, 0.1),
        artifact_window=(0, 2.25),
        artifact="template",
        smoothing="savgol",
        smoothing_params={
                            "size": 7,
                            "window_length": 15,
                            "polyorder": 2,
                            "cutoff": 2000.0,
                            "order": 2,
                        }
    )

    # load all files
    all_raw = load_bulk(all_files, stimuli=stimuli_list, repnum=3)

    # clean them up
    all_prep = preprocess(all_raw, params)

    # plot all traces to determine train/test split
    plot_all_files(all_prep, stimuli=[50, 100, 200, 300, 400, 500, 600], max_per_page=6)

    ## Make a train/test split and run the pipeline

    train_files = [
        "2026_05_28_0008.abf", "2026_06_01_0011.abf", "2026_06_01_0005.abf", 
        "2026_05_28_0017.abf", "2026_06_01_0009.abf", "2026_06_01_0011.abf", 
        "2026_06_01_0010.abf", "2026_05_28_0020.abf"
    ]

    test_files = [f for f in all_files if os.path.basename(f) not in train_files]
    train_files = [os.path.join(base_path, f) for f in train_files]

    template_ints = [300, 400, 500, 600]

    train_raw = load_bulk(
            train_files, 
            stimuli=stimuli_list, 
            repnum=3
        )

    test_raw = load_bulk(
        test_files,
        stimuli=stimuli_list,
        repnum=3
    )

    train_prep = preprocess(train_raw, params)
    test_prep = preprocess(test_raw, params)

    ca1_results = RecordingResult(preprocess_params=params)

    ca1_results.add("Fiber Volley", match_feature_ols(
        train_df=train_prep, test_df=test_prep, 
        template_window=(1.5, 3.0), search_window=(1.0, 3.2), 
        template_stimuli=template_ints, 
        r2_threshold=0.8,
        slope_transform=False
    ))

    ca1_results.add("fEPSP", match_feature_ols(
        train_df=train_prep, test_df=test_prep, 
        template_window=(3.0, 4.7), search_window=(2.5, 5.0), 
        template_stimuli=template_ints, 
        r2_threshold=0.4,
        slope_transform=True
    ))

    ca1_results.add("Population Spike", match_feature_ols(
        train_df=train_prep, test_df=test_prep, 
        template_window=(4.5, 6.0), search_window=(4.0, 6.5), 
        template_stimuli=template_ints, 
        r2_threshold=0.8,
        slope_transform=False
    ))

    ## Make cool plots

    ca1_trace_fig, ca1_trace_ax = plot_trace(test_prep, recording_result=ca1_results, id_value='2026_05_28_0009', features=["Fiber Volley", "fEPSP", "Population Spike"], stimuli=stimuli_list, annotated=True)
    ca1_io_fig, ca1_io_axes = plot_io_curve(ca1_results, features=["Fiber Volley", "fEPSP", "Population Spike"], stimuli=stimuli_list, rc_params={'font.size': 8})
    ca1_fit_fig, ca1_fit_axes = plot_fit(test_prep, ca1_results, features=["Fiber Volley", "fEPSP", "Population Spike"], stimulus=50, id_value='2026_05_28_0009')
    ca1_fit_fig1, ca1_fit_axes1 = plot_fit(test_prep, ca1_results, features=["Fiber Volley", "fEPSP", "Population Spike"], stimulus=600, id_value='2026_05_28_0009')
    ca1_detected_fig, ca1_detected_axes = plot_detected(ca1_results, features=["Fiber Volley", "fEPSP", "Population Spike"])
    ca1_detected_fig.savefig("detected.png", dpi=600, bbox_inches="tight")

    ## Save results

    save_results_xlsx(ca1_results, "ca1_2026_06_19.xlsx")


if __name__ == "__main__":
    main()
