import os
from evoked.base import RecordingResult
import polars_config_meta
from evoked.io_test_copy import load_bulk
from evoked.preprocess import preprocess
from evoked.matched_filter import match_feature
from evoked.lda import match_feature_lda
from evoked.glrt import match_feature_glrt
from evoked.plotting import plot_trace, plot_io_curve, plot_detected, plot_all_files
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    base_path = "C:/Users/bbyer/OneDrive/Documents/UniversityofKentucky/BachstetterLab/evoked/evoked_broken/src/data/newdata_ca1/"
    # files = [
    #     '2025_03_04_0002.abf',
    #     '2025_03_03_0000.abf',
    #     '2025_03_05_0004.abf',
    # ]

    # base_path = os.path.join(os.path.dirname(__file__), "data")

    # met_path = os.path.join(os.path.dirname(__file__), "metadata.csv")

    # # get all valid ABF files
    # all_files = [os.path.join(base_path, f) for f in files if f.endswith('.abf')]


    # get all valid ABF files
    all_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.abf')]
    # new_list = [os.path.basename(file) for file in all_files]
    # for f in new_list:
    #     print("- ", f)

    met_path = os.path.join(os.path.dirname(__file__), "recording_metadata.yml")

    test = load_bulk(all_files, met_path)

    pre = preprocess(test)

    ca1_results = RecordingResult()

    ca1_results.add("Fiber Volley", match_feature(
        pre,
        window=(0.00125, 0.003),
        noise_window=(0.02324, 0.02499),
        p_value_threshold=0.05,
        snr_threshold=10.0,
        slope_transform=False
    ))

    ca1_results.add("fEPSP", match_feature(
        pre,
        window=(0.003, 0.0047),
        noise_window=(0.02324, 0.02499),
        p_value_threshold=0.05,
        snr_threshold=25.0,
        slope_transform=True
    ))

    # ca1_results.add("Population Spike", match_feature(
    #     pre,
    #     window=(0.0045, 0.006),
    #     threshold=0.8,
    #     snr_threshold=100.0,
    #     noise_window=(0.02324, 0.02499),
    #     slope_transform=False
    # ))

    # ca1_results.add("Fiber Volley LDA", match_feature_lda(
    #     pre,
    #     window=(0.00125, 0.003),
    #     noise_window=(0.02324, 0.02499),
    #     threshold=0.8,
    #     snr_threshold=200.0,
    #     slope_transform=False
    # ))

    # ca1_results.add("Fiber Volley GLRT", match_feature_glrt(
    #     pre,
    #     window=(0.00125, 0.003),
    #     noise_window=(0.02324, 0.02499),
    #     threshold=0.05,
    #     snr_threshold=10.0,
    #     slope_transform=False
    # ))

    print(ca1_results)
    # print(ca1_results.results.get("Fiber Volley").result.config_meta.get_metadata())

    ca1_trace_fig, ca1_trace_ax = plot_trace(pre, recording_result=ca1_results, id_value='2026_06_01_0000', channel=0, features=["Fiber Volley", "fEPSP"], stimuli=["25", "50", "75", "150", "200", "250", "300", "400", "500", "600"], annotated=True)
    # ca1_io_fig, _ = plot_io_curve(ca1_results, features=["Fiber Volley", "fEPSP", "Fiber Volley GLRT"], stimuli=["25", "50", "75", "150", "200", "250", "300", "400", "500", "600"])
    # ca1_detected_fig, _ = plot_detected(ca1_results, features=["Fiber Volley", "fEPSP", "Fiber Volley LDA"])
    ca1_trace_fig.savefig("ca1_testfig8.png", dpi=600, bbox_inches="tight")
    # plot_all_files(pre, stimuli=["25", "50", "75", "150", "200", "250", "300", "400", "500", "600"])

    # template = [ 0.0459179 ,  0.03676628,  0.01975369, -0.0015044 , -0.02067768,
    #    -0.03544458, -0.04895909, -0.0631387 , -0.07910559]
    
    # plt.plot(np.arange(len(template)), template)
    # plt.show()


if __name__ == "__main__":
    main()