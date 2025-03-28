import os
import pickle
import shutil
import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# from pygifsicle import optimize
from openalea.mtg.traversal import post_order2
from openalea.mtg.plantframe import color
from openalea.mtg import turtle as turt
from math import floor, ceil, trunc, log10
import pandas as pd

# Set options to display all rows and columns
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Show full content of each column
pd.set_option("display.width", 0)  # Adjusts to screen width

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.ticker import FuncFormatter
import xarray as xr
from pint import UnitRegistry
import re
import pyvista as pv
from random import random
import warnings

from log.visualize import plot_mtg, plot_xr, custom_colorbar
import openalea.plantgl.all as pgl

from analyze.workflow.cnwheat_comparisions import compare_shoot_outputs



balance_dicts_C = {"hexose": dict(hexose_exudation={"type": "output", "conversion": 1.},
                        hexose_uptake_from_soil={"type": "input", "conversion": 1.},
                        mucilage_secretion={"type": "output", "conversion": 1.},
                        cells_release={"type": "output", "conversion": 1.},
                        maintenance_respiration={"type": "output", "conversion": 1 / 6},
                        hexose_consumption_by_growth={"type": "output", "conversion": 1.},
                        hexose_diffusion_from_phloem={"type": "input", "conversion": 1.},
                        hexose_active_production_from_phloem={"type": "input", "conversion": 1.},
                        sucrose_loading_in_phloem={"type": "output", "conversion": 2},
                        hexose_mobilization_from_reserve={"type": "input", "conversion": 1.},
                        hexose_immobilization_as_reserve={"type": "output", "conversion": 1.},
                        deficit_hexose_root={"type": "output", "conversion": 1.},
                        AA_synthesis={"type": "output", "conversion": 1.4},
                        AA_catabolism={"type": "input", "conversion": 1 / 1.4},
                        N_metabolic_respiration={"type": "output", "conversion": 1 / 6}),

                "AA": dict(diffusion_AA_phloem={"type": "input", "conversion": 1.},
                        import_AA={"type": "input", "conversion": 1.},
                        diffusion_AA_soil={"type": "output", "conversion": 1.},
                        export_AA={"type": "output", "conversion": 1.},
                        AA_synthesis={"type": "input", "conversion": 1.},
                        storage_synthesis={"type": "output", "conversion": 65},
                        storage_catabolism={"type": "input", "conversion": 1 / 65},
                        AA_catabolism={"type": "output", "conversion": 1.},
                        amino_acids_consumption_by_growth={"type": "output", "conversion": 1.}
                        ), 

                "Nm": dict(import_Nm={"type": "input", "conversion": 1.},
                    diffusion_Nm_soil={"type": "output", "conversion": 1.},
                    diffusion_Nm_xylem={"type": "input", "conversion": 1.},
                    export_Nm={"type": "output", "conversion": 1.},
                    AA_synthesis={"type": "output", "conversion": 1.4},
                    AA_catabolism={"type": "input", "conversion": 1./1.4}),
                "rhizodeposits": dict(Gross_Hexose_Exudation={"type": "output", "conversion": 1.},
                    Gross_AA_Exudation={"type": "output", "conversion": 1.})
                 }

balance_dicts_no_C = {
                "AA": dict(diffusion_AA_phloem={"type": "input", "conversion": 1.},
                        import_AA={"type": "input", "conversion": 1.},
                        diffusion_AA_soil={"type": "output", "conversion": 1.},
                        export_AA={"type": "output", "conversion": 1.},
                        AA_synthesis={"type": "input", "conversion": 1.},
                        struct_synthesis={"type": "output", "conversion": 1},
                        storage_synthesis={"type": "output", "conversion": 65},
                        storage_catabolism={"type": "input", "conversion": 1 / 65},
                        AA_catabolism={"type": "output", "conversion": 1.}
                        ),

                "Nm": dict(import_Nm={"type": "input", "conversion": 1.},
                    diffusion_Nm_soil={"type": "output", "conversion": 1.},
                    diffusion_Nm_xylem={"type": "input", "conversion": 1.},
                    export_Nm={"type": "output", "conversion": 1.},
                    AA_synthesis={"type": "output", "conversion": 1.4},
                    AA_catabolism={"type": "input", "conversion": 1./1.4}),

                "labile_N": dict(import_Nm={"type": "input", "conversion": 1.},
                        diffusion_Nm_soil={"type": "output", "conversion": 1.},
                        diffusion_Nm_xylem={"type": "input", "conversion": 1.},
                        export_Nm={"type": "output", "conversion": 1.},
                        diffusion_AA_phloem={"type": "input", "conversion": 1.},
                        import_AA={"type": "input", "conversion": 1.},
                        diffusion_AA_soil={"type": "output", "conversion": 1.},
                        export_AA={"type": "output", "conversion": 1.},
                        struct_synthesis={"type": "output", "conversion": 1},
                        storage_synthesis={"type": "output", "conversion": 65},
                        storage_catabolism={"type": "input", "conversion": 1 / 65}
                        ),
                        
                "rhizodeposits": dict(Gross_AA_Exudation={"type": "output", "conversion": 1.})
                 }

color_palette = dict(
    black="#000000",
    lightorange="#E69F00",
    lightblue="#56B4E9",
    green="#009E73",
    yellow="#F0E442",
    blue="#0072B2",
    orange="#D55E00",
    pink="#CC79A7")

ureg = UnitRegistry()


def analyze_data(scenarios, outputs_dirpath, inputs_dirpath, on_sums=False, on_raw_logs=False, animate_raw_logs=False, on_shoot_logs=False, on_performance=False,
                 target_properties=None, subdir_custom_name=None, **kwargs):
    # TODO if not available, return not performed
    print("[INFO] Starting data analysis")
    if on_sums:
        for scenario in scenarios:
            print("     [INFO] Producing 2D plots from summed and averaged properties")
            plot_csv(csv_dirpath=os.path.join(outputs_dirpath, scenario, "MTG_properties/MTG_properties_summed"),
                    csv_name="plant_scale_properties.csv", properties=target_properties)
            # plot_csv(csv_dirpath=os.path.join(outputs_dirpath, scenario, "MTG_properties/MTG_properties_summed"),
            #         csv_name="plant_scale_properties.csv", properties=["import_Nm", "diffusion_AA_soil"], stacked=True)
            # plot_csv(csv_dirpath=os.path.join(outputs_dirpath, scenario, "MTG_properties/MTG_properties_summed"),
            #         csv_name="plant_scale_properties.csv", properties=["axial_export_water_up", "Nm_root_shoot_xylem", "AA_root_shoot_phloem_record"], stacked=True)
            print("     [INFO] Finished 2d plots")
    if on_raw_logs:
        print("     [INFO] Starting deep learning analysis on raw logs...")
        from analyze.workflow.STM_analysis.main_workflow import run_analysis
        from analyze.workflow.global_sensivity.run_global_sensitivity import regression_analysis
        xarray_deep_learning()
        print("     [INFO] Finished DL")
    if animate_raw_logs:
        print("     [INFO] Starting plot production from raw logs...")
        
        fps=5
        dataset = open_and_merge_datasets(scenarios=scenarios, root_outputs_path=outputs_dirpath)
        #dataset["NAE"] = Indicators.Nitrogen_Aquisition_Efficiency(d=dataset)
        #dataset["Cumulative_NAE"] = Indicators.Cumulative_Nitrogen_Aquisition_Efficiency(d=dataset)
        #dataset["Cumulative_Nitrogen_Uptake"] = Indicators.Cumulative_Nitrogen_Uptake(d=dataset)
        #dataset["Cumulative_Carbon_Costs"] = Indicators.Cumulative_Carbon_Costs(d=dataset)
        #dataset["Gross_Hexose_Exudation"] = Indicators.Gross_Hexose_Exudation(d=dataset)
        dataset["Net AA Exudation"] = Indicators.compute(d=dataset, formula="diffusion_AA_soil + diffusion_AA_soil_xylem - import_AA")
        #dataset["Gross_C_Rhizodeposition"] = Indicators.Gross_C_Rhizodeposition(d=dataset)
        #dataset["Rhizodeposits_CN_Ratio"] = Indicators.Rhizodeposits_CN_Ratio(d=dataset)
        #dataset["CN_Ratio_Cumulated_Rhizodeposition"] = Indicators.CN_Ratio_Cumulated_Rhizodeposition(d=dataset)
        #dataset["z2"] = - dataset["z2"]
        dataset["Root_Hairs_Surface"] = Indicators.Root_Hairs_Surface(d=dataset)
        dataset["Root_Hairs_Proportion"] = Indicators.Root_Hairs_Proportion(d=dataset)
        dataset["Labile_Nitrogen"] = Indicators.Labile_Nitrogen(d=dataset)
        dataset["cylinder_surface"] = Indicators.cylinder_surface(d=dataset)
        dataset["Net mineral N uptake"] = Indicators.compute(d=dataset, formula="import_Nm + mycorrhizal_mediated_import_Nm - diffusion_Nm_soil - diffusion_Nm_soil_xylem")
        dataset["Lengthy mineral N uptake"] = Indicators.compute(d=dataset, formula="(import_Nm + mycorrhizal_mediated_import_Nm - diffusion_Nm_soil - diffusion_Nm_soil_xylem) / length")
        dataset["Massic mineral N uptake"] = Indicators.compute(d=dataset, formula="(import_Nm + mycorrhizal_mediated_import_Nm - diffusion_Nm_soil - diffusion_Nm_soil_xylem) / struct_mass")

        # Z contributions
        zcontrib_flow = "import_Nm"
        fig_zcontrib, ax_zcontrib = plt.subplots(1, 1)
        fig_zcontrib.set_size_inches(10.5, 10.5)

        #print(dataset.root_order)

        # First individual analyses
        oldest_scenario = scenarios[-1]
        oldest_dataset = dataset
        # oldest_dataset = filter_dataset(dataset, scenario=oldest_scenario)
        step = 0.005
        distance_bins = np.arange(oldest_dataset["distance_from_tip"].min(), 
                                  oldest_dataset["distance_from_tip"].max() + step, step)
        scenario_times = dict(zip(scenarios, dataset.t.values))
        grouping_distances = []
        normalized_input_flux = []
        sucrose_input_df = pd.read_csv("inputs/sucrose_input_Swinnen_et_al_1994_20degrees_interpolated.csv", sep=';')

        for scenario in scenarios:
            print(f"[INFO] Processing scenario {scenario}")
            raw_dirpath = os.path.join(outputs_dirpath, scenario, "MTG_properties/MTG_properties_raw/")
            mtg_dirpath = os.path.join(outputs_dirpath, scenario, "MTG_files/")

            if len(scenarios) > 1:
                scenario_dataset = filter_dataset(dataset, scenario=scenario)
            else:
                scenario_dataset = dataset

            # recolorize_glb(100, scenario_dataset, property="Nm", glb_dirpath="", 
            #                colormap="jet")
            # try:
            #     # RootCyNAPSFigures.Fig_1_c(scenario_dataset, raw_dirpath)
            #     # RootCyNAPSFigures.Fig_1_d(scenario_dataset, raw_dirpath)
            #     RootCyNAPSFigures.Fig_2(scenario_dataset, raw_dirpath)
            # except:
            #     print(f"Figure creation failed for scenario {scenario}")
            RootCyNAPSFigures.Fig_1_c(scenario_dataset, raw_dirpath)

            print(np.unique(scenario_dataset["type"].values))
            unique, counts = np.unique(scenario_dataset["label"].values, return_counts=True)
            print(dict(zip(unique, counts))["Apex"])

            unique, counts = np.unique(scenario_dataset["axis_index"].values, return_counts=True)
            # print(dict(zip(unique, counts)))

            seminal_id = [axis_id for axis_id in unique if axis_id.startswith("seminal")]
            nodal_id = [axis_id for axis_id in unique if axis_id.startswith("adventitious")]
            laterals_id = [axis_id for axis_id in unique if axis_id.startswith("lateral")]
            print(seminal_id)
            
            final_dataset = scenario_dataset.sel(t=scenario_times[scenario])[["distance_from_tip", "root_order", "axis_index", "struct_mass", "length", "Net mineral N uptake", "Net AA Exudation"]]
            
            seminal_dataset = final_dataset.where(final_dataset["axis_index"].isin(seminal_id), drop=True)
            nodal_dataset = final_dataset.where(final_dataset["axis_index"].isin(nodal_id), drop=True)
            lateral_dataset = final_dataset.where(final_dataset["axis_index"].isin(laterals_id), drop=True)

            print(seminal_dataset, nodal_dataset, lateral_dataset)

            
            grouping_distance, total_struct_mass = RootCyNAPSFigures.Fig_2(final_dataset, raw_dirpath, distance_bins, flow="Net mineral N uptake", normalization_property="length")
            RootCyNAPSFigures.Fig_2_lines(final_dataset, raw_dirpath, distance_bins, flow="Net mineral N uptake", normalization_property="length")
            grouping_distances.append(grouping_distance)
            sucrose_input_rate = sucrose_input_df.loc[sucrose_input_df["t"] == scenario_times[scenario], "sucrose_input_rate"].item()
            normalized_input_flux.append(sucrose_input_rate/total_struct_mass)

            # print(scenario_dataset.where(scenario_dataset.distance_from_tip < 0.01, drop=True).where(scenario_dataset.z1 < -0.10, drop=True))
            # CN_balance_animation_pipeline(dataset=scenario_dataset, outputs_dirpath=os.path.join(outputs_dirpath, scenario), fps=fps, C_balance=True, target_vid=122)
            # CN_balance_animation_pipeline(dataset=scenario_dataset, outputs_dirpath=os.path.join(outputs_dirpath, scenario), fps=fps, C_balance=True, target_vid=100)
            #surface_repartition(dataset, output_dirpath=outputs_dirpath, fps=fps)

            # apex_zone_contribution(scenario_dataset, output_dirpath=raw_dirpath, apex_zone_length=0.05,
                                # flow="import_Nm", summed_input="diffusion_AA_phloem", color_prop="root_exchange_surface")
            # apex_zone_contribution(scenario_dataset, output_dirpath=raw_dirpath, apex_zone_length=0.02,
            #                        flow="import_Nm", summed_input="diffusion_AA_phloem", color_prop="C_hexose_root")
            # apex_zone_contribution(dataset, output_dirpath=outputs_dirpath, apex_zone_length=0.02,
            #                        flow="import_Nm", summed_input="diffusion_AA_phloem", color_prop="C_hexose_root")
            # trajectories_plot(dataset, output_dirpath=outputs_dirpath, x="distance_from_tip", y="NAE",
            #                  color=None, fps=fps)
            
            #z_zone_contribution(fig_zcontrib, ax_zcontrib, dataset=scenario_dataset, scenario=scenario, zmin=0.08, zmax=0.12, flow=zcontrib_flow)
            # z_zone_contribution(fig_zcontrib, ax_zcontrib, dataset=scenario_dataset, scenario=scenario, zmin=0.08, zmax=0.12, flow="Gross_Hexose_Exudation")
            #z_zone_contribution(fig_zcontrib, ax_zcontrib, dataset=scenario_dataset, scenario=scenario, zmin=0.08, zmax=0.12, flow="Gross_AA_Exudation")
            
            # Snapshots over specific days
            # ignore, snapshot_length, snapshot_number = 72, 1100, 1
            # props = ["root_exchange_surface", "NAE", "Rhizodeposits_CN_Ratio"]
            # props = ["root_exchange_surface"]
            # mean_and_std = [False, True, True]
            # x_max = [0.03, 2, 200]

            # distance = int((max(scenario_dataset.t.values) - ignore - snapshot_length) / snapshot_number)
            # for snp in range(1, snapshot_number+1):
            #     tstart, tstop = ignore + snp*distance, ignore + snp*distance + snapshot_length
            #     for i, prop in enumerate(props):
            #         pipeline_z_bins_animations(dataset=scenario_dataset, prop=prop, metabolite="hexose", output_path=raw_dirpath, fps=fps, t_start=tstart, t_stop=tstop, mean_and_std=mean_and_std[i], x_max=x_max[i])
            #         pipeline_z_bins_animations(dataset=scenario_dataset, prop=prop, metabolite="AA", output_path=raw_dirpath, fps=fps, t_start=tstart, t_stop=tstop, mean_and_std=mean_and_std[i], x_max=x_max[i])
            #         pipeline_z_bins_animations(dataset=scenario_dataset, prop=prop, metabolite="Nm", output_path=raw_dirpath, fps=fps, t_start=tstart, t_stop=tstop, mean_and_std=mean_and_std[i], x_max=x_max[i])

            # post_color_mtg(os.path.join(mtg_dirpath, "root_1570.pckl"), mtg_dirpath, property="import_Nm", flow_property=True, 
            #                recording_off_screen=False, background_color="brown", imposed_min=1e-10, imposed_max=1.5e-9, log_scale=True, spinning=False, root_hairs=True)
            #post_color_mtg(os.path.join(mtg_dirpath, "root_1527.pckl"), mtg_dirpath, property="import_Nm", flow_property=True, 
            #                recording_off_screen=False, background_color="white", imposed_min=1e-10, imposed_max=1.5e-9, log_scale=True, spinning=True)

        fig, ax = plt.subplots()
        # plotted = ax.scatter(grouping_distances, normalized_input_flux, c=np.array(list(scenario_times.values())) / 24)
        plotted = ax.scatter(np.array(list(scenario_times.values())) / 24, grouping_distances)
        
        fig.colorbar(plotted, ax=ax, label=f"Plant age (days)")
        xunit = "m"
        ax.set_xlabel(rf"$Distance from apex to group 50% of root system N uptake\ \mathrm{{({xunit})}}$")
        ax.set_xlabel(f"Distance from apex to group 50% of root system N uptake({xunit})")
        yunit = "mol.g-1.s-1"
        # ax.set_ylabel(rf"$Normalized sucrose input rate\ \mathrm{{({yunit})}}$")
        ax.set_ylabel(f"Normalized sucrose input rate ({yunit})")
        fig.savefig(os.path.join(outputs_dirpath, "C_input_dependancy.png"))


        # Then scenario comparisions
        if subdir_custom_name:
            comparisions_dirpath = os.path.join(outputs_dirpath, "comparisions", subdir_custom_name)
            if not os.path.isdir(comparisions_dirpath):
                os.mkdir(comparisions_dirpath)
        else:
            comparisions_dirpath = os.path.join(outputs_dirpath, "comparisions")
        # apex_zone_contribution_final(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="import_Nm", final_time=48, mean_and_std=True, x_proportion=True)
        # apex_zone_contribution_final(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="Gross_AA_Exudation", final_time=48, mean_and_std=True, x_proportion=True)
        # dataset = filter_dataset(dataset, prop="root_order", propis=1)
        # apex_zone_contribution_final(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="import_Nm", grouped_geometry="cylinder_surface", final_time=48, mean_and_std=True, x_proportion=True)
        # root_length_x_percent_contributors(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="import_Nm", grouped_geometry="length", final_time=48, mean_and_std=True, x_proportion=True)
        #top_percent_contributors(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="import_Nm", grouped_geometry="cylinder_surface", unit="m**2", final_time=48, mean_and_std=False)
        #top_percent_contributors(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="import_Nm", grouped_geometry="cylinder_surface", unit="m**2", final_time=48, mean_and_std=True)
        
        #apex_zone_contribution_final(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="Gross_AA_Exudation", grouped_geometry="struct_mass", final_time=48, mean_and_std=True, x_proportion=True)
        #apex_zone_contribution_final(dataset=dataset, scenarios=scenarios, outputs_dirpath=comparisions_dirpath, flow="radial_import_water", grouped_geometry="struct_mass", final_time=48, mean_and_std=True, x_proportion=True)
        
        # #!!!!! R1 !!!!!
        # pipeline_compare_z_bins_animations(dataset=dataset, scenarios=scenarios, output_path=comparisions_dirpath, prop="root_exchange_surface", metabolic_flow="import_Nm", 
        #                                    fps=fps, t_start=12, t_stop=max(scenario_dataset.t.values)-12, step=24, stride=24, mean_and_std=False, x_max_down=2.5, x_max_up=4e-8)

        # #!!!!! R2 !!!!!
        # pipeline_compare_z_bins_animations(dataset=dataset, scenarios=scenarios, output_path=comparisions_dirpath, prop="length", metabolic_flow="import_Nm", 
        #                                     fps=fps, t_start=12, t_stop=max(scenario_dataset.t.values)-12, step=24, stride=24, mean_and_std=False, x_max_down=20, special_case=True)

        # #!!!!! R3 !!!!!
        # pipeline_compare_z_bins_animations(dataset=dataset, scenarios=scenarios, output_path=comparisions_dirpath, prop="struct_mass", metabolic_flow="Gross_C_Rhizodeposition", 
        #                                     fps=fps, t_start=12, t_stop=max(scenario_dataset.t.values)-12, step=24, stride=24, mean_and_std=False, x_max_down=1, x_max_up=3e-8, log_scale=False)
        
        # #!!!!! R4 !!!!!
        # stride = 24
        # pipeline_compare_z_bins_animations(dataset=dataset, scenarios=scenarios, output_path=comparisions_dirpath, prop="struct_mass", metabolic_flow="CN_Ratio_Cumulated_Rhizodeposition", 
        #                                     fps=fps, t_start=int(stride/2), t_stop=max(scenario_dataset.t.values)-int(stride/2), step=24, stride=stride, mean_and_std=True, x_max_down=1, x_max_up=250)

        # screenshot_time = 1152
        # pipeline_compare_z_bins_animations(dataset=dataset, scenarios=scenarios, output_path=comparisions_dirpath, prop="struct_mass", metabolic_flow="CN_Ratio_Cumulated_Rhizodeposition", 
        #                                     fps=fps, t_start=screenshot_time, t_stop=screenshot_time, step=1, stride=1, mean_and_std=True, x_max_down=1, x_max_up=1000, screenshot=True, log_scale=True)
        

        # !!!!! R Test !!!!!
        # pipeline_compare_z_bins_animations(dataset=dataset, scenarios=scenarios, output_path=comparisions_dirpath, prop="struct_mass", metabolic_flow="Root_Hairs_Proportion", 
        #                                     fps=fps, t_start=12, t_stop=max(scenario_dataset.t.values)-12, step=24, stride=24, mean_and_std=True, x_max_down=1, x_max_up=0.01, log_scale=False)
        
        # pipeline_compare_z_bins_animations(dataset=dataset, scenarios=scenarios, output_path=comparisions_dirpath, prop="struct_mass", metabolic_flow="Root_Hairs_Surface", 
        #                                     fps=fps, t_start=12, t_stop=max(scenario_dataset.t.values)-12, step=24, stride=24, mean_and_std=True, x_max_down=1, x_max_up=1e-5, log_scale=False)
        

        # ax_zcontrib.legend()
        # ax_zcontrib.set_ylabel("proportion")
        # ax_zcontrib.set_title("Contributions of the patch zones relative to the whole root system")
        #
        # fig_zcontrib.savefig(os.path.join(comparisions_dirpath, f"z_zone_contribution_{zcontrib_flow}.png"))
        # plt.close()
        #
        #pipeline_compare_to_experimental_data(dataset=dataset, output_path=comparisions_dirpath)

        print("     [INFO] Finished plotting raw logs")

    if on_shoot_logs:
        for scenario in scenarios:
            print(" [INFO] Starting producing CN-Wheat plots...")
            cnwheat_plot_csv(csv_dirpath=os.path.join(outputs_dirpath, scenario, "MTG_properties/shoot_properties"))
            print(" [INFO] Finished  CN-Wheat plots")

            print(" [INFO] Starting comparision plots on CN-Wheat outputs...")
            compare_shoot_outputs(reference_dirpath=os.path.join(inputs_dirpath, "postprocessing"),
                                  newsimu_dirpath=os.path.join(outputs_dirpath, scenario, "MTG_properties/shoot_properties"),
                                  meteo_data_dirpath=os.path.join(inputs_dirpath, "meteo_Ljutovac2002.csv"))
            print(" [INFO] Finished comparision plots on CN-Wheat outputs...")

    if on_performance:
        for scenario in scenarios:
            print(" [INFO] Analysing running performances...")
            plot_csv(csv_dirpath=os.path.join(outputs_dirpath, scenario), csv_name="simulation_performance.csv", stacked=True)
            print(" [INFO] Finished plotting performances")



def test_output_range(outputs_dirpath, scenarios, test_file_dirpath):
    
    # List to store warnings
    logged_warnings = []

    # Custom warning handler to capture warnings without real-time output
    def capture_warning(message, category, filename, lineno, file=None, line=None):
        warning_msg = f"{filename}:{lineno}: {category.__name__}: {message}"
        logged_warnings.append(warning_msg)

    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    for scenario in scenarios:
        print("\n")
        print(f"{GREEN}LAUNCHING {scenario} OUTPUT RANGE CHECKS...{RESET}")
        print("\n")

        dataset = open_and_merge_datasets(scenarios=[scenario], root_outputs_path=outputs_dirpath)

        # remove the very first step where all fluxes are 0
        dataset = dataset.where(dataset.t > 0).dropna(dim='t', how='all')

        test_df = pd.read_excel(test_file_dirpath)
        test_df = test_df.replace({np.nan: None})

        failed_tests = 0
        passed_tests = 0
        failed_names = []
        passed_names = []
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = capture_warning

            for test_row in test_df.iterrows():
                row = test_row[1].to_dict()

                output_name = str(row["output"])
                if "+" in output_name or "-" in output_name or "*" in output_name or "/" in output_name:
                    # the variable doesn't exist and has to be computed
                    dataset[output_name] = eval(output_name, {}, dataset)
                if hasattr(dataset, output_name):
                    observed_min = float(row["min"])
                    observed_max = float(row["max"])
                    normalization_variable = row["normalize_by"]
                    checks = dict(check_single_values	= bool(row["check_single_values"]),
                                check_collar = bool(row["check_collar"]),
                                check_sum = bool(row["check_sum"]),
                                check_mean = bool(row["check_mean"]),
                                check_correlation = row["expected_correlation_with"])
                    positive_correlation = bool(row["positive_correlation"])

                    for check, check_value in checks.items():
                        if check == "check_single_values" and check_value:
                            if normalization_variable:
                                target_name = f"{output_name}_to_{normalization_variable}"
                                dataset[target_name] = dataset[output_name] / dataset[normalization_variable]
                            else:
                                target_name = output_name

                            test = (observed_min <= dataset[target_name]) & (dataset[target_name] <= observed_max)
                            if False in test and dataset[target_name].where(~test).dropna(dim='vid', how='all').dropna(dim='t', how='all').values.size > 0:
                                shown_df = dataset[target_name].where(~test).dropna(dim='vid', how='all').dropna(dim='t', how='all').to_dataframe()

                                warnings.warn(f"\n{RED}{target_name}{RESET} outside boundaries for {shown_df.unstack(level=1)}\n", UserWarning)
                                failed_tests += 1
                                failed_names.append(target_name)
                            else:
                                passed_tests += 1
                                passed_names.append(target_name)

                        if check == "check_collar" and check_value:
                            if normalization_variable:
                                target_name = f"{output_name}_to_{normalization_variable}"
                                dataset[target_name] = dataset[output_name] / dataset[normalization_variable].sum(dim="vid")
                            else:
                                target_name = output_name
                            
                            test_low = observed_min <= dataset[target_name].sel(vid=1)
                            test_high = dataset[target_name].sel(vid=1) <= observed_max
                            test = test_low & test_high
                            rejection_lower = round(((test_low == False).sum() / dataset[target_name].size).values * 100, 1)
                            rejection_higher = round(((test_high == False).sum() / dataset[target_name].size).values * 100, 1)
                            rejection_suffix = ""
                            if rejection_lower != 0:
                                min_error_time = min(test_low.t.where(~test_low).dropna(dim='t', how='all').t.values)
                                max_error_time = max(test_low.t.where(~test_low).dropna(dim='t', how='all').t.values)
                                rejection_suffix = rejection_suffix + f"{rejection_lower}% too-low ({min_error_time}-{max_error_time}h),"
                            if rejection_higher != 0:
                                min_error_time = min(test_high.t.where(~test_high).dropna(dim='t', how='all').t.values)
                                max_error_time = max(test_high.t.where(~test_high).dropna(dim='t', how='all').t.values)
                                rejection_suffix = rejection_suffix + f"{rejection_higher}% too-high ({min_error_time}-{max_error_time}h),"

                            if False in test and dataset[target_name].where(~test).dropna(dim='vid', how='all').dropna(dim='t', how='all').values.size > 0:
                                warnings.warn(f"\n{RED}{target_name}{RESET} outside boundaries for {dataset[target_name].where(~test).dropna(dim='vid', how='all').dropna(dim='t', how='all').sel(vid=1).to_dataframe()}\n", UserWarning)
                                failed_tests += 1
                                failed_names.append(f"collar_{target_name} : {rejection_suffix}")
                            else:
                                passed_tests += 1
                                passed_names.append(f"collar_{target_name}")

                        if check == "check_sum" and check_value:
                            total_dataset = dataset.sum(dim="vid")
                            if normalization_variable:
                                target_name = f"{output_name}_to_{normalization_variable}"
                                total_dataset[target_name] = total_dataset[output_name] / dataset[normalization_variable].sum(dim="vid")
                            else:
                                target_name = output_name

                            test_low = observed_min <= total_dataset[target_name]
                            test_high = total_dataset[target_name] <= observed_max
                            test = test_low & test_high

                            rejection_lower = round(((test_low == False).sum() / total_dataset[target_name].size).values * 100, 1)
                            rejection_higher = round(((test_high == False).sum() / total_dataset[target_name].size).values * 100, 1)
                            rejection_suffix = ""
                            if rejection_lower != 0:
                                min_error_time = min(test_low.t.where(~test_low).dropna(dim='t', how='all').t.values)
                                max_error_time = max(test_low.t.where(~test_low).dropna(dim='t', how='all').t.values)
                                rejection_suffix = rejection_suffix + f"{rejection_lower}% too-low ({min_error_time}-{max_error_time}h),"
                            if rejection_higher != 0:
                                min_error_time = min(test_high.t.where(~test_high).dropna(dim='t', how='all').t.values)
                                max_error_time = max(test_high.t.where(~test_high).dropna(dim='t', how='all').t.values)
                                rejection_suffix = rejection_suffix + f"{rejection_higher}% too-high ({min_error_time}-{max_error_time}h),"

                            if False in test and total_dataset[target_name].where(~test).dropna(dim='t', how='all').values.size > 0:
                                warnings.warn(f"\n{RED}Total {target_name}{RESET} outside boundaries for {total_dataset[target_name].where(~test).dropna(dim='t', how='all').to_dataframe()}\n", UserWarning)
                                failed_tests += 1
                                failed_names.append(f"sum_{target_name} : {rejection_suffix}")
                            else:
                                passed_tests += 1
                                passed_names.append(f"sum_{target_name}")

                        if check == "check_mean" and check_value:
                            if normalization_variable:
                                target_name = f"{output_name}_to_{normalization_variable}"
                                dataset[target_name] = dataset[output_name] / dataset[normalization_variable]
                            else:
                                target_name = output_name

                            total_dataset = dataset.mean(dim="vid")
                            
                            test_low = observed_min <= total_dataset[target_name]
                            test_high = total_dataset[target_name] <= observed_max
                            test = test_low & test_high

                            rejection_lower = round(((test_low == False).sum() / total_dataset[target_name].size).values * 100, 1)
                            rejection_higher = round(((test_high == False).sum() / total_dataset[target_name].size).values * 100, 1)
                            rejection_suffix = ""
                            if rejection_lower != 0:
                                min_error_time = min(test_low.t.where(~test_low).dropna(dim='t', how='all').t.values)
                                max_error_time = max(test_low.t.where(~test_low).dropna(dim='t', how='all').t.values)
                                rejection_suffix = rejection_suffix + f"{rejection_lower}% too-low ({min_error_time}-{max_error_time}h),"
                            if rejection_higher != 0:
                                min_error_time = min(test_high.t.where(~test_high).dropna(dim='t', how='all').t.values)
                                max_error_time = max(test_high.t.where(~test_high).dropna(dim='t', how='all').t.values)
                                rejection_suffix = rejection_suffix + f"{rejection_higher}% too-high ({min_error_time}-{max_error_time}h),"

                            if False in test and dataset[target_name].where(~test).dropna(dim='vid', how='all').dropna(dim='t', how='all').values.size > 0:
                                warnings.warn(f"\n{RED}Mean {target_name}{RESET} outside boundaries for {total_dataset[target_name].where(~test).dropna(dim='t', how='all').to_dataframe()}\n", UserWarning)
                                failed_tests += 1
                                failed_names.append(f"mean_{target_name} : {rejection_suffix}")
                            else:
                                passed_tests += 1
                                passed_names.append(f"mean_{target_name}")
                else:
                    print(f"{output_name} check is invalid")
                    failed_tests += 1

            print(f"\n{RED}FAILED : {failed_tests} {failed_names}{RESET}")
            print(f"{GREEN}PASSED : {passed_tests} {passed_names}{RESET}\n")

            show_more = input("Show related warnings? [y]/n")
            if show_more in ("y", ""):
                ct = 1
                for warning in logged_warnings:
                    print(f"\n{RED}Warning {ct}:{RESET}")
                    print(warning)
                    ct += 1


def scientific_formatter(x, pos):
    return f'{x:.0e}'


def plot_csv(csv_dirpath, csv_name, properties=None, stacked=False, twin=False):
    log = pd.read_csv(os.path.join(csv_dirpath, csv_name))

    units = log.iloc[0]

    # Ignore unit value for plots and initialization values for plots' readability
    log = log[3:].astype(float)

    plot_path = os.path.join(csv_dirpath, "plots")
    
    if os.path.isdir(plot_path) and not stacked:
        shutil.rmtree(plot_path)
        os.mkdir(plot_path)

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    if properties is None:
        properties = log.columns

    if stacked:
        fig, ax = plt.subplots()

    plot_number = 0
    twin_axes = {}
    twin_legends = {}

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colormap = plt.cm.tab20

    # Generate evenly spaced values between 0 and 1
    color_indices = np.linspace(0, 1, len(properties))

    # Sample the colormap to create the list of custom colors
    default_colors = [colormap(i) for i in color_indices]

    for prop in properties:
        if prop in log.columns and prop != "Unnamed: 0":
            if len(prop) > 15:
                label = prop[:15]
            else:
                label = prop
            if not stacked:
                fig, ax = plt.subplots()

            if stacked:
                if plot_number == 0 or not twin:
                    ax.plot(log.index.values, log[prop], label=label, c=default_colors[plot_number])
                else:
                    twin_axes[plot_number] = ax.twinx()
                    twin_axes[plot_number].yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
                    twin_axes[plot_number].tick_params(axis='y', rotation=45)
                    twin_axes[plot_number].spines['right'].set_position(('outward', (plot_number-1) * 70))
                    twin_axes[plot_number].plot(log.index.values, log[prop], label=label, c=default_colors[plot_number]) 
                    twin_legends[plot_number] = twin_axes[plot_number].legend(loc='center right', bbox_to_anchor=(1.2 + (plot_number - 1) * 0.23, 0.5), handlelength=1)
                    for text in twin_legends[plot_number].get_texts():
                        text.set_rotation(90)
            
            else:
                ax.plot(log.index.values, log[prop], label=label)
                ax.set_title(f"{prop} ({units.loc[prop]})")
                ax.set_xlabel("t (h)")
                #ax.ticklabel_format(axis='y', useOffset=True, style="sci", scilimits=(0, 0))
                fig.savefig(os.path.join(plot_path, prop + ".png"))
                plt.close()
            
            plot_number += 1

    if stacked:
        if twin:
            legend = ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5), handlelength=1)
            for text in legend.get_texts():
                text.set_rotation(90)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(-0.5, 0.5))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.tick_params(axis='y', rotation=45)
        #ax.set_title(f"Stack_of_{properties}")
        ax.set_xlabel("t (h)")
        #ax.ticklabel_format(axis='y', useOffset=True, style="sci", scilimits=(0, 0))
        if len(str(properties)) <= 20:
            filename = properties
        else:
            filename = "many_steps"

        fig.savefig(os.path.join(plot_path, f"Stack_of_{filename}.png"), bbox_inches="tight")
        plt.close()


def plot_csv_stackable(fig, ax, csv_dirpath, csv_name, property, std_prop=None, units = False, scatter=True):
    log = pd.read_csv(os.path.join(csv_dirpath, csv_name), sep=';')

    if units:
        units = log.iloc[0]

        # Ignore unit value
        log = log[1:].astype(float)

    if scatter:
        
        if std_prop:
            ax.errorbar(log["t"], log[property], yerr=log[std_prop], fmt=".", color='black', linestyle='')
            ax.scatter(log["t"], log[property], label=property, s=30)
        else:
            ax.scatter(log["t"], log[property], label=property, s=30)
    else:
        ax.plot(log["t"], log[property], label=property)

    return fig, ax


def plot_timeline_xarray_stackable(fig, ax, dataset, x_name, y_name, mean_and_std=True):
    if mean_and_std:
        ds_mean = dataset["y_name"].mean(dim="vid")
        ds_std = dataset["y_name"].std(dim="vid")
        ax.fill_between(dataset[x_name], (ds_mean - ds_std).values[0], (ds_mean + ds_std).values[0])
        ds_mean.plot.line(x=x_name, ax=ax, label=f"{y_name} over {x_name}")

    else:
        dataset[y_name].sum(dim="vid").plot.line(x=x_name, ax=ax, label=f"{y_name} over {x_name}")

    return fig, ax


def plot_xarray_vertical_bins(fig, ax, colors, grouped_ds, bins_center, prop, bin_z_width, mean_and_std=False):

    if mean_and_std:
        ax.barh(-bins_center, grouped_ds.mean()[prop].values, height=bin_z_width-0.001, color="g")

    else:
        bin_summed_ds = grouped_ds.sum()
        if isinstance(prop, list):
            left_pos = np.zeros_like(bins_center)
            left_neg = np.zeros_like(bins_center)
            for k in range(len(bins_center)):
                for p in prop:
                    if k == 0:
                        label=p
                    else:
                        label=None
                    if bin_summed_ds[p][k] >= 0:
                        ax.barh(-bins_center[k], bin_summed_ds[p][k], left=left_pos[k], label=label, height=bin_z_width-0.001, color=colors[p])
                        left_pos[k] += bin_summed_ds[p][k]
                    else:
                        ax.barh(-bins_center[k], bin_summed_ds[p][k], left=left_neg[k], label=label, height=bin_z_width-0.001, color=colors[p])
                        left_neg[k] += bin_summed_ds[p][k]

        else:
            ax.barh(-bins_center, bin_summed_ds[prop].values, height=bin_z_width-0.001, color='g')

    return fig, ax

def plot_compare_xarray_vertical_bins(fig, ax, grouped_ds, bins_center, prop, bin_z_width, colors, mean_and_std=False, special_case=False):
    z_centering = - 0.005
    for name, scenario_groups in grouped_ds.items():
        if special_case:
            ds = scenario_groups.sum()
            ds["Nitrate_Carbon_Costs"] = ds["Cumulative_Carbon_Costs"] / ds["Cumulative_Nitrogen_Uptake"].where(ds["Cumulative_Nitrogen_Uptake"] > 0.)
            ax.barh(-(bins_center+z_centering), ds["Nitrate_Carbon_Costs"].values, height=bin_z_width-0.001, label=name, alpha=1, color=colors[name])
            
        else:
            if mean_and_std:
                ax.barh(-(bins_center+z_centering), scenario_groups.mean()[prop], height=bin_z_width-0.001, label=name, alpha=1, color=colors[name])

            else:
                bin_summed_ds = scenario_groups.sum()
                ax.barh(-(bins_center+z_centering), bin_summed_ds[prop].values, height=bin_z_width-0.001, label=name, alpha=1, color=colors[name])
        z_centering += 0.01
    return fig, ax


def cnwheat_plot_csv(csv_dirpath):
    plot_path = os.path.join(csv_dirpath, "plots")

    if not os.path.isdir(csv_dirpath):
        os.mkdir(csv_dirpath)

    if os.path.isdir(plot_path):
        shutil.rmtree(plot_path)

    os.mkdir(plot_path)

    from fspmwheat import cnwheat_facade

    # --- Generate graphs from postprocessing files
    plt.ioff()
    delta_t = 3600
    df_elt = pd.read_csv(os.path.join(csv_dirpath, "elements_outputs.csv"))
    df_org = pd.read_csv(os.path.join(csv_dirpath, "organs_outputs.csv"))
    df_hz = pd.read_csv(os.path.join(csv_dirpath, "hiddenzones_outputs.csv"))
    df_SAM = pd.read_csv(os.path.join(csv_dirpath, "axes_outputs.csv"))
    df_soil = pd.read_csv(os.path.join(csv_dirpath, "soil_outputs.csv"))

    postprocessing_df_dict = {}
    pp_df_ax, pp_df_hz, pp_df_org, pp_df_elt, pp_df_soil = cnwheat_facade.CNWheatFacade.postprocessing(
                                axes_outputs_df=df_SAM,
                                hiddenzone_outputs_df=df_hz,
                                organs_outputs_df=df_org,
                                elements_outputs_df=df_elt,
                                soils_outputs_df=df_soil,
                                delta_t=delta_t)
    
    save_postprocessing = True
    if save_postprocessing:
        postprocessing_dirpath = os.path.join(csv_dirpath, "postprocessing")
        if not os.path.exists(postprocessing_dirpath):
            # Then we create it:
            os.mkdir(postprocessing_dirpath)

        pp_df_ax.to_csv(os.path.join(postprocessing_dirpath, "axes_postprocessing.csv"), na_rep='NA', index=False, float_format='%.{}f'.format(8))
        pp_df_hz.to_csv(os.path.join(postprocessing_dirpath, "hiddenzones_postprocessing.csv"), na_rep='NA', index=False, float_format='%.{}f'.format(8))
        pp_df_org.to_csv(os.path.join(postprocessing_dirpath, "organs_postprocessing.csv"), na_rep='NA', index=False, float_format='%.{}f'.format(8))
        pp_df_elt.to_csv(os.path.join(postprocessing_dirpath, "elements_postprocessing.csv"), na_rep='NA', index=False, float_format='%.{}f'.format(8))
        pp_df_soil.to_csv(os.path.join(postprocessing_dirpath, "soil_postprocessing.csv"), na_rep='NA', index=False, float_format='%.{}f'.format(8))

    cnwheat_facade.CNWheatFacade.graphs(
        axes_postprocessing_df=pp_df_ax,
        hiddenzones_postprocessing_df=pp_df_hz,
        organs_postprocessing_df=pp_df_org,
        elements_postprocessing_df=pp_df_elt,
        soils_postprocessing_df=pp_df_soil,
        graphs_dirpath=plot_path)

    # --- Additional graphs
    from cnwheat import tools as cnwheat_tools
    colors = ['blue', 'darkorange', 'green', 'red', 'darkviolet', 'gold', 'magenta', 'brown', 'darkcyan', 'grey',
              'lime']
    colors = colors + colors

    # 0) Phyllochron
    df_SAM = df_SAM[df_SAM['axis'] == 'MS']
    grouped_df = pp_df_hz[pp_df_hz['axis'] == 'MS'].groupby(['plant', 'metamer'])[['t', 'leaf_is_emerged']]
    leaf_emergence = {}
    for group_name, data in grouped_df:
        plant, metamer = group_name[0], group_name[1]
        if metamer == 3 or True not in data['leaf_is_emerged'].unique():
            continue
        leaf_emergence_t = data[data['leaf_is_emerged'] == True].iloc[0]['t']
        leaf_emergence[(plant, metamer)] = leaf_emergence_t

    phyllochron = {'plant': [], 'metamer': [], 'phyllochron': []}
    for key, leaf_emergence_t in sorted(leaf_emergence.items()):
        plant, metamer = key[0], key[1]
        if metamer == 4:
            continue
        phyllochron['plant'].append(plant)
        phyllochron['metamer'].append(metamer)
        prev_leaf_emergence_t = leaf_emergence[(plant, metamer - 1)]
        if df_SAM[(df_SAM['t'] == leaf_emergence_t) | (df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.count() == 2:
            phyllo_DD = df_SAM[(df_SAM['t'] == leaf_emergence_t)].sum_TT.values[0] - \
                        df_SAM[(df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.values[0]
        else:
            phyllo_DD = np.nan
        phyllochron['phyllochron'].append(phyllo_DD)

    if len(phyllochron['metamer']) > 0:
        fig, ax = plt.subplots()
        plt.xlim((int(min(phyllochron['metamer']) - 1), int(max(phyllochron['metamer']) + 1)))
        plt.ylim(ymin=0, ymax=150)
        ax.plot(phyllochron['metamer'], phyllochron['phyllochron'], color='b', marker='o')
        for i, j in zip(phyllochron['metamer'], phyllochron['phyllochron']):
            ax.annotate(str(int(round(j, 0))), xy=(i, j + 2), ha='center')
        ax.set_xlabel('Leaf number')
        ax.set_ylabel('Phyllochron (Degree Day)')
        ax.set_title('phyllochron')
        plt.savefig(os.path.join(plot_path, 'phyllochron' + '.PNG'))
        plt.close()

    # 2) LAI

    PLANT_DENSITY = {1: 250.}
    pp_df_elt['green_area_rep'] = pp_df_elt.green_area * pp_df_elt.nb_replications
    grouped_df = pp_df_elt[(pp_df_elt.axis == 'MS') & (pp_df_elt.element == 'LeafElement1')].groupby(['t', 'plant'])
    LAI_dict = {'t': [], 'plant': [], 'LAI': []}
    for name, data in grouped_df:
        t, plant = name[0], name[1]
        LAI_dict['t'].append(t)
        LAI_dict['plant'].append(plant)
        LAI_dict['LAI'].append(data['green_area_rep'].sum() * PLANT_DENSITY[plant])

    cnwheat_tools.plot_cnwheat_ouputs(pd.DataFrame(LAI_dict), 't', 'LAI', x_label='Time (Hour)', y_label='LAI',
                                      plot_filepath=os.path.join(plot_path, 'LAI.PNG'), explicit_label=False)

    


# Define function for string formatting of scientific notation
def sci_notation(num, just_print_ten_power=True, decimal_digits=0, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        if num != 0.:
            if num >= 1:
                exponent = int(ceil(log10(abs(num))))
            else:
                exponent = int(floor(log10(abs(num))))
        else:
            exponent = 0
    coeff = round(num / float(10 ** exponent), decimal_digits)

    if precision is None:
        precision = decimal_digits

    if num == 0:
        return r"${}$".format(0)

    if just_print_ten_power:
        return r"$10^{{{0:d}}}$".format(exponent)
    else:
        return r"${0:.{2}f}/cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


# Function that draws a colorbar:
def colorbar(title="Radius (m)", cmap='jet', lognorm=True, n_thicks_for_linear_scale=6, vmin=1e-12, vmax=1e3):
    """
    This function creates a colorbar for showing the legend of a plot.
    :param title: the name of the property to be displayed on the bar
    :param cmap: the name of the specific colormap in Python
    :param lognorm: if True, the scale will be a log scale, otherwise, it will be a linear scale
    :param n_thicks_for_linear_scale: the number of thicks to represent for a linear scale
    :param vmin: the min value of the color scale
    :param vmax: the max value of the color scale
    :return: the new colorbar object
    """

    # CREATING THE COLORBAR
    #######################

    # Creating the box that will contain the colorbar:
    fig, ax = plt.subplots(figsize=(36, 6))
    fig.subplots_adjust(bottom=0.5)

    _cmap = color.get_cmap(cmap)

    # If the bar is to be displayed with log scale:
    if lognorm:
        if vmin <= 0.:
            print("WATCH OUT: when making the colorbar, vmin can't be equal or below zero when lognorm is TRUE. "
                  "vmin has been turned to 1e-10 by default.")
            vmin = 1e-10
        # We create the log-scale color bar:
        norm = color.LogNorm(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax,
                                         cmap=cmap,
                                         norm=norm,
                                         orientation='horizontal')
    # Otherwise the colorbar is in linear scale:
    else:
        # We create the normal-scale color bar:
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        ticks = np.linspace(vmin, vmax, n_thicks_for_linear_scale)
        cbar = mpl.colorbar.ColorbarBase(ax,
                                         cmap=cmap,
                                         norm=norm,
                                         ticks=ticks,  # We specify a number of ticks to display
                                         orientation='horizontal')

    # In any case, we remove stupid automatic tick labels:
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    cbar.outline.set_linewidth(3)  # Thickness of the box lines
    cbar.set_label(title, fontsize=40, weight='bold', labelpad=-130)  # Adjust the caption under the bar

    # We specify the characteristics of the ticks:
    cbar.ax.tick_params(which="major",
                        direction="in",  # Position of the ticks in or out the bar
                        labelsize=0,  # Size of the text
                        length=20,  # Length of the ticks
                        width=5,  # Thickness of the ticks
                        pad=-60  # Distance between ticks and label
                        )
    cbar.ax.tick_params(which="minor",
                        direction="in",  # Position of the ticks in or out the bar
                        labelsize=0,  # Size of the text
                        length=10,  # Length of the ticks
                        width=3,  # Thickness of the ticks
                        pad=-60  # Distance between ticks and label
                        )

    # For adding minor ticks:
    ax.minorticks_on()
    # minorticks = [0.1, 0.2, 0.3]
    # ax.xaxis.set_ticks(minorticks, minor=True)
    # ax.yaxis.set_ticks(minorticks, minor=True)

    # Manually adding the labels of the ticks:
    ##########################################

    # If the bar is to be displayed with log scale:
    if lognorm:
        # We get the exponents of the powers of 10th closets from vmin and vmax:
        min10 = ceil(np.log10(vmin))
        max10 = floor(np.log10(vmax))
        # We calculate the interval to cover:
        n_intervals = int(abs(max10 - min10))

        # We initialize empty lists:
        list_number = []
        numbers_to_display = []
        x_positions = []

        # We start from the closest power of tenth equal or higher than vmin:
        number = 10 ** min10
        # And we start at a specific position from which we will add intervals for positioning the text:
        position = -0.012
        # We cover the range from vmin to vmax:
        for i in range(0, n_intervals):
            list_number.append(number)
            x_positions.append(position)
            number = number * 10
            position = position + 1 / float(n_intervals)
        # We correct the first position, if needed:
        x_positions[0] = 0.005

        # We create the list of strings in a scientific format for displaying the numbers on the colorbar:
        for number in list_number:
            # numbers_to_display.append("{:.0e}".format(number))
            numbers_to_display.append(sci_notation(number, just_print_ten_power=True))

    # Otherwise the colorbar is in linear scale:
    else:

        # We calculate the interval to cover:
        n_intervals = n_thicks_for_linear_scale - 1

        # We initialize empty lists:
        list_number = []
        numbers_to_display = []
        x_positions = []

        # We start from vmin:
        number = vmin
        # And we start at a specific position from which we will add intervals for positioning the text:
        position = -0.007
        # We cover the range from vmin to vmax:
        for i in range(0, n_intervals + 1):
            list_number.append(number)
            x_positions.append(position)
            number = number + (vmax - vmin) / float(n_intervals)
            position = position + 1 / float(n_intervals)
        # We correct the first position, if needed:
        x_positions[0] = 0.005

        # We create the list of strings in a scientific format for displaying the numbers on the colorbar:
        for number in list_number:
            # numbers_to_display.append("{:.0e}".format(number))
            numbers_to_display.append(sci_notation(number, decimal_digits=0, just_print_ten_power=False))
        # We remove first and last point, if needed:
        numbers_to_display[0] = ""
        numbers_to_display[-1] = ""

    # We cover each number to add on the colorbar:
    for i in range(0, len(numbers_to_display)):
        position = 'left'
        # We add the corresponding number on the colorbar:
        cbar.ax.text(x=x_positions[i],
                     y=0.4,
                     s=numbers_to_display[i],
                     va='top',
                     ha=position,
                     fontsize=40,
                     fontweight='bold',  # This doesn't change much the output, unfortunately...
                     transform=ax.transAxes)

    print("The colorbar has been made!")
    return fig


# Definition of a function that can resize a list of images and make a movie from it:
#------------------------------------------------------------------------------------
def resizing_and_film_making(outputs_path='outputs',
                             images_folder='root_images',
                             resized_images_folder='root_images_resized',
                             film_making=True,
                             film_name="root_movie.gif",
                             image_transforming=True,
                             resizing=False, dividing_size_by=1.,
                             colorbar_option=True, colorbar_position=1,
                             colorbar_title="Radius (m)",
                             colorbar_cmap='jet', colorbar_lognorm=True,
                             n_thicks_for_linear_scale=6,
                             vmin=1e-6, vmax=1e0,
                             time_printing=True, time_position=1,
                             time_step_in_days=1., sampling_frequency=1, fps=24,
                             title=""):
    """
    This function enables to resize some images, add a time indication and a colorbar on them, and create a movie from it.
    :param outputs_path: the general path in which the folders containing images are located
    :param images_folder: the name of the folder in which images have been stored
    :param resized_images_folder: the name of the folder to create, in which transformed images will be saved
    :param film_making: if True, a movie will be created from the original or transformed images
    :param film_name: the name of the movie file to be created
    :param image_transforming: if True, images will first be transformed
    :param resizing: if True, images can be resized
    :param dividing_size_by: the number by which the original dimensions will be divided to create the resized image
    :param colorbar_option: if True, a colorbar will be added
    :param colorbar_position: the position of the colorbar (1 = bottom right, 2 = bottom middle),
    :param colorbar_title: the name of the property to be displayed on the bar
    :param colorbar_cmap: the name of the specific colormap in Python
    :param colorbar_lognorm: if True, the scale will be a log scale, otherwise, it will be a linear scale
    :param n_thicks_for_linear_scale: the number of thicks to represent for a linear scale
    :param vmin: the min value of the color scale
    :param vmax: the max value of the color scale
    :param time_printing: if True, a time indication will be calculated and displayed on the image
    :param time_position: the position of the time indication (1 = top left for root graphs, 2 = bottom right for z-barplots)
    :param time_step_in_days: the original time step at which MTG images were generated
    :param sampling_frequency: the frequency at which images should be picked up and included in the transformation/movie (i.e. 1 image every X images)
    :param fps: frames per second for the .gif movie to create
    :param title: the name of the movie file
    :return:
    """

    images_directory = os.path.join(outputs_path, images_folder)
    resized_images_directory = os.path.join(outputs_path, resized_images_folder)

    # Getting a list of the names of the images found in the directory "video":
    filenames = [f for f in os.listdir(images_directory) if ".png" in f]
    filenames = sorted(filenames)

    # We define the final number of images that will be considered, based on the "sampling_frequency" variable:
    number_of_images = floor(len(filenames) / float(sampling_frequency))

    if colorbar_option:
        path_colorbar = os.path.join(outputs_path, 'colorbar.png')
        # We create the colorbar:
        bar = colorbar(title=colorbar_title,
                       cmap=colorbar_cmap,
                       lognorm=colorbar_lognorm,
                       n_thicks_for_linear_scale=n_thicks_for_linear_scale,
                       vmin=vmin, vmax=vmax)
        # We save it in the output directory:
        bar.savefig(path_colorbar, facecolor="None", edgecolor="None")
        # We reload the bar with Image package:
        bar = Image.open(path_colorbar)
        new_size = (1200, 200)
        bar = bar.resize(new_size)
        if colorbar_position == 1:
            box_colorbar = (-120, 1070)
        elif colorbar_position == 2:
            box_colorbar = (-120, 870)

    # 1. COMPRESSING THE IMAGES:
    if image_transforming:
        # If this directory doesn't exist:
        if not os.path.exists(resized_images_directory):
            # Then we create it:
            os.mkdir(resized_images_directory)
        else:
            # Otherwise, we delete all the images that are already present inside:
            for root, dirs, files in os.walk(resized_images_directory):
                for file in files:
                    os.remove(os.path.join(root, file))

        # We modify each image:
        print("Transforming the images and copying them into the directory 'root_images_resized'...")
        # We initialize the counts:
        number = 0
        count = 0
        remaining_images = number_of_images

        # We calculate the dimensions of the new images according to the variable size_division:
        dimensions = (int(1600 / dividing_size_by), int(1055 / dividing_size_by))

        # We cover each image in the directory:
        for filename in filenames:

            # The time is calculated:
            time_in_days = time_step_in_days * (number_of_images - remaining_images) * sampling_frequency
            # The count is increased:
            count += 1
            # If the count corresponds to the target number, the image is added to the gif:
            if count == sampling_frequency:
                print("Transforming the images - please wait:", str(int(remaining_images)), "image(s) left")

                # Opening the image to modify:
                im = Image.open(filename)

                # Adding colorbar:
                if colorbar_option:
                    im.paste(bar, box_colorbar, bar.convert('RGBA'))

                # Adding text:
                if time_printing:

                    # OPTION 1 FOR ROOT SYSTEMS:
                    # ---------------------------
                    if time_position == 1:
                        draw = ImageDraw.Draw(im)
                        # For time display:
                        # ------------------
                        # draw.text((x, y),"Sample Text",(r,g,b))
                        font_time = ImageFont.truetype("./timesbd.ttf", 35)
                        # See a list of available fonts on: https://docs.microsoft.com/en-us/typography/fonts/windows_10_font_list
                        (x1, y1) = (40, 40)
                        time_text = "t = " + str(int(floor(time_in_days))) + " days"
                        draw.rectangle((x1 - 10, y1 - 10, x1 + 200, y1 + 50), fill=(255, 255, 255, 200))
                        draw.text((x1, y1), time_text, fill=(0, 0, 0), font=font_time)
                        # draw.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

                    # OPTION 2 FOR Z BARPLOTS:
                    # -----------------------
                    if time_position == 2:
                        draw = ImageDraw.Draw(im)
                        # For time display:
                        # ------------------
                        # draw.text((x, y),"Sample Text",(r,g,b))
                        font_time = ImageFont.truetype("./timesbd.ttf", 20)
                        # See a list of available fonts on: https://docs.microsoft.com/en-us/typography/fonts/windows_10_font_list
                        (x1, y1) = (650, 420)
                        time_text = "t = " + str(int(floor(time_in_days))) + " days"
                        draw.rectangle((x1 - 10, y1 - 10, x1 + 200, y1 + 30), fill=(255, 255, 255, 0))
                        draw.text((x1, y1), time_text, fill=(0, 0, 0), font=font_time)
                        # draw.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

                    # # For caption of colorbar:
                    # # ------------------------
                    # title_text_position=(100,1020)
                    # font_title = ImageFont.truetype("./timesbd.ttf", 23)
                    # # text_color =(200,200,200,255) #RGBA, the last digit corresponds to alpha canal (transparency)
                    # # draw.text(title_text_position, title, (0, 0, 0), font=font_title, fill=text_color)
                    # draw.text(title_text_position, title, (0, 0, 0), font=font_title)

                # Transforming the image:
                if resizing:
                    im_to_print = im.resize(dimensions, resample=0)
                else:
                    im_to_print = im

                # We get the last characters of the path of the file, which correspond to the actual name 'rootXXXXX':
                name = filename[-13:-4] + '.png'
                # Saving the new image:
                image_name = os.path.join(resized_images_directory, name)
                im_to_print.save(image_name, quality=20, optimize=True)

                # We update the local counts:
                number = number + 1
                remaining_images = remaining_images - 1
                count = 0
        print("The new images have been transformed!")

    # 2. CREATING THE VIDEO FILE:
    if film_making:

        print("Making the video...")

        with imageio.get_writer(os.path.join(outputs_path, film_name), mode='I', fps=fps) as writer:
            if image_transforming:
                filenames = [f for f in os.listdir(images_directory) if ".png" in f]
                filenames = sorted(filenames)
                sampling_frequency = 1
            else:
                filenames = [f for f in os.listdir(images_directory) if ".png" in f]
                filenames = sorted(filenames)
                sampling_frequency = sampling_frequency
            remaining_images = floor(len(filenames) / float(sampling_frequency)) + 1
            print(remaining_images, "images are considered at this stage.")
            # We add the first image:
            filename = filenames[0]
            image = imageio.imread(str(filename))
            writer.append_data(image)
            # We reduce the number of images left:
            remaining_images = remaining_images - 1
            # We start the count at 0:
            count = 0
            # We cover each image in the directory:
            for filename in filenames:
                # The count is increased:
                count += 1
                # If it corresponds to the target number, the image is added to the gif:
                if count == sampling_frequency:
                    print("Creating the video - please wait:", str(int(remaining_images)), "image(s) left")
                    image = imageio.imread(str(filename))
                    writer.append_data(image)
                    remaining_images = remaining_images - 1
                    # We reset the count to 0:
                    count = 0
        print("The video has been made!")

    return


# Definition of a function that can create a similar movie for different scenarios' outputs
#-------------------------------------------------------------------------------------------
def resizing_and_film_making_for_scenarios(general_outputs_folder='outputs',
                                           images_folder="root_images",
                                           resized_images_folder="root_images_resided",
                                           scenario_numbers=[1, 2, 3, 4],
                                           film_making=True,
                                           film_name="root_movie.gif",
                                           image_transforming=True, resizing=False, dividing_size_by=1.,
                                           colorbar_option=True, colorbar_position=1,
                                           colorbar_title="Radius (m)",
                                           colorbar_cmap='jet', colorbar_lognorm=True,
                                           n_thicks_for_linear_scale=6,
                                           vmin=1e-6, vmax=1e0,
                                           time_printing=True, time_position=1,
                                           time_step_in_days=1., sampling_frequency=1, frames_per_second=24,
                                           title=""
                                           ):
    """
    This function creates the same type of movie in symetric outputs generated from different scenarios.
    :param general_outputs_folder: the path of the general foleder, in which respective output folders from different scenarios have been recorded
    :param images_folder: the name of the images folder in each scenario
    :param resized_images_folder: the image of the transformed images folder in each scenario
    :param scenario_numbers: a list of numbers corresponding to the different scenarios to consider
    :[other parameters]: [cf the parameters from the function 'resizing_and_film_making']
    :return:
    """

    for i in scenario_numbers:
        scenario_name = 'Scenario_%.4d' % i
        scenario_path = os.path.join(general_outputs_folder, scenario_name)

        print("")
        print("Creating a movie for", scenario_name, "...")

        resizing_and_film_making(outputs_path=scenario_path,
                                 images_folder=images_folder,
                                 resized_images_folder=resized_images_folder,
                                 film_making=film_making,
                                 film_name=film_name,
                                 sampling_frequency=sampling_frequency, fps=frames_per_second,
                                 time_step_in_days=time_step_in_days,
                                 image_transforming=image_transforming,
                                 time_printing=time_printing, time_position=time_position,
                                 colorbar_option=colorbar_option, colorbar_position=colorbar_position,
                                 colorbar_title=colorbar_title,
                                 colorbar_cmap=colorbar_cmap, colorbar_lognorm=colorbar_lognorm,
                                 n_thicks_for_linear_scale=n_thicks_for_linear_scale,
                                 vmin=vmin, vmax=vmax,
                                 resizing=resizing, dividing_size_by=dividing_size_by,
                                 title=title)

    return


def xarray_deep_learning(dataset, mtg, global_state_extracts, global_flow_extracts, state_extracts, flow_extracts,
                         output_dir="", global_sensitivity=False, global_plots=False, plot_architecture=False,
                         STM_clustering=False):
    if global_sensitivity:
        # TERMINAL SENSITIVITY ANALYSIS
        # TODO : general sensitivity analysis on time-series data, but issue of post simulation Sensitivity Methods not existing
        # Global sensitivity analysis at the end of the simulation for now
        # Using a linear regression

        print("     [INFO] Performing regression sensitivity on model final global states...")
        regression_analysis(dataset=dataset, output_path=output_dir, extract_prop=global_state_extracts)

    if global_plots:
        # PLOTTING GLOBAL OUTPUTS
        print("     [INFO] Plotting global properties...")
        plot_xr(datasets=dataset, selection=list(global_state_extracts.keys()))
        plot_xr(datasets=dataset, selection=list(global_flow_extracts.keys()))

    if plot_architecture:
        # PLOTTING ARCHITECTURED VID LEGEND
        print("     [INFO] Plotting topology and coordinate map...")

        custom_colorbar(min(mtg.properties()["index"].values()), max(mtg.properties()["index"].values()),
                        unit="Vid number")

        scene = pgl.Scene()
        scene += plot_mtg(mtg,
                          prop_cmap="v",
                          lognorm=False,  # to avoid issues with negative values
                          vmin=min(mtg.properties()["struct_mass"].keys()),
                          vmax=max(mtg.properties()["struct_mass"].keys()))
        pgl.Viewer.display(scene)
        pgl.Viewer.saveSnapshot(output_dir + "/vid_map.png")

    if STM_clustering:
        # RUNNING STM CLUSTERING AND SENSITIVITY ANALYSIS
        # For some reason, dataset should be loaded before umap, and the run() call should be made at the end of
        # the workflow because tkinter locks everything
        # TODO : adapt to sliding windows along roots ?
        print("     [INFO] Performing local organs' physiology clustering...")
        pool_locals = {}
        pool_locals.update(state_extracts)
        pool_locals.update(flow_extracts)
        run_analysis(file=dataset, output_path=output_dir, extract_props=pool_locals)


def CN_balance_animation_pipeline(dataset, outputs_dirpath, fps, C_balance=True, target_vid=None):
    print("     [INFO] Producing balance animations...")

    if C_balance:
        bar_balance_xarray_animations(dataset, output_dirpath=outputs_dirpath, pool="C_hexose_root", balance_dict=balance_dicts_C["hexose"], fps=fps, fixed_ylim=None, target_vid=target_vid)

        bar_balance_xarray_animations(dataset, output_dirpath=outputs_dirpath, pool="Labile_Nitrogen",
                                      balance_dict=balance_dicts_no_C["labile_N"],
                                      fps=fps, fixed_ylim=None, target_vid=target_vid)

        bar_balance_xarray_animations(dataset, output_dirpath=outputs_dirpath, pool="AA", balance_dict=balance_dicts_C["AA"],
                          fps=fps, fixed_ylim=5e-10, target_vid=target_vid)

        bar_balance_xarray_animations(dataset, output_dirpath=outputs_dirpath, pool="Nm", balance_dict=balance_dicts_C["Nm"],
                          fps=fps, fixed_ylim=3e-10, target_vid=target_vid)
    else:
        bar_balance_xarray_animations(dataset, output_dirpath=outputs_dirpath, pool="Labile_Nitrogen",
                                      balance_dict=balance_dicts_no_C["labile_N"],
                                      fps=fps, fixed_ylim=3e-9, target_vid=target_vid)
        
        bar_balance_xarray_animations(dataset, output_dirpath=outputs_dirpath, pool="AA",
                                      balance_dict=balance_dicts_no_C["AA"],
                                      fps=fps, fixed_ylim=5e-10, target_vid=target_vid)

        bar_balance_xarray_animations(dataset, output_dirpath=outputs_dirpath, pool="Nm",
                                      balance_dict=balance_dicts_no_C["Nm"],
                                      fps=fps, fixed_ylim=3e-10, target_vid=target_vid)
    
    print("     [INFO] Finished")


def pie_balance_xarray_animations(dataset, output_dirpath, pool, balance_dict, input_composition=False, fps=15):

    used_dataset = dataset[list(balance_dict.keys())].sum(dim="vid")

    for name, meta in balance_dict.items():
        if meta["type"] == "output":
            used_dataset[name] = - used_dataset[name] * meta["conversion"]
        else:
            used_dataset[name] = used_dataset[name] * meta["conversion"]

    only_inputs = used_dataset.where(used_dataset > 0., 0.).to_array()
    only_outputs = - used_dataset.where(used_dataset < 0., 0.).to_array()

    if input_composition:
        fig, ax = plt.subplots(3, 1)
    else:
        fig, ax = plt.subplots(2, 1)

    fig.set_size_inches(10.5, 18.5)
    colors = np.array([np.random.rand(3,) for k in range(len(balance_dict.keys()))])
    def update(time):
        ax[0].clear()
        to_plot = np.array(only_inputs.sel(t=time)).reshape(1, -1)[0]
        ds_mean = dataset[pool].mean(dim="vid")
        ds_std = dataset[pool].std(dim="vid")
        ax[0].plot([time, time], [0, ds_mean.max()], c="r")
        ax[0].fill_between(ds_mean.t, (ds_mean-ds_std).values[0], (ds_mean+ds_std).values[0])
        ds_mean.plot.line(x="t", ax=ax[0], c="b")
        if not input_composition:
            ax[0].set_title(f"Input flows : {'{:.2E}'.format(np.sum(to_plot))} mol.s-1")

        ax[1].clear()
        to_plot = np.array(only_outputs.sel(t=time)).reshape(1, -1)[0]
        labels = np.array(list(balance_dict.keys()))[to_plot > 0.]
        ax[1].pie(to_plot[to_plot > 0.], startangle=0, colors=colors[to_plot > 0.])
        ax[1].set_title(f"Ouput flows : {'{:.2E}'.format(np.sum(to_plot))} mol.s-1")
        ax[1].legend(labels=labels, loc='best', bbox_to_anchor=(0.85, 1.025))

        if input_composition:
            ax[2].clear()
            to_plot = np.array(only_inputs.sel(t=time)).reshape(1, -1)[0]
            labels = np.array(list(balance_dict.keys()))[to_plot > 0.]
            ax[2].pie(to_plot[to_plot > 0.], startangle=0, colors=colors[to_plot > 0.])
            ax[2].set_title(f"Input flows : {'{:.2E}'.format(np.sum(to_plot))} mol.s-1")
            ax[2].legend(labels=labels, loc='best', bbox_to_anchor=(0.85, 1.025))

    animation = FuncAnimation(fig, update, frames=only_outputs.t[1:], repeat=False)
    FFwriter = FFMpegWriter(fps=fps, codec="mpeg4", bitrate=5000)
    animation.save(os.path.join(output_dirpath, f"MTG_properties\MTG_properties_raw\{pool}_pies.mp4"), writer=FFwriter, dpi=100)

y_limits = [1e-10 for k in range(100)]
    
def bar_balance_xarray_animations(dataset, output_dirpath, pool, balance_dict, input_composition=False, fps=15, fixed_ylim=None, target_vid=None):
    if target_vid:
        filtered_dataset = filter_dataset(d=dataset, vids=[target_vid])
    else:
        filtered_dataset = dataset

    used_dataset = (filtered_dataset[list(balance_dict.keys())] / filtered_dataset.length).sum(dim="vid") 

    for name, meta in balance_dict.items():
        if meta["type"] == "output":
            used_dataset[name] = - used_dataset[name] * meta["conversion"]
        else:
            used_dataset[name] = used_dataset[name] * meta["conversion"]

    only_inputs = used_dataset.where(used_dataset > 0., 0.).to_array()
    only_outputs = - used_dataset.where(used_dataset < 0., 0.).to_array()

    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(balance_dict)))
    prop_colors = {k: c for k, c in zip(balance_dict.keys(), colors)}
    

    fig, ax = plt.subplots(2, 1)

    fig.set_size_inches(10.5, 18.5)

    
    def update(time):
        global y_limits
        ax[0].clear()
        to_plot = np.array(only_inputs.sel(t=time)).reshape(1, -1)[0]
        ds_mean = filtered_dataset[pool].mean(dim="vid")
        ds_std = filtered_dataset[pool].std(dim="vid")
        ax[0].plot([time, time], [0, ds_mean.max()], c="r")
        ax[0].fill_between(ds_mean.t, (ds_mean-ds_std).values[0], (ds_mean+ds_std).values[0])
        ds_mean.plot.line(x="t", ax=ax[0], c="b")
        if not input_composition:
            ax[0].set_title(f"Input flows : {'{:.2E}'.format(np.sum(to_plot))} mol.s-1")

        ax[1].clear()
        to_plot = np.array(only_inputs.sel(t=time)).reshape(1, -1)[0]
        labels = np.array(list(balance_dict.keys()))[to_plot > 0.]
        to_plot = to_plot[to_plot > 0.]
        bottom = 0
        for k in range(len(to_plot)):
            ax[1].bar("Input flows", to_plot[k], label=labels[k], color=prop_colors[labels[k]], bottom=bottom)
            bottom += to_plot[k]

        y_limits = y_limits[1:] + [bottom]

        to_plot = np.array(only_outputs.sel(t=time)).reshape(1, -1)[0]
        labels = np.array(list(balance_dict.keys()))[to_plot > 0.]
        to_plot = to_plot[to_plot > 0.]
        bottom = 0
        for k in range(len(to_plot)):
            ax[1].bar("Output flows", to_plot[k], label=labels[k], color=prop_colors[labels[k]], bottom=bottom)
            bottom += to_plot[k]
        
        y_limits = y_limits[1:] + [bottom]

        if fixed_ylim:
            ax[1].set_ylim(0, fixed_ylim)
        else:
            ax[1].set_ylim(0, np.mean(y_limits)*2)
        
        ax[1].legend(loc='best', bbox_to_anchor=(0.85, 1.025))

    animation = FuncAnimation(fig, update, frames=only_outputs.t[1:], repeat=False)
    FFwriter = FFMpegWriter(fps=fps, codec="mpeg4", bitrate=5000)
    if target_vid:
        filename = f"MTG_properties\MTG_properties_raw\{pool}_bars_on_{target_vid}.mp4"
    else:
        filename = f"MTG_properties\MTG_properties_raw\{pool}_bars.mp4"

    animation.save(os.path.join(output_dirpath, filename), writer=FFwriter, dpi=100)

def surface_repartition(dataset, output_dirpath, fps):

    to_plot = dataset[["distance_from_tip", "volume", "root_exchange_surface"]]
    to_plot["normalized_exchange_surface"] = to_plot.root_exchange_surface / to_plot.volume

    fig, ax = plt.subplots()
    fig.set_size_inches(10.5, 10.5)

    def update(time):
        ax.clear()
        time_step_data = to_plot.sel(t=time).dropna(dim="vid")
        ax.scatter(time_step_data.distance_from_tip.values[0], time_step_data.normalized_exchange_surface.values[0],
                   c=time_step_data.vid)
        ax.set_xlabel("distance_to_tip (m)")
        ax.set_ylabel("normalized_exchange_surface (m2.m-3)")
        ax.set_title(f"time = {time}")

    animation = FuncAnimation(fig, update, frames=to_plot.t[1:], repeat=False)
    FFwriter = FFMpegWriter(fps=fps, codec="mpeg4", bitrate=5000)
    animation.save(os.path.join(output_dirpath, f"MTG_properties\MTG_properties_raw\surface_scatter.mp4"), writer=FFwriter,
                   dpi=100)


def apex_zone_contribution(d, output_dirpath="", apex_zone_length=0.01, flow="", summed_input="", color_prop="", plotting=True):
    """
    Description : Computes two plots. First is the apex zone contribution to the overall selected zone. Apex zone length is user defined.
    Second plot shows for all vids summed when the apices group outperform the other segements relative to the length they represent.
    """

    apex_zone = d.where(d["distance_from_tip"] <= apex_zone_length, 0.)
    apex_proportion = apex_zone[flow].sum(dim="vid") / d[flow].sum(dim="vid")
    length_proportion = apex_zone["length"].sum(dim="vid") / d["length"].sum(dim="vid")

    if plotting:
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(10.5, 18.5)
        
        apex_proportion.plot.line(x="t", ax=ax[0], label=f"{flow} proportion")
        length_proportion.plot.line(x="t", ax=ax[0], label="length proportion")
        ax[0].legend()
        ax[0].set_ylabel("proportion")

        apices_outperform = 100 * (apex_proportion - length_proportion) / length_proportion

        m = ax[1].scatter(d[summed_input].sum(dim="vid") / d["struct_mass"].sum(dim="vid"), apices_outperform, c=apex_zone[color_prop].mean(dim="vid"))
        fig.colorbar(m, ax=ax[1], label=color_prop)
        ax[1].set_xlabel("Summed input par mass unit : " + summed_input + " (mol.g-1.s-1)")
        ax[1].set_ylabel("Outperforming of mean per length exchanges (%)")
        ax[1].legend()

        fig.savefig(os.path.join(output_dirpath, f"apex_contribution_{flow}.png"))
        plt.close()

    return apex_proportion, length_proportion

def z_zone_contribution(fig, ax, dataset, zmin, zmax, flow, scenario="", mean_proportion=False,
                                                                    per_surface=False,
                                                                    per_length=False):

    z_zone = filter_dataset(d=dataset, prop="z2", propmin=zmin, propmax=zmax)

    if mean_proportion:
        z_proportion = z_zone[flow].mean(dim="vid") / dataset[flow].mean(dim="vid")
    else:
        z_proportion = z_zone[flow].sum(dim="vid") / dataset[flow].sum(dim="vid")

    length_proportion = z_zone["length"].sum(dim="vid") / dataset["length"].sum(dim="vid")
    surface_proportion = z_zone["root_exchange_surface"].sum(dim="vid") / dataset["root_exchange_surface"].sum(dim="vid")
    
    z_proportion.plot.line(x="t", ax=ax, label=f"{flow} proportion {scenario}")
    length_proportion.plot.line(x="t", ax=ax, label=f"length proportion {scenario}")
    #surface_proportion.plot.line(x="t", ax=ax, label=f"surface proportion {scenario}")



def trajectories_plot(dataset, output_dirpath, x, y, color=None, fps=15):
    fig, ax = plt.subplots()
    fig.set_size_inches(10.5, 10.5)
    
    def update(time):
        ax.clear()
        time_step_data = dataset.sel(t=time).dropna(dim="vid")
        #ax.set_xlim(dataset[x].min(), dataset[x].max())
        #ax.set_ylim(dataset[y].min(), dataset[y].max())
        ax.set_xlim(time_step_data[x].mean(dim="vid") - 3 * time_step_data[x].std(dim="vid"),
                    time_step_data[x].mean(dim="vid") + 3 * time_step_data[x].std(dim="vid"))

        ax.set_ylim(time_step_data[y].mean(dim="vid") - 3* time_step_data[y].std(dim="vid"),
                    time_step_data[y].mean(dim="vid") + 3*time_step_data[y].std(dim="vid"))
        if color is None:
            ax.scatter(time_step_data[x].values[0], time_step_data[y].values[0], c=time_step_data.vid)
        else:
            ax.scatter(time_step_data[x].values[0], time_step_data[y].values[0],
                       c=time_step_data[color].values[0])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"time = {time}")

    animation = FuncAnimation(fig, update, frames=dataset.t, repeat=False)
    FFwriter = FFMpegWriter(fps=15, codec="mpeg4", bitrate=5000)
    animation.save(os.path.join(output_dirpath, f"MTG_properties\MTG_properties_raw\{y} f({x}) C {color}_scatter.mp4"),
                   writer=FFwriter, dpi=100)


def compare_to_exp_biomass_pipeline(dataset, output_path):

    fig, ax = plt.subplot()

    plot_csv_stackable(fig, ax, "inputs/postprocessing", "Drew_biomass_si.csv", property="total_root_biomass_control", std_prop="total_root_biomass_control_std")
    plot_csv_stackable(fig, ax, "inputs/postprocessing", "Drew_biomass_si.csv", property="total_root_biomass_patch", std_prop="total_root_biomass_patch_std")

    ax.legend()
    ax.set_title(f"compare to biomass from Drew 1975")
    ax.set_xlabel("t (h)")
    ax.ticklabel_format(axis='y', useOffset=True, style="sci", scilimits=(0, 0))
    fig.savefig(os.path.join(output_path, "compare_to_exp_biomass.png"))
    plt.close()


def filter_dataset(d, scenario=None, time=None, tmin=None, tmax=None, vids=[], only_keep=None, prop=None, propmin=None, propmax=None, propis=None):
    
    if scenario:
        
        d = d.where(d.scenario == scenario).sum(dim="scenario")

    if only_keep:
        d = d[only_keep]

    if time:
        d = d.where(d.t == time).sum(dim="t")
    else:
        if tmin:
            d = d.where(d.t >= tmin)

        if tmax:
            d = d.where(d.t <= tmax)

    if len(vids) > 0:
        condition = False
        for vid in vids:
            condition = condition or d.vid == vid
        d = d.where(condition)
    
    if propmin and prop:
        d = d.where(d[prop] >= propmin)
    
    if propmax and prop:
        d = d.where(d[prop] <= propmax)

    if propis and prop:
        d = d.where(d[prop] == propis)

    return d


def open_and_merge_datasets(scenarios, root_outputs_path = "outputs"):
    print("         [INFO] Openning xarrays...")

    default_path_in_outputs = "MTG_properties/MTG_properties_raw/merged.nc"

    per_scenario_files = [os.path.join(root_outputs_path, name, default_path_in_outputs) for name in scenarios]
    
    if len(per_scenario_files) == 1:
        dataset = xr.open_dataset(per_scenario_files[0], engine="netcdf4")
        print("         [INFO] Finished")
        return dataset
    else:
        inidvidual_datasets = [xr.open_dataset(fp) for fp in per_scenario_files]
        datasets_with_new_dim = []
        for i, ds in enumerate(inidvidual_datasets):
            ds_expanded = ds.expand_dims("scenario")
            ds_expanded["scenario"] = [scenarios[i]]
            datasets_with_new_dim.append(ds_expanded)

        # Step 3: Combine the datasets along the new dimension
        merged_dataset = xr.concat(datasets_with_new_dim, dim="scenario", join="outer")
        print("         [INFO] Finished")
        return merged_dataset


def pipeline_z_bins_plots(dataset, output_path):
    fig, ax = plt.subplots(2, 1, figsize=(9, 16))

    dataset = filter_dataset(dataset, only_keep=["z2", "NAE", "hexose_exudation", "AA_synthesis", "amino_acids_consumption_by_growth"], 
                             tmin=1000, tmax=1024)
    
    z_min = dataset["z2"].max()
    depth_bins = np.arange(0, z_min, 0.01)
    bins_center = (depth_bins[:-1] + depth_bins[1:]) / 2

    grouped_ds = dataset.groupby_bins("z2", depth_bins)

    plot_xarray_vertical_bins(fig, ax[0], grouped_ds, bins_center=bins_center, prop="NAE", bin_z_width=0.01, mean_and_std=True, tmin=1000, tmax=1024)
    plot_xarray_vertical_bins(fig, ax[1], grouped_ds, bins_center=bins_center, prop=["hexose_exudation", "AA_synthesis"], right=False, bin_z_width=0.01, time=1000)
    plot_xarray_vertical_bins(fig, ax[1], grouped_ds, bins_center=bins_center, prop=["amino_acids_consumption_by_growth", "AA_synthesis"], right=True, bin_z_width=0.01, time=1000)

    ax[1].legend()
    fig.savefig(os.path.join(output_path, "NAE_depth_bins.png"))
    plt.close()

def pipeline_z_bins_animations(dataset, output_path, prop, metabolite, t_start=400, t_stop=450, fps=15, bin_z_width=0.01, mean_and_std=True, step=1, stride=1, x_min=0, x_max=1):
    print(f"    [INFO] Starting vertical bins animations for {metabolite} balance to explain {prop}")
    fig, ax = plt.subplots(2, 1, figsize=(9, 16))

    inputs_and_outputs = list(balance_dicts[metabolite].keys())
    dataset = filter_dataset(dataset, only_keep=inputs_and_outputs + [prop, "z2"], 
                             tmin=t_start, tmax=t_stop)
    
    for name, meta in balance_dicts[metabolite].items():
        if meta["type"] == "input":
            dataset[name] *= -meta["conversion"] 
        elif meta["type"] == "output":
            dataset[name] *= meta["conversion"] 

    times_to_animate = [int(t) for t in np.arange(t_start, t_stop, 1.) if (t in dataset.t.values and t%step==0)]
    
    z_min = dataset["z2"].max()
    depth_bins = np.arange(0, z_min, 0.01)
    bins_center = (depth_bins[:-1] + depth_bins[1:]) / 2

    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(inputs_and_outputs)))
    prop_colors = {k: c for k, c in zip(inputs_and_outputs, colors)}

    def update(time):
        print(f"        {time - t_start+1} / {t_stop - t_start}", end='\r', flush=True)
        [a.clear() for a in ax]
        if stride == 1:
            grouped_ds = dataset.isel(t=time).groupby_bins("z2", depth_bins)
        else:
            grouped_ds = dataset.isel(t=slice(time - int(stride/2), time + int(stride/2))).groupby_bins("z2", depth_bins)
        plot_xarray_vertical_bins(fig, ax[0], prop_colors, grouped_ds=grouped_ds, bins_center=bins_center, prop=prop, bin_z_width=bin_z_width, mean_and_std=mean_and_std)
        plot_xarray_vertical_bins(fig, ax[1], prop_colors, grouped_ds=grouped_ds, bins_center=bins_center, prop=inputs_and_outputs, bin_z_width=bin_z_width, mean_and_std=mean_and_std)
        
        fontsize = 15

        ax[0].set_xlim((x_min, x_max))
        ax[0].set_ylim((-0.20, 0.))
        ax[0].set_ylabel("depth (m)", fontsize=fontsize)
        ax[0].set_xlabel(f"{prop}", fontsize=fontsize)

        xlim = max(np.abs(ax[1].get_xlim()))
        xlim = 5e-10
        ax[1].set_xlim((0, x_max))
        ax[1].set_ylim((-0.20, 0.))
        ax[1].set_ylabel("depth (m)", fontsize=fontsize)
        ax[1].set_xlabel(f"mol of {metabolite}.s-1", fontsize=fontsize)
        ax[1].set_title(f"(left=inputs) Metabolite fluxes for {metabolite} balance (right=outputs)", fontsize=fontsize)
        ax[1].legend(loc="lower right", fontsize=fontsize)
        fig.suptitle(f't = {time}', fontsize=16)

    animation = FuncAnimation(fig, update, frames=times_to_animate, repeat=False)
    FFwriter = FFMpegWriter(fps=fps, codec="mpeg4", bitrate=5000)
    animation.save(os.path.join(output_path, f"{prop}_and_{metabolite}_{t_start}_to_{t_stop}_vertical_bins_animation.mp4"), writer=FFwriter, dpi=100)

    print("         [INFO] Finished")


def pipeline_compare_z_bins_animations(dataset, scenarios, output_path, prop, metabolic_flow="import_Nm", t_start=400, t_stop=450, fps=15, bin_z_width=0.01, mean_and_std=True, step=1, stride=1, x_max_down=1, x_max_up=1, special_case=False, screenshot=False, log_scale=False):
    print(f"    [INFO] Starting vertical bins animations for {metabolic_flow} balance to explain {prop}")
    fig, ax = plt.subplots(2, 1, figsize=(9, 16))
    
    
    filter_prop = None
    propmin = None
    propmax = None

    # if special_case:
    #     filter_prop = "distance_from_tip"
    #     propmax = 0.01

    # scenarios_names_translator ={"Drew_1975_low":"Uniform 0.01 mM", "Drew_1975_1":"0.01 mM + 1 mM patch from 8 to 12 cm"}
    # colors = {"Uniform 0.01 mM": "silver", "0.01 mM + 1 mM patch from 8 to 12 cm":"limegreen"}
    
    scenarios_names_translator ={"Drew_1975_low":"Uniform 0.01 mM", "no_root_hairs":"hairless"}
    colors = {"Uniform 0.01 mM": "silver", "hairless":"limegreen"}
    per_scenario_data = {scenarios_names_translator[scenario]: filter_dataset(dataset, scenario=scenario, only_keep=[prop, metabolic_flow, "z2", "Cumulative_Nitrogen_Uptake", "Cumulative_Carbon_Costs", "distance_from_tip", "struct_mass", "Rhizodeposits_CN_Ratio"], 
                        tmin=t_start, tmax=t_stop, prop=filter_prop, propmin=propmin, propmax=propmax) for scenario in scenarios}

    times_to_animate = [int(t) for t in np.arange(t_start, t_stop, 1.) if (t in dataset.t.values and t%step==0)]
    
    z_min = dataset["z2"].max()
    depth_bins = np.arange(0, z_min, 0.02)
    bins_center = (depth_bins[:-1] + depth_bins[1:]) / 2

    def update(time):
        print(f"        {time - t_start+1} / {t_stop - t_start}", end='\r', flush=True)
        [a.clear() for a in ax]
        if stride == 1:
            grouped_ds = {name: ds.isel(t=time).groupby_bins("z2", depth_bins) for name, ds in per_scenario_data.items()}
        else:
            grouped_ds = {name: ds.isel(t=slice(time - int(stride/2), time + int(stride/2))).groupby_bins("z2", depth_bins) for name, ds in per_scenario_data.items()}

        plot_compare_xarray_vertical_bins(fig, ax[1], grouped_ds=grouped_ds, bins_center=bins_center, prop=prop, bin_z_width=bin_z_width, colors=colors, mean_and_std=mean_and_std, special_case=special_case)
        plot_compare_xarray_vertical_bins(fig, ax[0], grouped_ds=grouped_ds, bins_center=bins_center, prop=metabolic_flow, bin_z_width=bin_z_width, colors=colors, mean_and_std=mean_and_std)
        
        fontsize = 15

        ax[1].set_xlim((0, x_max_down))
        # ax[1].set_xscale('log')
        ax[1].set_ylim((-z_min, 0.))
        ax[1].set_ylabel("depth (m)", fontsize=fontsize+5)
        ax[1].set_xlabel(f"{prop} (m2)", fontsize=fontsize+5)
        #ax[1].set_title(f"Comparisions between homogeneous and patchy concentrations", fontsize=fontsize)
        ax[1].legend(loc="lower right", fontsize=fontsize)

        ax[0].set_xlim((0, x_max_up))
        if log_scale:
            ax[0].set_xscale('log')
        ax[0].set_ylim((-z_min, 0.))
        ax[0].set_ylabel("depth (m)", fontsize=fontsize+5)
        ax[0].set_xlabel(f"{metabolic_flow} (mol.s-1)", fontsize=fontsize+5)
        #ax[0].set_title(f"Comparisions of {metabolic_flow} between homogeneous and patchy concentrations", fontsize=fontsize)
        ax[0].legend(loc="lower right", fontsize=fontsize)
        fig.suptitle(f'day = {int(time/24)}', fontsize=fontsize + 10)

    if not screenshot:
        animation = FuncAnimation(fig, update, frames=times_to_animate, repeat=False)
        FFwriter = FFMpegWriter(fps=fps, codec="mpeg4", bitrate=5000)
        animation.save(os.path.join(output_path, f"{prop}_and_{metabolic_flow}_{t_start}_to_{t_stop}_vertical_bins_animation.mp4"), writer=FFwriter, dpi=100)

    else:
        update(t_start)
        fig.savefig(os.path.join(output_path, f"Cumulated_Rhizodeposits_CN_Ratio_at_{t_start}.png"))

    print("         [INFO] Finished")


def pipeline_compare_to_experimental_data(dataset, output_path):
    fig, ax = plt.subplots(3, 1, figsize=(9, 16))

    def thermal_time_shift(d):
        times = d.t
        # Relationship derived from Fischer 1966 data
        time_shift =  (1.28e-6*(times**2) -5.30e-4*times + 8.05) / 17
        return (times * time_shift).values

    # Compare total biomasses
    plot_csv_stackable(fig, ax[0], csv_dirpath="inputs/postprocessing", csv_name="Drew_biomass_si.csv", 
                       property="total_root_biomass_control", std_prop="total_root_biomass_control_std")
    control_dataset = filter_dataset(dataset, scenario="Drew_1975_low", only_keep=["struct_mass", "import_Nm", "z2"])
    ax[0].plot(thermal_time_shift(control_dataset), control_dataset["struct_mass"].sum(dim="vid").values[0], label="Simulated total root biomass control")
    plot_csv_stackable(fig, ax[1], csv_dirpath="inputs/postprocessing", csv_name="Drew_biomass_si.csv", 
                       property="patch_root_biomass_control", std_prop="patch_root_biomass_control_std")
    control_dataset_patch_zone = filter_dataset(control_dataset, prop="z2", propmin=0.08, propmax=0.12)
    ax[1].plot(thermal_time_shift(control_dataset_patch_zone), control_dataset_patch_zone["struct_mass"].sum(dim="vid").values[0], label="Simulated 8-14 cm root biomass control")
    
    

    # Compare patch biomasses
    plot_csv_stackable(fig, ax[0], csv_dirpath="inputs/postprocessing", csv_name="Drew_biomass_si.csv", 
                       property="total_root_biomass_patch", std_prop="total_root_biomass_patch_std")
    test_dataset = filter_dataset(dataset, scenario="Drew_1975_1", only_keep=["struct_mass", "import_Nm", "z2"])
    ax[0].plot(thermal_time_shift(test_dataset), test_dataset["struct_mass"].sum(dim="vid").values[0], label="Simulated total root biomass with patch")
    plot_csv_stackable(fig, ax[1], csv_dirpath="inputs/postprocessing", csv_name="Drew_biomass_si.csv", 
                       property="patch_root_biomass_patch", std_prop="patch_root_biomass_patch_std")
    test_dataset_patch_zone = filter_dataset(test_dataset, prop="z2", propmin=0.08, propmax=0.12)
    ax[1].plot(thermal_time_shift(test_dataset_patch_zone), test_dataset_patch_zone["struct_mass"].sum(dim="vid").values[0], label="Simulated 8-14 cm root biomass with patch")
    
    ax[0].legend(fontsize=15)
    ax[0].set_title("Total dry mass comparisons", fontsize=20)
    #ax[0].set_xlim(0, 700)
    #ax[0].set_ylim(0, 0.3)
    #ax[0].set_xlabel("t (h)", fontsize=15)
    ax[0].set_ylabel("structural mass (g)", fontsize=15)
    ax[1].legend(fontsize=15)
    ax[1].set_title("8-12 cm dry mass comparisons", fontsize=20)
    #ax[1].set_xlim(0, 700)
    #ax[1].set_ylim(0, 0.08)
    ax[1].set_xlabel("t (h)", fontsize=15)
    ax[1].set_ylabel("structural mass (g)", fontsize=15)

    # Compare nitrogen uptake rates
    plot_csv_stackable(fig, ax[2], csv_dirpath="inputs/postprocessing", csv_name="Drew_biomass_si.csv", 
                       property="patch_zone_nitrate_uptake_control")
    ax[2].plot(thermal_time_shift(control_dataset_patch_zone), control_dataset_patch_zone["import_Nm"].sum(dim="vid").values[0], label="Simulated nitrate uptake in patch zone, control")
    plot_csv_stackable(fig, ax[2], csv_dirpath="inputs/postprocessing", csv_name="Drew_biomass_si.csv", 
                       property="patch_zone_nitrate_uptake_patch")
    ax[2].plot(thermal_time_shift(test_dataset_patch_zone), test_dataset_patch_zone["import_Nm"].sum(dim="vid").values[0], label="Simulated nitrate uptake in fertilized patch zone")
    
    ax[2].legend()
    ax[2].set_title("Nitrate uptake comparisions")

    fig.savefig(os.path.join(output_path, "biomasses comparision.png"))
    plt.close()

def apex_zone_contribution_final(dataset, scenarios, outputs_dirpath, flow="import_Nm", grouped_geometry="length", final_time: int = 48, mean_and_std=True, x_proportion=True):

    final_dataset = filter_dataset(dataset, time=final_time-1)
    
    # Dataframe storing computations
    df = pd.DataFrame()

    # First individual analyses
    for scenario in scenarios:
        find_day = scenario.find("_D")
        replicate = scenario[find_day-2:find_day]
        age = float(scenario[find_day+2:])
        print(f"Processing replicate {replicate} at age {age} days...")

        raw_dirpath = os.path.join(outputs_dirpath, scenario, "MTG_properties/MTG_properties_raw/")

        if len(scenarios) > 1:
            scenario_dataset = filter_dataset(final_dataset, scenario=scenario)
        else:
            scenario_dataset = final_dataset
        
        if x_proportion:
            scenario_dataset = scenario_dataset.sum(dim="default")
            total_length = scenario_dataset[grouped_geometry].sum(dim="vid")
            total_flow = abs(scenario_dataset[flow]).sum(dim="vid")
            sorted_dataset = scenario_dataset.sortby("distance_from_tip")

            # Compute the cumulative sum over the sorted 'length' variable along the 'vid' dimension
            cumulative_length_proportion = (sorted_dataset[grouped_geometry].cumsum(dim="vid") / total_length).to_numpy()
            cumulative_flow_proportion = (sorted_dataset[flow].cumsum(dim="vid") / total_flow).to_numpy()

            df = pd.concat([df, pd.DataFrame({"apex_zone_length":cumulative_length_proportion, 
                                                "apex_zone_contribution":cumulative_flow_proportion,
                                                "age":[age for _ in range(len(cumulative_length_proportion))],
                                                "replicate":[replicate for _ in range(len(cumulative_length_proportion))]
                                                }
                                            )
                            ], ignore_index=True)
            
        else:
            for azl in np.linspace(0, 0.17, num=100):
                stabilized_value, length_proportion = apex_zone_contribution(d=scenario_dataset, apex_zone_length=azl, flow=flow, plotting=False)
                df = pd.concat([df, pd.DataFrame({"apex_zone_length":[float(length_proportion[0])], 
                                                    "apex_zone_contribution":[float(stabilized_value[0])],
                                                    "age":[age],
                                                    "replicate":[replicate]
                                                    }
                                                )
                                ], ignore_index=True)

    fig, ax = plt.subplots()

    # Unique categories for color and marker
    color_categories = df['age'].unique()
    line_categories = df['replicate'].unique()

    import matplotlib.cm as cm
    
    # Choose a colormap and normalize it based on the number of categories
    colormap = cm.get_cmap('viridis', len(color_categories))  # You can choose any other colormap like 'plasma', 'coolwarm'
    colors = {category: colormap(i) for i, category in enumerate(color_categories)}

    # Assign markers to each marker category
    available_line_styles = ['-', '--', '-.', ':']
    line_style_map = {category: available_line_styles[i] for i, category in enumerate(line_categories)}

    # Plot each combination of color and marker, but avoid adding redundant legends
    
    if mean_and_std:
        
        for color_category in color_categories:
            # Variable bin number in case a root system is too small for bin meaning
            # TODO : almost working
            # bin_number = len(df[(df["age"] == color_category) & (df["apex_zone_length"] > 0.)]) / len(df[df["age"] == color_category]["replicate"].unique())
            # bin_edges = np.linspace(0, 1, num=int(bin_number / 2))
            bin_edges = np.linspace(0, 1, num=50)

            df['bin'] = pd.cut(df['apex_zone_length'], bins=bin_edges, labels=False)
            bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Calculate midpoints

            stat_df = df.groupby(['bin', 'age'])['apex_zone_contribution'].agg(['mean', 'std']).reset_index()
            stat_df['bin_center'] = stat_df['bin'].map(dict(enumerate(bin_centers)))  # Map bin centers

            subset = stat_df[stat_df['age'] == color_category]

            ax.plot(subset['bin_center'], subset['mean'], 
                            color=colors[color_category], 
                            linestyle="-")
            ax.fill_between(subset['bin_center'], subset['mean'] - subset["std"], subset['mean'] + subset["std"], color=colors[color_category], alpha=0.2,)

    else:
        for color_category in color_categories:
            for line_category in line_categories:
                subset = df[(df['replicate'] == line_category) & (df['age'] == color_category)]
                ax.plot(subset['apex_zone_length'], subset['apex_zone_contribution'], 
                            color=colors[color_category], 
                            linestyle=line_style_map[line_category])


    ax.plot([], [], c="black", linestyle='', label="Plant ages (day)")

    # Create the color legend separately
    for color_category in color_categories:
        ax.plot([], [], c=colors[color_category], linestyle='-', label=color_category)

    if not mean_and_std:
        ax.plot([], [], c="black", linestyle='', label="Replicates")

        # Create the marker legend separately
        for line_category in line_categories:
            ax.plot([], [], c='black', linestyle=line_style_map[line_category], label=line_category)

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if x_proportion:
        ax.set_xlabel(f'grouped {grouped_geometry} proportion from root apices')
    else:
        ax.set_xlabel('groupind distance from apex (m)')
    ax.set_ylabel(f'{flow} exchange proportion')

    if x_proportion:
        suffix = f"{grouped_geometry}_proportion"
    else:
        suffix = "grouping_distance_from_apex"

    if mean_and_std:
        filename = f"mean_{flow}_apex_contribution=f({suffix}).png"
    else:
        filename = f"raw_{flow}_apex_contribution=f({suffix}).png"

    fig.savefig(os.path.join(outputs_dirpath, filename), dpi=720, bbox_inches="tight")

    plt.close()


def root_length_x_percent_contributors(dataset, scenarios, outputs_dirpath, flow="import_Nm", grouped_geometry="length", proportion_of_geometry=0.1, final_time: int = 48, mean_and_std=True, x_proportion=True, histogram_organizer="distance_from_tip", ):

    final_dataset = filter_dataset(dataset, time=final_time-1)
    
    # Dataframe storing computations
    df = pd.DataFrame()

    fig, ax = plt.subplots()

    # First individual analyses
    for scenario in scenarios:
        find_day = scenario.find("_D")
        replicate = scenario[find_day-2:find_day]
        age = float(scenario[find_day+2:])
        print(f"Processing replicate {replicate} at age {age} days...")

        if len(scenarios) > 1:
            scenario_dataset = filter_dataset(final_dataset, scenario=scenario)
        else:
            scenario_dataset = final_dataset
        
        scenario_dataset = scenario_dataset.sum(dim="default")

        # Computing totals in case proportion of total is to be computed
        total_flow = abs(scenario_dataset[flow]).sum(dim="vid")
        total_geometry = scenario_dataset[grouped_geometry].sum(dim="vid")

        scenario_dataset[f"{grouped_geometry}_proportion"] = scenario_dataset[grouped_geometry] / total_geometry
        scenario_dataset[f"{flow}_proportion"] = scenario_dataset[flow] / total_flow

        # Bins along 
        bins = np.linspace(0., scenario_dataset["distance_from_tip"].max(), 40)

        binned_data = scenario_dataset.groupby_bins(scenario_dataset["distance_from_tip"], bins).sum(dim="vid")

        binned_data["distance_from_tip"] = scenario_dataset.groupby_bins(scenario_dataset["distance_from_tip"], bins).mean(dim="vid")["distance_from_tip"]
        binned_data = binned_data.sortby(binned_data["distance_from_tip"], ascending=True)

        sorted_binned_data = binned_data.sortby(binned_data[f"{flow}_proportion"], ascending=False)

        colors = []

        last_bin_to_split = True
        tests = sorted_binned_data[f"{grouped_geometry}_proportion"].cumsum() < proportion_of_geometry
        
        for test in (sorted_binned_data[f"{grouped_geometry}_proportion"].cumsum() < proportion_of_geometry).values:
            if test:
                colors.append("red")

            else:
                if (len(colors) == 0 or colors[-1] == "red") and last_bin_to_split:
                    colors.append("red")
                    last_bin_to_split = False
                else:
                    colors.append("blue")

        sorted_binned_data["color"] = xr.DataArray(colors, dims="distance_from_tip_bins", coords={"distance_from_tip_bins": sorted_binned_data["distance_from_tip_bins"]})
        extract = sorted_binned_data.where(sorted_binned_data["color"] == "red")
        
        grouped_length_prop = extract[f"{grouped_geometry}_proportion"].sum()
        grouped_flux_prop = extract[f"{flow}_proportion"].sum()
        print(grouped_length_prop, grouped_flux_prop)

        sorted_colors = sorted_binned_data.sortby(sorted_binned_data["distance_from_tip"], ascending=True)["color"].values

        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        ax.bar(bin_centers, binned_data[f"{flow}_proportion"].values, width=bins[1] - bins[0], color=sorted_colors)

    #plt.show()

    fig.savefig(os.path.join(outputs_dirpath, f"top_{int(proportion_of_geometry*100)}%_contributors_histo.png"), dpi=720, bbox_inches="tight")

    plt.close()
    

def top_percent_contributors(dataset, scenarios, outputs_dirpath, flow="import_Nm", grouped_geometry="length", unit="m", proportion_of_geometry=0.1, final_time: int = 48, mean_and_std=True, plotting_along="distance_from_tip"):

    final_dataset = filter_dataset(dataset, time=final_time-1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    replicate = []
    age = []
    for scenario in scenarios:
        find_day = scenario.find("_D")
        replicate.append(scenario[find_day-2:find_day])
        age.append(float(scenario[find_day+2:]))

    x_unit = "m"
    flow_unit = ureg("mol") / ureg("s")
    flow_unit_norm = flow_unit / ureg(unit)
    y_unit = f"{flow_unit_norm.units:~P}"

    import matplotlib.cm as cm
    import matplotlib.markers as mrk
    
    color_categories = np.unique(age)
    style_categories = np.unique(replicate)

    # Choose a colormap and normalize it based on the number of categories
    colormap = cm.get_cmap('viridis', len(color_categories))
    markermap = list(mrk.MarkerStyle.markers.keys())
    exceptions = [".", ","]
    for marker in exceptions:
        markermap.remove(marker)

    colors = {category: colormap(i) for i, category in enumerate(color_categories)}
    style = {category: markermap[i] for i, category in enumerate(style_categories)}

    # First individual analyses
    for scenario in scenarios:
        find_day = scenario.find("_D")
        replicate = scenario[find_day-2:find_day]
        age = float(scenario[find_day+2:])
        print(f"Processing replicate {replicate} at age {age} days...")

        if len(scenarios) > 1:
            scenario_dataset = filter_dataset(final_dataset, scenario=scenario)
        else:
            scenario_dataset = final_dataset
        
        scenario_dataset = scenario_dataset.sum(dim="default")

        # Computing totals in case proportion of total is to be computed
        total_flow = abs(scenario_dataset[flow]).sum(dim="vid")
        total_geometry = scenario_dataset[grouped_geometry].sum(dim="vid")

        scenario_dataset[f"{grouped_geometry}_proportion"] = scenario_dataset[grouped_geometry] / total_geometry
        scenario_dataset[f"{flow}_proportion"] = scenario_dataset[flow] / total_flow

        scenario_dataset[f"{flow}_per_geometry"] = scenario_dataset[flow] / (scenario_dataset[grouped_geometry] *(2 * np.pi * scenario_dataset["radius"]))
        # scenario_dataset[f"{flow}_per_length"] = scenario_dataset[flow] / scenario_dataset["length"]

        cumsummed_dataset = scenario_dataset.sortby(scenario_dataset[f"{flow}_per_geometry"], ascending = False)

        cumsummed_dataset[grouped_geometry] = (cumsummed_dataset[grouped_geometry] / total_geometry).cumsum(dim="vid")

        cropped_dataset = cumsummed_dataset.where(cumsummed_dataset[grouped_geometry] <= proportion_of_geometry)
        
        group_contribution = cropped_dataset[flow].sum(dim="vid") / total_flow
        
        if mean_and_std:
            ax.errorbar(cropped_dataset[plotting_along].mean(), cropped_dataset[f"{flow}_per_geometry"].mean(),
                        xerr=cropped_dataset[plotting_along].std(), yerr=cropped_dataset[f"{flow}_per_geometry"].std(),
                        fmt=style[replicate], color=colors[age])
        else:
            ax.scatter(cropped_dataset[plotting_along].values, cropped_dataset[f"{flow}_per_geometry"].values, marker=style[replicate], color=colors[age])
        
        if mean_and_std:
            # Get the current axis limits
            x_limits = plt.gca().get_xlim()
            y_limits = plt.gca().get_ylim()

            # Calculate the offset based on the axis range (e.g., 1% of the range)
            x_offset = (x_limits[1] - x_limits[0]) * 0.01
            y_offset = (y_limits[1] - y_limits[0]) * 0.01
            ax.text(cropped_dataset[plotting_along].mean() + x_offset, cropped_dataset[f"{flow}_per_geometry"].mean() + y_offset,
                            f"10% = {round(float(group_contribution)*100, 1)}% {flow}".replace("_", " "), fontsize=7)

    
    ax.plot([], [], c="black", linestyle='', label="Plant ages (day)")

    # Create the color legend separately
    for color_category in color_categories:
        ax.scatter([], [], c=colors[color_category], marker="s", label=color_category)

    ax.plot([], [], c="black", linestyle='', label="Replicates")

    # Create the marker legend separately
    for line_category in style_categories:
        ax.scatter([], [], c='black', marker=style[line_category], label=line_category)

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    ax.set_xlabel(f'{plotting_along} ({x_unit})'.replace("_", " "))
    ax.set_ylabel(f'{flow} flow per root {grouped_geometry} ({y_unit})'.replace("_", " "))
    fig.suptitle(f"Position of the top {int(100*proportion_of_geometry)}% most active root {grouped_geometry} according to {plotting_along}".replace("_", " "))
    
    #plt.show()

    if mean_and_std:
        filename = f"mean_{flow}_for_{int(100*proportion_of_geometry)}%_{grouped_geometry}=f({plotting_along}).png"
    else:
        filename = f"raw_{flow}_for_{int(100*proportion_of_geometry)}%_{grouped_geometry}=f({plotting_along}).png"

    fig.savefig(os.path.join(outputs_dirpath, filename), dpi=720, bbox_inches="tight")

    plt.close()


def log_mtg_coordinates(g):
    def root_visitor(g, v, turtle, gravitropism_coefficient=0.06):
        n = g.node(v)

        # For displaying the radius or length X times larger than in reality, we can define a zoom factor:
        zoom_factor = 1.
        # We look at the geometrical properties already defined within the root element:
        radius = n.radius * zoom_factor
        length = n.length * zoom_factor
        angle_down = n.angle_down
        angle_roll = n.angle_roll

        # We get the x,y,z coordinates from the beginning of the root segment, before the turtle moves:
        position1 = turtle.getPosition()
        n.x1 = position1[0] / zoom_factor
        n.y1 = position1[1] / zoom_factor
        n.z1 = position1[2] / zoom_factor

        # The direction of the turtle is changed:
        turtle.down(angle_down)
        turtle.rollL(angle_roll)

        # Tropism is then taken into account:
        # diameter = 2 * n.radius * zoom_factor
        # elong = n.length * zoom_factor
        # alpha = tropism_intensity * diameter * elong
        # turtle.rollToVert(alpha, tropism_direction)
        # if g.edge_type(v)=='+':
        # diameter = 2 * n.radius * zoom_factor
        # elong = n.length * zoom_factor
        # alpha = tropism_intensity * diameter * elong
        turtle.elasticity = gravitropism_coefficient * (n.original_radius / g.node(1).original_radius)
        turtle.tropism = (0, 0, -1)

        # The turtle is moved:
        turtle.setId(v)
        if n.type != "Root_nodule":
            # We define the radius of the cylinder to be displayed:
            turtle.setWidth(radius)
            # We move the turtle by the length of the root segment:
            turtle.F(length)
        else:  # SPECIAL CASE FOR NODULES
            # We define the radius of the sphere to be displayed:
            turtle.setWidth(radius)
            # We "move" the turtle, but not according to the length (?):
            turtle.F()

        # We get the x,y,z coordinates from the end of the root segment, after the turtle has moved:
        position2 = turtle.getPosition()
        n.x2 = position2[0] / zoom_factor
        n.y2 = position2[1] / zoom_factor
        n.z2 = position2[2] / zoom_factor

    # We initialize a turtle in PlantGL:

    turtle = turt.PglTurtle()
    # We make the graph upside down:
    turtle.down(180)
    # We initialize the scene with the MTG g:
    turt.TurtleFrame(g, visitor=root_visitor, turtle=turtle, gc=False)

def post_color_mtg(mtg_file_path, output_dirpath, property, recording_off_screen=False, flow_property=False, background_color="brown", 
                   imposed_min=None, imposed_max=None, log_scale=False, spinning=False, root_hairs=True):
    from log.visualize import plot_mtg_alt
    with open(mtg_file_path, "rb") as f:
        g = pickle.load(f)

    print(len(g.properties()["struct_mass"]))
    log_mtg_coordinates(g)
    props = g.properties()
    
    sizes = {"landscape": [1920, 1080], "portrait": [1088, 1920], "square": [1080, 1080],
                "small_height": [960, 1280]}

    if recording_off_screen:
        pv.start_xvfb()

    plotter = pv.Plotter(off_screen=recording_off_screen, window_size=sizes["portrait"], lighting="three lights")
    plotter.set_background(background_color)
    step_back_coefficient = 0.2
    camera_coordinates = (step_back_coefficient, 0., 0.)
    move_up_coefficient = 0.01
    horizontal_aiming = (0., 0., 1.)
    collar_position = (0., 0., -move_up_coefficient)
    plotter.camera_position = [camera_coordinates,
                                    collar_position,
                                    horizontal_aiming]

    plotter.show(interactive_update=True)

    # Then add initial states of plotted compartments
    if not root_hairs:
        root_system_mesh, color_property = plot_mtg_alt(g, cmap_property=property, flow_property=flow_property)
    else:
        root_system_mesh, color_property, root_hairs_system = plot_mtg_alt(g, cmap_property=property, flow_property=flow_property, root_hairs=root_hairs)


    if 0. in color_property:
                color_property.remove(0.)
    if imposed_min:
        clim_min = imposed_min
    else:
        clim_min = min(color_property)

    if imposed_max:
        clim_max = imposed_max
    else:
        clim_max = max(color_property)

    plotter.add_mesh(root_system_mesh, scalars=property+".m-1", cmap="jet", clim=[clim_min, clim_max], show_edges=False, log_scale=log_scale)
    #plotter.add_text(f"MTG displaying {property} at day", position="upper_left")
    if root_hairs:
        plotter.add_mesh(root_hairs_system, scalars="living_root_hairs_struct_mass", opacity=0.05, cmap="gist_gray", show_edges=False)

    if spinning:
        plotter.open_movie(os.path.join(output_dirpath, f"{property}_spinning_view.mp4"))
        n_frames = 360  # One rotation
        spinning_speed = 0.9
        zoom_factor = 1.002
        down_factor = 1.0065
        for i in range(n_frames):

            plotter.camera_position = [
                np.array(plotter.camera_position[0]) * zoom_factor,
                np.array(plotter.camera_position[1]) * down_factor,
                horizontal_aiming]
            
            plotter.camera.azimuth += 1 * spinning_speed
            plotter.update()
            plotter.write_frame()

    input("Save current view?")

    plotter.screenshot(os.path.join(output_dirpath, f'{property}_plot_snapshot.png'))

def add_root_order_when_branching_is_wrong(g):
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)
    g.properties().setdefault("root_order", {})
    root_order = g.property("root_order")
    radius = g.property("radius")
    # We travel in the MTG from the root tips to the base:
    known_vids = []
    for vid in post_order2(g, root):
        if vid not in known_vids:
            axis = g.Axis(vid)
            axis_mean_radius = np.mean([radius[v] for v in axis])
            if axis_mean_radius > 1e-4 and len(axis) > 10:
                order = 1
            else:
                order = 2
            for v in axis:
                root_order[v] = order

            known_vids += axis

def recolorize_glb(time_step, dataset, property, glb_dirpath, colormap):
    from pygltflib import GLTF2, BufferFormat, GLBResource
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    # Load GLTF model and extract mesh using trimesh
    scene = trimesh.load(glb_dirpath)

    # Convert coordinates to numpy array for easier lookup
    logged_coords = dataset[['x1', 'y1', 'z1']].values
    logged_property = dataset['property'].values  # Replace with actual column.

    # Extract vertex positions from the GLTF model
    vertices = np.array(scene.vertices)  # Assuming trimesh loads mesh correctly

    # Find closest matches between logged data and GLTF vertices

    tree = cKDTree(logged_coords)
    _, indices = tree.query(vertices)

    # Assign colors based on the new property
    min_val, max_val = logged_property.min(), logged_property.max()
    normalized_values = (logged_property[indices] - min_val) / (max_val - min_val)

    # Convert property values to RGB (example: using a colormap)

    cmap = plt.get_cmap(colormap)
    colors = cmap(normalized_values)[:, :3]  # Get RGB values

    # Ensure colors are in [0, 255] range
    colors = (colors * 255).astype(np.uint8)

    for mesh in scene.geometry.values():
        mesh.visual.vertex_colors = np.hstack([colors, np.full((colors.shape[0], 1), 255)])  # RGBA

    # Save the updated GLTF model
    scene.export(f"{property}.glb")


def inspect_mtg_structure(g):
    print(g.components_at_scale(1, 1))
    axis_scale = [g.get_vertex_property(vid) for vid in g.components_at_scale(1, 2)]
    axis_vids = [vid for vid in g.components_at_scale(1, 2)]
    print(dict(zip(axis_vids, [e["label"] for e in axis_scale])))
    chosenid = int(input("which axis?"))
    print(g.get_vertex_property(chosenid))
    metamer_scale = [g.get_vertex_property(vid) for vid in g.components_at_scale(chosenid, 3)]
    metamer_vids = [vid for vid in g.components_at_scale(chosenid, 3)]
    print(dict(zip(metamer_vids, [e["label"] for e in metamer_scale])))
    chosenid = int(input("which metamer?"))
    print(g.get_vertex_property(chosenid))
    organ_scale = [g.get_vertex_property(vid) for vid in g.components_at_scale(chosenid, 4)]
    organ_vids = [vid for vid in g.components_at_scale(chosenid, 4)]
    print(dict(zip(organ_vids, [e["label"] for e in organ_scale])))
    chosenid = int(input("which organ?"))
    print(g.get_vertex_property(chosenid))
    elt_scale = [g.get_vertex_property(vid) for vid in g.components_at_scale(chosenid, 5)]
    elt_vids = [vid for vid in g.components_at_scale(chosenid, 5)]
    print(dict(zip(elt_vids, [e["label"] for e in elt_scale])))
    chosenid = int(input("which element?"))
    print(g.get_vertex_property(chosenid))
    input()


class RootCyNAPSFigures:

    def Fig_1_c(scenario_dataset, outputs_path):
        final_time = max(scenario_dataset.t.values)
        final_dataset = scenario_dataset.sel(t=final_time)

        XarrayPlotting.scatter_xarray(final_dataset, outputs_dirpath=outputs_path, x="distance_from_tip", y="Lengthy Net mineral N uptake", c="root_order", discrete=True, s=1, name_suffix=f"_{final_time}")
        XarrayPlotting.scatter_xarray(final_dataset, outputs_dirpath=outputs_path, x="distance_from_tip", y="Massic mineral N uptake", c="root_order", discrete=True, s=1, name_suffix=f"_{final_time}")

    def Fig_1_d(scenario_dataset, outputs_dirpath):
        final_time = max(scenario_dataset.t.values)
        final_dataset = scenario_dataset.sel(t=final_time)

        total_struct_mass = final_dataset["struct_mass"].sum()

        Nm_export_to_shoot_per_struct_mass = final_dataset["Net mineral N uptake"].sum() / total_struct_mass
        simple_uptake_per_struct_mass = final_dataset["simple_import_Nm"].sum() / total_struct_mass

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

        labels = ["Root-CyNAPS", "Simple whole roots system Michaelis-Menten"]
        ax.bar(range(len(labels)), 
               [Nm_export_to_shoot_per_struct_mass, simple_uptake_per_struct_mass], color=["green", "purple"], log=True)

        ax.set_ylabel("Simulated inorganic N uptake (mol.g-1.s-1)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim([1e-9, 1e-7])

        filename = f"Bar_N_uptake_{final_time}.png"

        fig.savefig(os.path.join(outputs_dirpath, filename), dpi=720, bbox_inches="tight")

        return fig, ax

    def Fig_2(final_dataset, outputs_dirpath, distance_bins, flow, normalization_property):
        
        total_normalization_property = final_dataset[normalization_property].sum()
        total_flow = final_dataset[flow].sum()

        first_dataset = final_dataset.where(final_dataset.root_order == 1)
        if np.any((final_dataset.root_order > 1).values):
            lateral_dataset = final_dataset.where(final_dataset.root_order > 1)
        else:
            lateral_dataset = None
        # binned_dataset = final_dataset.groupby_bins("distance_from_tip", distance_bins).sum()
        first_binned_dataset = first_dataset.groupby_bins("distance_from_tip", distance_bins).sum()
        if lateral_dataset:
            lateral_binned_dataset = lateral_dataset.groupby_bins("distance_from_tip", distance_bins).sum()

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create labels for each bin
        bin_labels = [0]
        for bin in first_binned_dataset.distance_from_tip_bins.values:
            bin_labels.append(bin.right)
        
        # X locations for the bars
        x = np.arange(len(bin_labels))
        width = 0.35  # Width of each bar

        # Plot side-by-side proportion of uptake and struct mass for each root type
        ax.bar(x[:-1] + 0.5 - width/2, (100 * first_binned_dataset[flow] / total_flow).values, width, label=f'seminal + nodal {flow.replace("_", " ")}', color="green")
        if lateral_dataset:
            ax.bar(x[:-1] + 0.5 - width/2, (100 * lateral_binned_dataset[flow] / total_flow).values, width, bottom=(100 * first_binned_dataset[flow] / total_flow).values ,label=f'lateral {flow.replace("_", " ")}', color="limegreen")

        ax.bar(x[:-1] + 0.5 + width/2, (100 * first_binned_dataset[normalization_property] / total_normalization_property).values, width, label='seminal + nodal {normalization_property.replace("_", " ")}', color="chocolate")
        if lateral_dataset:
            ax.bar(x[:-1] + 0.5 + width/2, (100 * lateral_binned_dataset[normalization_property] / total_normalization_property).values, width, bottom=(100 * first_binned_dataset[normalization_property] / total_normalization_property).values , label=f'lateral {normalization_property.replace("_", " ")}', color="darkorange")

        # Add labels and title
        ax.set_xlabel("Distance from tip groups (m)")
        ax.set_ylabel("% of root system total")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.legend(loc="upper right")
        ax.set_xlim([-0.5, 30])
        ax.set_ylim([0, 30])
        # ax.grid(True, linestyle='--', alpha=0.6)

        ymin, ymax = ax.get_ylim()

        distance_at_threshold, norm_prop_at_threshold = RootCyNAPSFigures.Fig_2_polarisation(final_dataset, flow="Net mineral N uptake", normalization_property=normalization_property)

        line_x = 30 * distance_at_threshold / 0.2
        ax.plot([line_x, line_x], [ymin, ymax], color="black", linestyle='dashed')

        ax.text(line_x + 1, 0.9 * ymax, f"50% of flux\n={norm_prop_at_threshold:.2%} of {normalization_property.replace('_', ' ')}")

        filename = f"Bin_contribution_{flow.replace('_', ' ')}_{final_dataset.t.values}.png"

        fig.savefig(os.path.join(outputs_dirpath, filename), dpi=720, bbox_inches="tight")

        plt.close()

        return distance_at_threshold, float(total_normalization_property.values)

        

    def Fig_2_polarisation(final_dataset, flow, normalization_property):

        total_normalization_property = final_dataset[normalization_property].sum(dim="vid")
        total_flow = abs(final_dataset[flow]).sum(dim="vid")
        sorted_dataset = final_dataset.sortby("distance_from_tip")

        # Compute the cumulative sum over the sorted 'length' variable along the 'vid' dimension
        cumulative_normalization_property_proportion = (sorted_dataset[normalization_property].cumsum(dim="vid") / total_normalization_property)
        cumulative_flow_proportion = (sorted_dataset[flow].cumsum(dim="vid") / total_flow)

        # Find threshold index where cumulative import exceeds 50%
        threshold_idx = (cumulative_flow_proportion >= 0.5).argmax().item()

        # Get distance and struct mass proportion at that index
        distance_at_threshold = sorted_dataset["distance_from_tip"][threshold_idx]
        normalization_property_at_threshold = cumulative_normalization_property_proportion[threshold_idx]

        return float(distance_at_threshold.values), float(normalization_property_at_threshold.values)
    
    
    def Fig_2_lines(final_dataset, outputs_dirpath, distance_bins, flow, normalization_property):
        
        total_normalization_property = final_dataset[normalization_property].sum()
        total_flow = final_dataset[flow].sum()

        first_dataset = final_dataset.where(final_dataset.root_order == 1)
        if np.any((final_dataset.root_order > 1).values):
            lateral_dataset = final_dataset.where(final_dataset.root_order > 1)
        else:
            lateral_dataset = None

        binned_dataset = final_dataset.groupby_bins("distance_from_tip", distance_bins).sum()
        first_binned_dataset = first_dataset.groupby_bins("distance_from_tip", distance_bins).sum()
        if lateral_dataset:
            lateral_binned_dataset = lateral_dataset.groupby_bins("distance_from_tip", distance_bins).sum()

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create labels for each bin
        bin_labels = [0]
        for bin in first_binned_dataset.distance_from_tip_bins.values:
            bin_labels.append(bin.right)
        
        # X locations for the bars
        x = np.arange(len(bin_labels))
        width = 0.35  # Width of each bar

        # Plot side-by-side proportion of uptake and struct mass for each root type
        
        if lateral_dataset:
            ax.plot(x[:-1] + 0.5 - width/2, (100 * first_binned_dataset[flow] / total_flow).values, width, label=f'seminal + nodal {flow.replace("_", " ")}', color="green")
            ax.plot(x[:-1] + 0.5 - width/2, (100 * lateral_binned_dataset[flow] / total_flow).values, width, label=f'lateral {flow.replace("_", " ")}', color="limegreen")
        ax.plot(x[:-1] + 0.5 - width/2, (100 * binned_dataset[flow] / total_flow).values, width, label=f'total bin {flow.replace("_", " ")}', color="darkgreen")
        
        if lateral_dataset:
            ax.plot(x[:-1] + 0.5 + width/2, (100 * first_binned_dataset[normalization_property] / total_normalization_property).values, width, label=f'seminal + nodal {normalization_property.replace("_", " ")}', color="chocolate")
            ax.plot(x[:-1] + 0.5 + width/2, (100 * lateral_binned_dataset[normalization_property] / total_normalization_property).values, width, label=f'lateral {normalization_property.replace("_", " ")}', color="darkorange")
        ax.plot(x[:-1] + 0.5 + width/2, (100 * binned_dataset[normalization_property] / total_normalization_property).values, width, label=f'total bin {normalization_property.replace("_", " ")}', color="black")


        # Add labels and title
        ax.set_xlabel("Distance from tip groups (m)")
        ax.set_ylabel("% of root system total")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.legend(loc="upper right")
        ax.set_xlim([-0.5, 30])
        ax.set_ylim([0, 30])
        # ax.grid(True, linestyle='--', alpha=0.6)

        ymin, ymax = ax.get_ylim()

        distance_at_threshold, norm_prop_at_threshold = RootCyNAPSFigures.Fig_2_polarisation(final_dataset, flow="Net mineral N uptake", normalization_property=normalization_property)

        line_x = 30 * distance_at_threshold / 0.2
        ax.plot([line_x, line_x], [ymin, ymax], color="black", linestyle='dashed')

        ax.text(line_x + 1, 0.9 * ymax, f"50% of flux\n={norm_prop_at_threshold:.2%} of {normalization_property.replace('_', ' ')}")

        filename = f"Bin_contribution_{flow.replace('_', ' ')}_{final_dataset.t.values}.png"

        fig.savefig(os.path.join(outputs_dirpath, filename), dpi=720, bbox_inches="tight")

        plt.close()

        return distance_at_threshold, float(total_normalization_property.values)


class Indicators:

    def Nitrogen_Aquisition_Efficiency(d):
        """
        Ratio between Net nitrogen acquisition by roots (mol N.s-1) and C consumption for growth and respiration.
        """
        nitrogen_net_aquisition = d.import_Nm - d.diffusion_Nm_soil - d.diffusion_Nm_soil_xylem
        carbon_structural_mass_costs = (d.hexose_consumption_by_growth * 6) + (d.amino_acids_consumption_by_growth * 5) + d.maintenance_respiration + d.N_metabolic_respiration
        return nitrogen_net_aquisition / carbon_structural_mass_costs.where(carbon_structural_mass_costs > 0.)

    def Cumulative_Nitrogen_Aquisition_Efficiency(d):
        """
        Ratio between Net nitrogen acquisition by roots (mol N.s-1) and C consumption for growth and respiration.
        """
        nitrogen_net_aquisition = (d.import_Nm - d.diffusion_Nm_soil - d.diffusion_Nm_soil_xylem).cumsum(dim="t")
        carbon_structural_mass_costs = ((d.hexose_consumption_by_growth * 6) + (d.amino_acids_consumption_by_growth * 5) + d.maintenance_respiration + d.N_metabolic_respiration).cumsum(dim="t")
        return nitrogen_net_aquisition / carbon_structural_mass_costs.where(carbon_structural_mass_costs > 0.)

    def Cumulative_Carbon_Costs(d):
        return ((d.hexose_consumption_by_growth * 6) + (d.amino_acids_consumption_by_growth * 5) + d.maintenance_respiration + d.N_metabolic_respiration).cumsum(dim="t")
    
    def Cumulative_Nitrogen_Uptake(d):
        return (d.import_Nm - d.diffusion_Nm_soil - d.diffusion_Nm_soil_xylem).cumsum(dim="t")
    

    def Hexose_Root_Soil_gradient(d):
        return (d.C_hexose_root * d.struct_mass / d.symplasmic_volume) - d.C_hexose_soil
    
    def Amino_Acids_Root_Soil_gradient(d):
        return (d.AA * d.struct_mass / d.symplasmic_volume) - d.C_amino_acids_soil
    
    def Nm_Root_Soil_gradient(d):
        return (d.Nm * d.struct_mass / d.symplasmic_volume) - d.C_mineralN_soil
    
    def Gross_Hexose_Exudation(d):
        """
        Net hexose exudation root wise, doesn't account for soil respiration so corresponds to gross rhizodeposition for experimental data
        """
        return d.hexose_exudation + d.phloem_hexose_exudation + d.cells_release + d.mucilage_secretion - d.hexose_uptake_from_soil - d.phloem_hexose_uptake_from_soil
    
    def Gross_AA_Exudation(d):
        """
        Net amino acid exudation root wise, doesn't account for soil respiration so corresponds to gross rhizodeposition for experimental data
        """
        return d.diffusion_AA_soil + d.diffusion_AA_soil_xylem - d.import_AA
    
    def Gross_C_Rhizodeposition(d):
        return d.Gross_Hexose_Exudation * 6 + d.Gross_AA_Exudation * 5

    def Rhizodeposits_CN_Ratio(d):
        return (d.Gross_Hexose_Exudation * 6 + d.Gross_AA_Exudation * 5) / (d.Gross_AA_Exudation.where(d.Gross_AA_Exudation > 0.) *1.4)
    
    def CN_Ratio_Cumulated_Rhizodeposition(d):
        gross_C = d.hexose_exudation + d.phloem_hexose_exudation + d.cells_release + d.mucilage_secretion - d.hexose_uptake_from_soil - d.phloem_hexose_uptake_from_soil
        gross_N = d.diffusion_AA_soil + d.diffusion_AA_soil_xylem - d.import_AA
        cum_gross_C = gross_C.cumsum(dim="t")
        cum_gross_N = gross_N.cumsum(dim="t")
        return (cum_gross_C * 6 + cum_gross_N * 5) / (cum_gross_N.where(cum_gross_N > 0.) * 1.4)
    
    def Root_Hairs_Surface(d):
        return ((6e-6 * 2 * np.pi) * d.root_hair_length) * d.total_root_hairs_number
    
    def Root_Hairs_Proportion(d):
        return d.Root_Hairs_Surface / d.root_exchange_surface.where(d.root_exchange_surface > 0.)
    
    def Labile_Nitrogen(d):
        "Mol of Nitrogen per gram of dry mass for all labile nitrogen forms (nitrate, amonium, amino acids, ...)"
        return d.Nm + d.AA * 1.4
    
    def cylinder_surface(d):
        return d.length * (2 * np.pi * d.radius)

    def compute(d, formula):
        variables = set(re.findall(r'\b[a-zA-Z_]\w*\b', formula))

        units = {var: ureg(d[var].unit.replace("-", "^-")) for var in variables}

        result = eval(formula, {}, d)

        result_unit = eval(formula, {}, units)

        result.attrs.update({"unit": f"{result_unit.units:~P}"})
        
        return result
    
class XarrayPlotting:

    def scatter_xarray(dataset, outputs_dirpath, x: str, y: str, c: str=None, discrete: bool=False, s: int=None, name_suffix: str=""):
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

        x_unit = ureg(dataset[x].unit.replace("-", "^-"))
        y_unit = ureg(dataset[y].unit.replace("-", "^-"))
        if c:
            c_unit = ureg(dataset[c].unit.replace("-", "^-"))

        if not isinstance(dataset, list):
            dataset = [dataset]

        unique_values = np.unique(d[c].values)
        for i, d in enumerate(dataset):
            if discrete:
                plotted = ax.scatter(d[x].values, d[y].values, c=list(color_palette.values())[i], s=s, label=unique_values[i])
            else:
                plotted = ax.scatter(d[x].values, d[y].values, c=d[c].values, s=s)
        
        ax.set_xlabel(f"{x.replace('_', ' ')} ({x_unit.units:~P})")
        ax.set_ylabel(f"{y.replace('_', ' ')} ({y_unit.units:~P})")

        if c:
            shown_name = f"{c_unit.units:~P}"
            if shown_name == "":
                shown_name = "adim"

            if discrete:
                ax.legend()
            else:
                fig.colorbar(plotted, ax=ax, label=f"{c.replace('_', ' ')} ({shown_name})")

        filename = f"Scatter_{y}_vs_{x}_colored_by_{c}{name_suffix}.png"

        fig.savefig(os.path.join(outputs_dirpath, filename), dpi=720, bbox_inches="tight")

        plt.close()