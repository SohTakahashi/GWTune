#%%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import optuna
import glob
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
import pandas as pd
import seaborn as sns

# %%
main_compute = True
main_visualize = True

# GWOT parameters
eps_list = [1e-2, 1e-0]
num_trial = 100

# optuna.logging.set_verbosity(optuna.logging.WARNING)

# Parameters
n_points = 20  # Total number of points
rot_deg_list = np.arange(1, 7) # Rotation degrees
sampler_initilizations = ["random_tpe", "random_grid", "uniform_grid"]

#%%
# Parameter for toolbox
vis_config = VisualizationConfig(
    show_figure=False,
    figsize=(8, 6), 
    title_size = 10, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial",
    color_label_width=3,
    xlabel=f"{n_points} items",
    ylabel=f"{n_points} items",
    xlabel_size=10,
    ylabel_size=10,
)

vis_config_ot = VisualizationConfig(
    show_figure=False,
    figsize=(8, 6), 
    title_size = 10, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial",
    color_label_width=3,
    xlabel=f"{n_points} items of X",
    ylabel=f"{n_points} items of Y",
    xlabel_size=10,
    ylabel_size=10,
)

vis_log = VisualizationConfig(
    show_figure=False,
    figsize=(8, 6), 
    title_size = 10, 
    cmap = "viridis",
    cbar_ticks_size=15,
    xlabel_size=20,
    xticks_size=15,
    ylabel_size=20,
    yticks_size=15,
    cbar_label_size=15,
    plot_eps_log=True,
    fig_ext='svg'
)

#%%
def add_independent_noise_to_all_dimensions(points, noise_deg=0.0001):
    """
    Adds independent Gaussian noise to all dimensions of all points.
    """
    noise = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape)
    return points + noise

#%%
def add_noise_to_one_dimension(points, noise_deg=0.0001, dimension=0):
    """
    Adds Gaussian noise to only one dimension of each point in the point cloud.
    This is a sanity check to observe the effect of noise on a single dimension.
    """
    noise = np.zeros_like(points)
    noise[:, dimension] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[0])
    return points + noise

#%%
def create_circle_data(n_points):
    t = np.linspace(0, 2 * np.pi, n_points)
    x = np.cos(t)
    y = np.sin(t)
    one_data = np.vstack((x, y)).T
    return one_data

rot_data = lambda data, theta: data @ np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta),  np.cos(theta)],
])

# %%
def run_starfish_experiment(
    shape1, 
    shape2, 
    sampler_init,
    rot_deg:int,
):
    # define the main results directory and the representation names
    initialization, sampler = sampler_init.split("_")
    main_results_dir = f"../results/circle/{sampler_init}"
    
    # Create representations
    rep1 = Representation(name=f"shape1", metric="euclidean", embedding=shape1)
    rep2 = Representation(name=f"shape2_rot_π-{rot_deg:.2f}", metric="euclidean", embedding=shape2)

    config = OptimizationConfig(
        eps_list=eps_list,
        num_trial=num_trial,
        db_params={"drivername": "sqlite"},
        sinkhorn_method="sinkhorn",
        n_iter=1,
        max_iter = 200,
        to_types="numpy",  
        device="cpu",
        data_type="double", 
        sampler_name=sampler,
        init_mat_plan=initialization,
        show_progress_bar=False,
    )

    alignment = AlignRepresentations(
        config=config,
        representations_list=[rep1, rep2],
        main_results_dir=main_results_dir,
        data_name=f"circle_{n_points}_points",
    )

    # RSA
    fig_dir = f"../results/figs/circle/"
    os.makedirs(fig_dir, exist_ok=True)
    alignment.show_sim_mat(
        visualization_config=vis_config, 
        show_distribution=False,
        fig_dir=f"{fig_dir}/RDM"
    )
    alignment.RSA_get_corr(metric="pearson")

    # GW
    data_path = f"{main_results_dir}/{alignment.data_name}_{rep1.name}_vs_{rep2.name}/*/data/*.npy"
    npy_list = glob.glob(data_path)
    
    if len(npy_list) >= num_trial:
        print(f"{alignment.data_name} was already computed.")
        
    else:
        alignment.gw_alignment(
            compute_OT=True,
            delete_results=False,
            visualization_config=vis_config_ot,
            delete_confirmation=False
        )
        
        alignment.show_OT(fig_dir=f"{fig_dir}/{sampler}_{initialization}", visualization_config=vis_config_ot)
        alignment.show_optimization_log(fig_dir=f"{fig_dir}/{sampler}_{initialization}", visualization_config=vis_log)

#%%
def main_test(n_points, rot_deg:int):
    shape1 = create_circle_data(n_points)
    shape2 = rot_data(shape1, np.pi / rot_deg)
    
    # Visualize the shapes
    fig = plt.figure(figsize=(10, 4))

    rot_str = f"$\pi$/{rot_deg:.2f}"
    # Shape 1
    ax1 = fig.add_subplot(121)
    ax1.axis("equal")
    ax1.scatter(shape1[:, 0], shape1[:, 1], c="C0", label='Shape 1')
    ax1.set_title("Shape 1 (Original Circle)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid()
    ax1.legend(loc="upper right")

    # Shape 2
    ax2 = fig.add_subplot(122)
    ax2.axis("equal")
    ax2.scatter(shape2[:, 0], shape2[:, 1], c="C1", label=f'Shape 2 (rot π/{rot_deg:.2f})')
    ax2.set_title("Shape 2 (Rotated Circle)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid()
    ax2.legend(loc="upper right")

    plt.tight_layout()
    
    raw_save_fig_dir = f"../results/figs/circle/raw"
    os.makedirs(raw_save_fig_dir, exist_ok=True)
    plt.savefig(f"{raw_save_fig_dir}/{n_points}_points_rot_π-{rot_deg:.2f}.png") # 例えば、π/2とすると、/がパス判定になり、ファイルが作成されないため -で置換。
    plt.close()
    
    pool = ProcessPoolExecutor(len(sampler_initilizations))
    
    with pool:
        processes = []
        for _, sampler_init in enumerate(sampler_initilizations):                    
            future = pool.submit(
                run_starfish_experiment,
                shape1=shape1,
                shape2=shape2,
                sampler_init=sampler_init,
                rot_deg=rot_deg,
                # noise_deg=noise_deg,
            )

            processes.append(future)

        for future in as_completed(processes):
            future.result()


# %%
def get_result_from_database(n_points, rot_deg, main_results_dir):
    shape1_name = f"shape1"
    shape2_name = f"shape2_rot_π-{rot_deg:.2f}"
    
    data_name = f"circle_{n_points}_points_{shape1_name}_vs_{shape2_name}"
    
    df_path = glob.glob(f"{main_results_dir}/{data_name}/*/*.db")[0]
    
    study = optuna.load_study(study_name = os.path.basename(df_path).split(".db")[0], storage = f"sqlite:///{df_path}")
    
    return study

#%%
def get_ot(n_points, rot_deg, main_results_dir, min_index):
    shape1_name = f"shape1"
    shape2_name = f"shape2_rot_π-{rot_deg:.2f}"
    
    data_name = f"circle_{n_points}_points_{shape1_name}_vs_{shape2_name}"
    
    npy_path = glob.glob(f"{main_results_dir}/{data_name}/*/data/gw_{min_index}.npy")[0]
    
    ot = np.load(npy_path)
    
    return ot
    

# %%
if main_compute:
    n_jobs = 3
    main_pool = ProcessPoolExecutor(n_jobs)
    
    with main_pool:
        processes = [main_pool.submit(main_test, n_points, rot_deg) for rot_deg in rot_deg_list] 

        for future in tqdm(as_completed(processes), total=len(rot_deg_list), desc="Progress", leave=True):
            future.result()
                    
                    
#%%
# plot the results
if main_visualize:
    os.makedirs("../results/figs/circle/main_fig", exist_ok=True)
    for rot_deg in rot_deg_list:
        plt.figure(figsize=(10, 10))
        
        min_values = []
        for sampler_init in sampler_initilizations:
            main_results_dir = f"../results/circle/{sampler_init}"
            study = get_result_from_database(n_points, rot_deg, main_results_dir)
            df = study.trials_dataframe()
            
            plt.subplot(3, 1, sampler_initilizations.index(sampler_init) + 1)
            plt.scatter(df["params_eps"], df["value"], c = 100 * df["user_attrs_best_acc"], s=12)

            plt.xlabel("eps")
            plt.ylabel("GWD")
            plt.title(f"{sampler_init} (π/{int(rot_deg)})")
            plt.colorbar()
            plt.grid(True)
            
            min_value = df.index[df["value"] == df["value"].min()]
            min_values.append(min_value[0])
        
        plt.tight_layout()
        
        plt.savefig(f"../results/figs/circle/main_fig/comparison_log_π-{int(rot_deg)}.png")
        plt.close()
        
        plt.figure(figsize=(10,4))
        for i, sampler_init in enumerate(sampler_initilizations):
            main_results_dir = f"../results/circle/{sampler_init}"
            ot = get_ot(n_points, rot_deg, main_results_dir, min_values[i])
            
            plt.subplot(1, 3, i + 1)
            plt.imshow(ot, cmap="rocket_r")

            plt.xlabel(f"{n_points} points")
            plt.ylabel(f"{n_points} points")
            plt.title(f"OT {sampler_init} (π/{int(rot_deg)})")
            plt.colorbar(shrink=0.6)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"../results/figs/circle/main_fig/comparison_ot_π-{int(rot_deg)}.png")
        plt.close()
# %%
