import pandas as pd
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from .experiment import FlowExperiment


# ─────────────────────────────────────────────────────────────────────────────
#  _single_run
#
#  Worker function for a single training run. Called by each process in the
#  parallel version. Takes all arguments as a single tuple because mp.Pool
#  only supports single-argument worker functions.
# ─────────────────────────────────────────────────────────────────────────────

def _single_run(args):
    run_id, config, interp, dataset_base, dataset_target, train_kwargs, fid_steps = args

    exp = FlowExperiment(config, interp, stepping=train_kwargs['stepping'])
    exp.train(
        dataset_base   = dataset_base,
        dataset_target = dataset_target,
        n_iterations   = train_kwargs['n_iterations'],
        batch_size     = train_kwargs['batch_size'],
        base_lr        = train_kwargs['base_lr'],
        weight_decay   = train_kwargs['weight_decay'],
        rand_run       = train_kwargs['rand_run'],
        log_every      = train_kwargs['n_iterations'] + 1,
        out_name       = f"{train_kwargs['out_name']}_{run_id}",
    )

    fid = exp.fid(dataset_target, steps=fid_steps)
    print(f"Run {run_id} complete  |  FID: {fid:.4f}")
    return fid


# ─────────────────────────────────────────────────────────────────────────────
#  run_experiment_boxplot
#
#  Trains the same model configuration n_runs times sequentially from scratch
#  and collects a FID score for each run. Returns a list of FID values which
#  can be passed directly to plot_fid_boxplot().
#
#  Use this when you want simple sequential runs without worrying about
#  parallelism or VRAM management.
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_boxplot(
    config,
    interp,
    dataset_target,
    dataset_base   = None,
    n_runs         = 10,
    n_iterations   = 5000,
    batch_size     = 256,
    base_lr        = 1e-3,
    weight_decay   = 1e-7,
    rand_run       = True,
    stepping       = 'ode',
    fid_steps      = 500,
    out_name       = 'run',
) -> list:
    """
    Train the same model configuration n_runs times from scratch and collect
    FID scores for each run.

    Parameters
    ----------
    config         : FlowModelConfig
    interp         : Interpolant
    dataset_target : target distribution dataset
    dataset_base   : base distribution dataset. Ignored if rand_run=True,
                     in which case Gaussian noise is used as the base.
    n_runs         : how many times to train from scratch
    n_iterations   : gradient steps per run
    batch_size     : batch size per step
    base_lr        : learning rate
    weight_decay   : AdamW weight decay
    rand_run       : if True, base distribution is fresh Gaussian noise each step
    stepping       : 'ode' or 'sde'
    fid_steps      : number of Euler steps when computing FID
    out_name       : base name for saved checkpoints
    """
    if dataset_base is None:
        dataset_base = dataset_target

    fid_scores = []

    for run_id in range(n_runs):
        print(f"\n── Run {run_id + 1}/{n_runs} ──")

        exp = FlowExperiment(config, interp, stepping=stepping)
        exp.train(
            dataset_base   = dataset_base,
            dataset_target = dataset_target,
            n_iterations   = n_iterations,
            batch_size     = batch_size,
            base_lr        = base_lr,
            weight_decay   = weight_decay,
            rand_run       = rand_run,
            log_every      = n_iterations + 1,
            out_name       = f'{out_name}_{run_id}',
        )

        fid = exp.fid(dataset_target, steps=fid_steps)
        fid_scores.append(fid)
        print(f"Run {run_id + 1}/{n_runs} complete  |  FID: {fid:.4f}")

    return fid_scores


# ─────────────────────────────────────────────────────────────────────────────
#  run_experiment_boxplot_parallel
#
#  Same as run_experiment_boxplot but runs n_workers training jobs at the
#  same time using torch.multiprocessing. Only makes sense if you have enough
#  VRAM to hold n_workers models simultaneously.
#
#  To choose n_workers:
#    check VRAM per run with nvidia-smi during a single run, then:
#    n_workers = floor(total_vram / vram_per_run)
#    leave some headroom -- don't fill VRAM completely
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_boxplot_parallel(
    config,
    interp,
    dataset_target,
    dataset_base   = None,
    n_runs         = 10,
    n_iterations   = 5000,
    batch_size     = 256,
    base_lr        = 1e-3,
    weight_decay   = 1e-7,
    rand_run       = True,
    stepping       = 'ode',
    fid_steps      = 500,
    out_name       = 'run',
    n_workers      = 4,
) -> list:
    """
    Train the same model configuration n_runs times in parallel and collect
    FID scores for each run.

    Parameters
    ----------
    n_workers : how many runs to execute simultaneously.
                rule of thumb: floor(total_vram / vram_per_run)
    all other parameters are the same as run_experiment_boxplot.
    """
    if dataset_base is None:
        dataset_base = dataset_target

    train_kwargs = dict(
        n_iterations = n_iterations,
        batch_size   = batch_size,
        base_lr      = base_lr,
        weight_decay = weight_decay,
        rand_run     = rand_run,
        stepping     = stepping,
        out_name     = out_name,
    )

    args = [
        (run_id, config, interp, dataset_base, dataset_target, train_kwargs, fid_steps)
        for run_id in range(n_runs)
    ]

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=n_workers) as pool:
        fid_scores = pool.map(_single_run, args)

    return fid_scores


# ─────────────────────────────────────────────────────────────────────────────
#  plot_fid_boxplot
#
#  Plots a boxplot of FID scores from run_experiment_boxplot() or
#  run_experiment_boxplot_parallel() and prints summary statistics.
# ─────────────────────────────────────────────────────────────────────────────

def plot_fid_boxplot(fid_scores: list, title: str = "FID distribution"):
    """
    Plot a boxplot of FID scores and print summary statistics.

    Parameters
    ----------
    fid_scores : list of FID values, one per run
    title      : plot title and x-axis label
    """
    s = pd.Series(fid_scores)

    plt.figure(figsize=(6, 5))
    plt.boxplot(fid_scores, tick_labels=[title])
    plt.ylabel("FID")
    plt.title(title)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    print(f"mean:  {s.mean():.4f}")
    print(f"std:   {s.std():.4f}")
    print(f"min:   {s.min():.4f}")
    print(f"max:   {s.max():.4f}")