import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import preliz as pz
    return az, np, plt, pm, pz


@app.cell
def _(az, plt):
    az.style.use("arviz-grayscale")
    from cycler import cycler
    default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('figure', dpi=300)
    return


@app.cell
def _(np, pz):
    np.random.seed(123)
    trials = 4
    theta_real = 0.35 # unknown value in a real experiment
    data = pz.Binomial(n=1, p=theta_real).rvs(trials)
    data
    return (data,)


@app.cell
def _(data, pm):
    with pm.Model() as our_first_model:
        theta = pm.Beta('theta', alpha=1., beta=1.)
        y = pm.Bernoulli('y', p=theta, observed=data)
        idata = pm.sample(1000, random_seed=4591)
    return (idata,)


@app.cell
def _(az, idata):
    az.plot_trace(idata)
    return


@app.cell
def _(az, idata):
    az.plot_trace(idata, kind='rank_bars', combined=True, rank_kwargs={"colors": "k"})
    return


@app.cell
def _(az, idata):
    az.summary(idata, kind="stats").round(2)
    return


@app.cell
def _(az, idata):
    az.plot_posterior(idata, figsize=(12, 4))
    return


@app.cell
def _(az, idata, np):
    az.plot_bf(idata, var_name="theta", prior=np.random.uniform(0, 1, 10000), ref_val=0.5, figsize=(12, 4), colors=["C0", "C2"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
