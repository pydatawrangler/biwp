import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 7: Bayesian Additive Regression Trees
        """
    )
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import arviz as az
    import pymc as pm
    import pymc_bart as pmb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from cycler import cycler
    return az, cycler, np, pd, plt, pm, pmb


@app.cell
def _(az, np, plt):
    az.style.use('arviz-grayscale')
    plt.rcParams["figure.dpi"] = 300
    np.random.seed(5453)
    viridish = [(0.2823529411764706, 0.11372549019607843, 0.43529411764705883, 1.0),
                (0.1843137254901961, 0.4196078431372549, 0.5568627450980392, 1.0),
                (0.1450980392156863, 0.6705882352941176, 0.5098039215686274, 1.0),
                (0.6901960784313725, 0.8666666666666667, 0.1843137254901961, 1.0)]
    return (viridish,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## BART Bikes
        """
    )
    return


@app.cell
def _(np, pd):
    data = pd.read_csv("../data/bikes_hour.csv")
    data = data[::50]
    data.sort_values(by='hour', inplace=True)
    data.hour.values.astype(float)

    X = np.atleast_2d(data["hour"]).T
    Y = data["count"]
    return X, Y, data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 7.1
        """
    )
    return


@app.cell
def _(X, Y, pm, pmb):
    with pm.Model() as bart_g:
        _σ = pm.HalfNormal('σ', Y.std())
        μ = pmb.BART('μ', X, Y, m=50)
        _y = pm.Normal('y', μ, _σ, observed=Y)
        idata_bart_g = pm.sample(2000, chains=1)
    return idata_bart_g, μ


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 7.4
        """
    )
    return


@app.cell
def _(X, Y, cycler, idata_bart_g, plt, viridish, μ):
    _, _ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _ax[0].set_prop_cycle(cycler('color', viridish))
    _ax[1].set_prop_cycle(cycler('color', viridish))
    for i in range(3):
        _ax[0].plot(X, μ.owner.op.all_trees[i * 10][0][i * 2].predict(X)[0], 'o-', lw=1)
    posterior = idata_bart_g.posterior.stack(samples=('chain', 'draw'))
    for i in range(3):
        _ax[1].plot(X, posterior['μ'].sel(draw=i * 50), 'o-', lw=1)
    _ax[1].plot(X, Y, 'C2.', zorder=-1)
    _ax[0].set_ylabel('count')
    _ax[0].set_title('Individual trees')
    _ax[1].set_title('Sum of trees')
    _ax[1].set_xlabel('hour')
    _ax[1].set_ylabel('count')
    plt.savefig('img/chp07/BART_bikes_samples.png')
    return (posterior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 7.5
        """
    )
    return


@app.cell
def _(X, Y, az, plt, posterior):
    _, _ax = plt.subplots(1, 1, figsize=(12, 4))
    _ax.plot(X, Y, 'o', alpha=0.3, zorder=-1)
    az.plot_hdi(X[:, 0], posterior['μ'].T, smooth=True)
    az.plot_hdi(X[:, 0], posterior['μ'].T, hdi_prob=0.1, color='C4', smooth=True)
    _ax.set_xlabel('hour')
    _ax.set_ylabel('count')
    plt.savefig('img/chp07/BART_bikes.png')
    return


@app.cell
def _(np, pd, plt):
    space_in = pd.read_csv('../data/space_influenza.csv')
    X_1 = np.atleast_2d(space_in['age']).T
    Y_1 = space_in['sick']
    Y_jittered = np.random.normal(Y_1, 0.02)
    plt.plot(X_1[:, 0], Y_jittered, '.', alpha=0.5)
    return X_1, Y_1, Y_jittered


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Generalized BART Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 7.2 and Figures 7.6 and 7.7
        """
    )
    return


@app.cell
def _(X_1, Y_1, pm, pmb):
    idatas = {}
    ntrees = [2, 10, 20, 50]
    for ntree in ntrees:
        with pm.Model() as bart_b:
            μ_1 = pmb.BART('μ', X_1, Y_1, m=ntree)
            p = pm.Deterministic('p', pm.math.sigmoid(μ_1))
            _y = pm.Bernoulli('y', p, observed=Y_1)
            idata_bart_b = pm.sample(2000, idata_kwargs={'log_likelihood': True})
            idatas[f'{ntree}'] = idata_bart_b
    return (idatas,)


@app.cell
def _(az, idatas):
    cmp = az.compare(idatas)
    cmp
    return (cmp,)


@app.cell
def _(az, cmp, plt):
    az.plot_compare(cmp, figsize=(10, 2.5))
    plt.savefig("img/chp07/BART_space_flu_comp.png")
    return


@app.cell
def _(X_1, Y_jittered, az, idatas, np, plt):
    fig, _axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True, sharex=True)
    for _ax, (mtree, idata) in zip(np.ravel(_axes), idatas.items()):
        μs = idata.posterior['p'].stack({'draws': ['chain', 'draw']})
        _ax.plot(X_1, Y_jittered, 'C1.', alpha=0.5)
        X_idx = np.argsort(X_1[:, 0])
        _ax.plot(X_1[:, 0][X_idx], np.mean(μs, 1)[X_idx], 'k-')
        az.plot_hdi(X_1[:, 0], μs.T, ax=_ax, smooth=False, color='0.5')
        _ax.set_title(mtree)
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(['healthy', 'sick'])
    fig.text(0.55, -0.04, 'Age', ha='center', size=14)
    fig.text(-0.03, 0.5, 'Space Influenza', va='center', size=14, rotation=90)
    plt.savefig('img/chp07/BART_space_flu_fit.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interpretability of BARTs
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 7.8
        """
    )
    return


@app.cell
def _(np, plt):
    X_2 = np.random.normal(0, 1, size=(3, 250)).T
    Y_2 = np.random.normal(0, 1, size=250)
    X_2[:, 0] = np.random.normal(Y_2, 0.1)
    X_2[:, 1] = np.random.normal(Y_2, 0.2)
    plt.plot(X_2, Y_2, '.')
    return X_2, Y_2


@app.cell
def _(X_2, Y_2, pm, pmb):
    with pm.Model() as _model:
        μ_2 = pmb.BART('μ', X_2, Y_2, m=50)
        _σ = pm.HalfNormal('σ', 1)
        _y = pm.Normal('y', μ_2, _σ, observed=Y_2)
        _trace_u = pm.sample(2000, tune=1000, random_seed=42)
    return (μ_2,)


@app.cell
def _(X_2, plt, pmb, μ_2):
    pmb.plot_pdp(μ_2, X_2, grid='long')
    plt.savefig('img/chp07/partial_dependence_plot.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figures 7.9, 7.10, and 7.11 
        """
    )
    return


@app.cell
def _(data):
    X_3 = data[['hour', 'temperature', 'humidity', 'windspeed']]
    Y_3 = data['count']
    return X_3, Y_3


@app.cell
def _(X_3, Y_3, pm, pmb):
    with pm.Model() as bart_model_g:
        _σ = pm.HalfNormal('σ', Y_3.std())
        μ_3 = pmb.BART('μ', X_3, Y_3, m=50)
        _y = pm.Normal('y', μ_3, _σ, observed=Y_3)
        trace_bart = pm.sample(2000)
    return (μ_3,)


@app.cell
def _(X_3, plt, pmb, μ_3):
    pmb.plot_pdp(μ_3, X_3, grid=(2, 2), figsize=(12, 6), sharey=True)
    plt.savefig('img/chp07/partial_dependence_plot_bikes.png', bbox_inches='tight')
    return


@app.cell
def _(X_3, plt, pmb, μ_3):
    pmb.plot_ice(μ_3, X_3, grid=(2, 2), smooth=True, color_mean='C4')
    plt.savefig('img/chp07/individual_conditional_expectation_plot_bikes.png', bbox_inches='tight')
    return


@app.cell
def _(np, plt):
    X_4 = np.random.uniform(-1, 1, (250, 3))
    Z = np.where(X_4[:, 2] >= 0, np.zeros_like(X_4[:, 2]), np.ones_like(X_4[:, 2]))
    e = np.random.normal(0, 0.1, 250)
    Y_4 = 0.2 * X_4[:, 0] - 5 * X_4[:, 1] + 10 * X_4[:, 1] * Z + e
    plt.plot(X_4[:, 1], Y_4, '.')
    return X_4, Y_4


@app.cell
def _(X_4, Y_4, pm, pmb):
    with pm.Model() as _model:
        μ_4 = pmb.BART('μ', X_4, Y_4, m=200)
        _σ = pm.HalfNormal('σ', 1)
        _y = pm.Normal('y', μ_4, _σ, observed=Y_4)
        _trace_u = pm.sample()
    return (μ_4,)


@app.cell
def _(X_4, Y_4, plt, pmb, μ_4):
    _, _ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
    _ax[0].plot(X_4[:, 1], Y_4, '.')
    _ax[0].set_xlabel('X_1')
    _ax[0].set_ylabel('Observed Y')
    pmb.plot_pdp(μ_4, X_4, smooth=True, color='C4', var_idx=[1], ax=_ax[1])
    pmb.plot_ice(μ_4, X_4, centered=False, smooth=False, color_mean='C4', var_idx=[1], ax=_ax[2])
    plt.savefig('img/chp07/pdp_vs_ice_toy.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Variable Selection

        The new version of BART uses a different algorithm to compute the variable importance. With this method the variable importance is robust to the number of trees and results are easier to interpret. Instead of reporting the value of the variable importance we report the value of R² (the square of the pearson correlation coefficient) between the model with all variables, versus model with a subset of the variables. To see how this works in practice we are going to fit the 3 same models as in the book, but with the new version of BART.
        """
    )
    return


@app.cell
def _(np):
    Xs = []
    Ys = []
    Y_5 = np.random.normal(0, 1, 100)
    X_5 = np.random.normal(0, 1, (100, 10))
    X_5[:, 0] = np.random.normal(Y_5, 0.1)
    X_5[:, 1] = np.random.normal(Y_5, 0.2)
    Xs.append(X_5)
    Ys.append(Y_5)
    X_5 = np.random.uniform(0, 1, size=(100, 10))
    fx = 10 * np.sin(np.pi * X_5[:, 0] * X_5[:, 1]) + 20 * (X_5[:, 2] - 0.5) ** 2 + 10 * X_5[:, 3] + 5 * X_5[:, 4]
    Y_5 = np.random.normal(fx, 1)
    Xs.append(X_5)
    Ys.append(Y_5)
    Y_5 = np.random.normal(0, 1, 100)
    X_5 = np.random.normal(0, 1, (100, 10))
    Xs.append(X_5)
    Ys.append(Y_5)
    return Xs, Ys


@app.cell
def _(Xs, Ys, pm, pmb):
    idatas_1 = []
    bart_rvs = []
    for X_6, Y_6 in zip(Xs, Ys):
        with pm.Model() as _bart:
            _σ = pm.HalfNormal('σ', Y_6.std())
            μ_5 = pmb.BART('μ', X_6, Y_6)
            _y = pm.Normal('y', μ_5, _σ, observed=Y_6)
            bart_rvs.append(μ_5)
            idatas_1.append(pm.sample())
    return bart_rvs, idatas_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 7.12
        """
    )
    return


@app.cell
def _(Xs, bart_rvs, idatas_1, plt, pmb):
    _, _axes = plt.subplots(3, 1, figsize=(10, 6))
    for j, X_7 in enumerate(Xs):
        pmb.utils.plot_variable_importance(idatas_1[j], bart_rvs[j], X_7, figsize=(10, 5), ax=_axes[j])
    plt.savefig('img/chp07/bart_vi_toy.png', bbox_inches='tight')
    return


@app.cell
def _(pd):
    data_1 = pd.read_csv('../data/bikes_hour.csv')
    data_1.sort_values(by='hour', inplace=True)
    data_1 = data_1[::50]
    X_8 = data_1[['hour', 'temperature', 'humidity', 'windspeed']]
    Y_7 = data_1['count']
    return X_8, Y_7


@app.cell
def _(X_8, Y_7, pm, pmb):
    with pm.Model() as _bart:
        _σ = pm.HalfNormal('σ', Y_7.std())
        μ_6 = pmb.BART('μ', X_8, Y_7)
        _y = pm.Normal('y', μ_6, _σ, observed=Y_7)
        idata_1 = pm.sample()
    return idata_1, μ_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 7.13
        """
    )
    return


@app.cell
def _(X_8, idata_1, pmb, μ_6):
    pmb.utils.plot_variable_importance(idata_1, μ_6, X_8, figsize=(10, 2))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
