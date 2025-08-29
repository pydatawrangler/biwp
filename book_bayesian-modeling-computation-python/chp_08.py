import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 8: Approximate Bayesian Computation
        """
    )
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    from scipy import stats

    from scripts.rf_selector import select_model
    return az, np, pd, plt, pm, select_model, stats


@app.cell
def _(az, np, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams['figure.dpi'] = 300
    np.random.seed(1346)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Fitting a Gaussian the ABC-way
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 8.2
        """
    )
    return


@app.cell
def _(np, plt, stats):
    _a = stats.norm(-2.5, 0.5)
    _b = stats.norm(2.5, 1)
    c = stats.norm(0, 3)
    x = np.linspace(-6, 6, 500)
    lpdf = 0.65 * _a.pdf(x) + 0.35 * _b.pdf(x)
    ppdf = c.pdf(x)
    _, _ax = plt.subplots(figsize=(10, 4))
    for c, β in zip(['#A8A8A8', '#585858', '#000000', '#2a2eec'], [0, 0.2, 0.5, 1]):
        post = ppdf * lpdf ** β
        post = post / post.sum()
        _ax.plot(x, post, lw=3, label=f'β={β}', color=c)
    _ax.set_yticks([])
    _ax.set_xticks([])
    _ax.set_xlabel('θ')
    _ax.legend()
    plt.savefig('img/chp08/smc_tempering.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Fitting a Gaussian the ABC-way
        """
    )
    return


@app.cell
def _(np):
    data = np.random.normal(loc=0, scale=1, size=1000)

    def normal_sim(rng, a, b, size=1000):
        return rng.normal(_a, _b, size=size)
    return data, normal_sim


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.2 and Figure 8.3
        """
    )
    return


@app.cell
def _(data, normal_sim, pm):
    with pm.Model() as gauss:
        _μ = pm.Normal('μ', mu=0, sigma=1)
        _σ = pm.HalfNormal('σ', sigma=1)
        _s = pm.Simulator('s', normal_sim, params=[_μ, _σ], distance='gaussian', sum_stat='sort', epsilon=1, observed=data)
        trace_g = pm.sample_smc()
    return (trace_g,)


@app.cell
def _(az, trace_g):
    az.summary(trace_g)
    return


@app.cell
def _(az, plt, trace_g):
    az.plot_trace(trace_g, kind="rank_bars", figsize=(10, 4));
    plt.savefig('img/chp08/trace_g.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Choosing the Distance Function, $\epsilon$ and the Summary Statistics
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Codes 8.4, 8.5, 8.6, 8.7, and 8.8
        """
    )
    return


@app.cell
def _(data, normal_sim, pm):
    with pm.Model() as gauss_001:
        _μ = pm.Normal('μ', mu=0, sigma=1)
        _σ = pm.HalfNormal('σ', sigma=1)
        _s = pm.Simulator('s', normal_sim, params=[_μ, _σ], sum_stat='sort', epsilon=0.1, observed=data)
        idata_g_001 = pm.sample_smc()
        idata_g_001.extend(pm.sample_posterior_predictive(idata_g_001))
    with pm.Model() as gauss_01:
        _μ = pm.Normal('μ', mu=0, sigma=1)
        _σ = pm.HalfNormal('σ', sigma=1)
        _s = pm.Simulator('s', normal_sim, params=[_μ, _σ], sum_stat='sort', epsilon=1, observed=data)
        idata_g_01 = pm.sample_smc()
        idata_g_01.extend(pm.sample_posterior_predictive(idata_g_01))
    with pm.Model() as gauss_02:
        _μ = pm.Normal('μ', mu=0, sigma=1)
        _σ = pm.HalfNormal('σ', sigma=1)
        _s = pm.Simulator('s', normal_sim, params=[_μ, _σ], sum_stat='sort', epsilon=2, observed=data)
        idata_g_02 = pm.sample_smc()
        idata_g_02.extend(pm.sample_posterior_predictive(idata_g_02))
    with pm.Model() as gauss_05:
        _μ = pm.Normal('μ', mu=0, sigma=1)
        _σ = pm.HalfNormal('σ', sigma=1)
        _s = pm.Simulator('s', normal_sim, params=[_μ, _σ], sum_stat='sort', epsilon=5, observed=data)
        idata_g_05 = pm.sample_smc()
        idata_g_05.extend(pm.sample_posterior_predictive(idata_g_05))
    with pm.Model() as gauss_10:
        _μ = pm.Normal('μ', mu=0, sigma=1)
        _σ = pm.HalfNormal('σ', sigma=1)
        _s = pm.Simulator('s', normal_sim, params=[_μ, _σ], sum_stat='sort', epsilon=10, observed=data)
        idata_g_10 = pm.sample_smc()
        idata_g_10.extend(pm.sample_posterior_predictive(idata_g_10))
    with pm.Model() as gauss_NUTS:
        _μ = pm.Normal('μ', mu=0, sigma=1)
        _σ = pm.HalfNormal('σ', sigma=1)
        _s = pm.Normal('s', _μ, _σ, observed=data)
        idata_g_nuts = pm.sample()
    return idata_g_001, idata_g_01, idata_g_05, idata_g_10, idata_g_nuts


@app.cell
def _(az, idata_g_01, idata_g_05, idata_g_10, idata_g_nuts, plt):
    idatas = [idata_g_nuts, idata_g_01, idata_g_05, idata_g_10]
    az.plot_forest(idatas, model_names=["NUTS", "ϵ 1", "ϵ 5", "ϵ 10"],
                   colors=["#2a2eec", "#000000", "#585858", "#A8A8A8"],
                   figsize=(8, 3));
    plt.savefig("img/chp08/trace_g_many_eps.png")
    return


@app.cell
def _(az, idata_g_001, plt):
    az.plot_trace(idata_g_001, kind="rank_bars", figsize=(10, 4));
    plt.savefig("img/chp08/trace_g_eps_too_low.png")
    return


@app.cell
def _(az, data, idata_g_001, idata_g_01, idata_g_05, idata_g_10, np, plt):
    idatas_ = [idata_g_001, idata_g_01, idata_g_05, idata_g_10]
    epsilons = [0.1, 1, 5, 10]
    _, _axes = plt.subplots(2, 2, figsize=(10, 5))
    for _i, _ax in enumerate(_axes.ravel()):
        pp_vals = idatas_[_i].posterior_predictive['s'].values.reshape(8000, -1)
        tstat_pit = np.mean(pp_vals <= data, axis=0)
        _, tstat_pit_dens = az.kde(tstat_pit)
        _ax.axhline(1, color='w')
        az.plot_bpv(idatas_[_i], kind='u_value', ax=_ax, reference='analytical')
        _ax.tick_params(axis='both', pad=7)
        _ax.set_title(f'ϵ={epsilons[_i]}, mse={np.mean((1 - tstat_pit_dens) ** 2) * 100:.2f}')
    plt.savefig('img/chp08/bpv_g_many_eps_00.png')
    return epsilons, idatas_


@app.cell
def _(az, epsilons, idatas_, plt):
    _, _ax = plt.subplots(2, 2, figsize=(10, 5))
    _ax = _ax.ravel()
    for _i in range(4):
        az.plot_bpv(idatas_[_i], kind='p_value', reference='samples', color='C4', ax=_ax[_i], plot_ref_kwargs={'color': 'C2'})
        _ax[_i].set_title(f'ϵ={epsilons[_i]}')
    plt.savefig('img/chp08/bpv_g_many_eps_01.png')
    return


@app.cell
def _(az, epsilons, idatas_, plt):
    _, _axes = plt.subplots(2, 2, figsize=(10, 5))
    for _i, _ax in enumerate(_axes.ravel()):
        az.plot_ppc(idatas_[_i], num_pp_samples=100, ax=_ax, color='C2', mean=False, legend=False, observed=False)
        az.plot_kde(idatas_[_i].observed_data['s'].values, plot_kwargs={'color': 'C4'}, ax=_ax)
        _ax.set_xlabel('s')
        _ax.set_title(f'ϵ={epsilons[_i]}')
    plt.savefig('img/chp08/ppc_g_many_eps.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## g-and-k distributions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 8.9
        """
    )
    return


@app.cell
def _(pd):
    data_1 = pd.read_csv('../data/air_pollution_bsas.csv')
    bsas_co = data_1['co'].dropna().values
    return (bsas_co,)


@app.cell
def _(bsas_co, plt):
    _, _axes = plt.subplots(2, 1, figsize=(10, 4), sharey=True)
    _axes[0].hist(bsas_co, bins='auto', color='C1', density=True)
    _axes[0].set_yticks([])
    _axes[1].hist(bsas_co[bsas_co < 3], bins='auto', color='C1', density=True)
    _axes[1].set_yticks([])
    _axes[1].set_xlabel('CO levels (ppm)')
    plt.savefig('img/chp08/co_ppm_bsas.png')
    f'We have {sum(bsas_co > 3)} observations larger than 3 ppm'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.4 and Figure 8.10
        """
    )
    return


@app.cell
def _(f, np, optimize, stats):
    class g_and_k_quantile:

        def __init__(self):
            self.quantile_normal = stats.norm(0, 1).ppf
            self.pdf_normal = stats.norm(0, 1).pdf

        def ppf(self, x, a, b, g, k):
            z = self.quantile_normal(x)
            return _a + _b * (1 + 0.8 * np.tanh(_g * z / 2)) * (1 + z ** 2) ** _k * z

        def rvs(self, rng, a, b, g, k, size):
            x = rng.uniform(0, 1, size)
            return self.ppf(x, _a, _b, _g, _k)

        def cdf(self, x, a, b, g, k, zscale=False):
            optimize.fminbound(f, -5, 5)

        def pdf(self, x, a, b, g, k):
            z = x
            z_sq = z ** 2
            term1 = (1 + z_sq) ** _k
            term2 = 1 + 0.8 * np.tanh(_g * x / 2)
            term3 = (1 + (2 * _k + 1) * z_sq) / (1 + z_sq)
            term4 = 0.8 * _g * z / (2 * np.cosh(_g * z / 2) ** 2)
            deriv = _b * term1 * (term2 * term3 + term4)
            return self.pdf_normal(x) / deriv
    return (g_and_k_quantile,)


@app.cell
def _(az, g_and_k_quantile, np, plt):
    _gk = g_and_k_quantile()
    u = np.linspace(1e-14, 1 - 1e-14, 10000)
    params = ((0, 1, 0, 0), (0, 1, 0.4, 0), (0, 1, -0.4, 0), (0, 1, 0, 0.25))
    _, _ax = plt.subplots(2, 4, sharey='row', figsize=(10, 5))
    for _i, p in enumerate(params):
        _a, _b, _g, _k = p
        ppf = _gk.ppf(u, _a, _b, _g, _k)
        _ax[0, _i].plot(u, ppf)
        _ax[0, _i].set_title(f'a={_a}, b={_b},\ng={_g}, k={_k}')
        az.plot_kde(ppf, ax=_ax[1, _i], bw=0.5)
    plt.savefig('img/chp08/gk_quantile.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.5
        """
    )
    return


@app.cell
def _(np):
    def octo_summary(x):
        e1, e2, e3, e4, e5, e6, e7 = np.quantile(x, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        sa = e4
        sb = e6 - e2
        sg = (e6 + e2 - 2*e4)/sb
        sk = (e7 - e5 + e3 - e1)/sb
        return np.array([sa, sb, sg, sk])
    return (octo_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.6
        """
    )
    return


@app.cell
def _(bsas_co, g_and_k_quantile):
    _gk = g_and_k_quantile()

    def gk_simulator(rng, a, b, g, k, size=None):
        return _gk.rvs(rng, _a, _b, _g, _k, len(bsas_co))
    return (gk_simulator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.7 and Figure 8.11
        """
    )
    return


@app.cell
def _(bsas_co, gk_simulator, octo_summary, pm):
    with pm.Model() as gkm:
        _a = pm.HalfNormal('a', sigma=1)
        _b = pm.HalfNormal('b', sigma=1)
        _g = pm.HalfNormal('g', sigma=1)
        _k = pm.HalfNormal('k', sigma=1)
        _s = pm.Simulator('s', gk_simulator, params=[_a, _b, _g, _k], sum_stat=octo_summary, epsilon=0.1, observed=bsas_co)
        idata_gk = pm.sample_smc()
        idata_gk.extend(pm.sample_posterior_predictive(idata_gk))
    return (idata_gk,)


@app.cell
def _(az, idata_gk):
    az.summary(idata_gk)
    return


@app.cell
def _(az, idata_gk, plt):
    az.plot_trace(idata_gk, kind="rank_bars")
    plt.savefig("img/chp08/trace_gk.png")
    return


@app.cell
def _(az, idata_gk, plt):
    _axes = az.plot_pair(idata_gk, kind='kde', marginals=True, textsize=45, kde_kwargs={'contourf_kwargs': {'cmap': plt.cm.viridis}})
    for _ax, pad in zip(_axes[:, 0], (70, 30, 30, 30)):
        _ax.set_ylabel(_ax.get_ylabel(), rotation=0, labelpad=pad)
    plt.savefig('img/chp08/pair_gk.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Approximating moving averages
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.8 and Figure 8.12
        """
    )
    return


@app.cell
def _(np):
    def moving_average_2(θ1, θ2, n_obs=500):
        λ = np.random.normal(0, 1, n_obs + 2)
        _y = λ[2:] + _θ1 * λ[1:-1] + _θ2 * λ[:-2]
        return _y
    return (moving_average_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We are calling the simulator one more time to generate "observed data".
        """
    )
    return


@app.cell
def _(moving_average_2):
    θ1_true = 0.6
    θ2_true = 0.2
    y_obs = moving_average_2(θ1_true, θ2_true)
    return y_obs, θ1_true, θ2_true


@app.cell
def _(az, moving_average_2, plt, θ1_true, θ2_true):
    az.plot_trace({'one sample':moving_average_2(θ1_true, θ2_true),
                   'another sample':moving_average_2(θ1_true, θ2_true)},
                  trace_kwargs={'alpha':1},
                  figsize=(10, 4)
                 )
    plt.savefig("img/chp08/ma2_simulator_abc.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.9
        """
    )
    return


@app.cell
def _(np):
    def autocov(x):
        _a = np.mean(x[1:] * x[:-1])
        _b = np.mean(x[2:] * x[:-2])
        return np.array((_a, _b))

    def moving_average_2_1(rng, θ1, θ2, size=500):
        λ = rng.normal(0, 1, size[0] + 2)
        _y = λ[2:] + _θ1 * λ[1:-1] + _θ2 * λ[:-2]
        return _y
    return autocov, moving_average_2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.10 and Figure 8.13
        """
    )
    return


@app.cell
def _(autocov, moving_average_2_1, np, pm, y_obs):
    with pm.Model() as model_ma2:
        _θ1 = pm.Uniform('θ1', -2, 2)
        _θ2 = pm.Uniform('θ2', -1, 1)
        _p1 = pm.Potential('p1', pm.math.switch(_θ1 + _θ2 > -1, 0, -np.inf))
        _p2 = pm.Potential('p2', pm.math.switch(_θ1 - _θ2 < 1, 0, -np.inf))
        _y = pm.Simulator('y', moving_average_2_1, params=[_θ1, _θ2], sum_stat=autocov, epsilon=0.1, observed=y_obs)
        trace_ma2 = pm.sample_smc(3000)
    return (trace_ma2,)


@app.cell
def _(az, trace_ma2):
    az.summary(trace_ma2)
    return


@app.cell
def _(az, plt, trace_ma2):
    az.plot_trace(trace_ma2, kind="rank_bars", figsize=(10, 4))
    plt.savefig("img/chp08/ma2_trace.png")
    return


@app.cell
def _(az, plt, trace_ma2):
    _axes = az.plot_pair(trace_ma2, kind='kde', var_names=['θ1', 'θ2'], marginals=True, figsize=(10, 5), kde_kwargs={'contourf_kwargs': {'cmap': plt.cm.viridis}}, point_estimate='mean', point_estimate_kwargs={'ls': 'none'}, point_estimate_marker_kwargs={'marker': '.', 'facecolor': 'k', 'zorder': 2})
    _axes[1, 0].set_xlim(-2.1, 2.1)
    _axes[1, 0].set_ylim(-1.1, 1.1)
    _axes[1, 0].set_ylabel(_axes[1, 0].get_ylabel(), rotation=0)
    _axes[1, 0].plot([0, 2, -2, 0], [-1, 1, 1, -1], 'C2', lw=2)
    plt.savefig('img/chp08/ma2_triangle.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Model Comparison in the ABC context

        To reproduce the figures in the book, run `loo_abc.py`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##  Model choice via random forest
        """
    )
    return


@app.cell
def _(np):
    def moving_average_1(rng, θ1, size=(500,)):
        λ = rng.normal(0, 1, size[0] + 1)
        _y = λ[2:] + _θ1 * λ[1:-1]
        return _y

    def moving_average_2_2(rng, θ1, θ2, size=(500,)):
        λ = rng.normal(0, 1, size[0] + 2)
        _y = λ[2:] + _θ1 * λ[1:-1] + _θ2 * λ[1:-1]
        return _y
    rng = np.random.default_rng(1346)
    θ1_true_1 = 0.7
    θ2_true_1 = 0.3
    y_obs_1 = moving_average_2_2(rng, θ1_true_1, θ2_true_1)
    return moving_average_1, moving_average_2_2, y_obs_1


@app.cell
def _(np):
    def autocov_1(x, n=2):
        return np.array([np.mean(x[_i:] * x[:-_i]) for _i in range(1, n + 1)])
    return (autocov_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.12
        """
    )
    return


@app.cell
def _(autocov_1, moving_average_1, pm, y_obs_1):
    with pm.Model() as model_ma1:
        _θ1 = pm.Uniform('θ1', -2, 2)
        _y = pm.Simulator('y', moving_average_1, params=[_θ1], sum_stat=autocov_1, epsilon=0.2, observed=y_obs_1)
        idata_ma1 = pm.sample_smc(3000, idata_kwargs={'log_likelihood': True})
    return idata_ma1, model_ma1


@app.cell
def _(autocov_1, moving_average_2_2, np, pm, y_obs_1):
    with pm.Model() as model_ma2_1:
        _θ1 = pm.Uniform('θ1', -2, 2)
        _θ2 = pm.Uniform('θ2', -1, 1)
        _p1 = pm.Potential('p1', pm.math.switch(_θ1 + _θ2 > -1, 0, -np.inf))
        _p2 = pm.Potential('p2', pm.math.switch(_θ1 - _θ2 < 1, 0, -np.inf))
        _y = pm.Simulator('y', moving_average_2_2, params=[_θ1, _θ2], sum_stat=autocov_1, epsilon=0.1, observed=y_obs_1)
        idata_ma2 = pm.sample_smc(3000, idata_kwargs={'log_likelihood': True})
    return idata_ma2, model_ma2_1


@app.cell
def _(idata_ma1, idata_ma2, np):
    mll_ma2 = np.nanmean(np.concatenate([np.hstack(v) for v in idata_ma2.sample_stats.log_marginal_likelihood.values]))
    mll_ma1 = np.nanmean(np.concatenate([np.hstack(v) for v in idata_ma1.sample_stats.log_marginal_likelihood.values]))

    mll_ma2/mll_ma1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.13
        """
    )
    return


@app.cell
def _(az, idata_ma1, idata_ma2):
    cmp = az.compare({"model_ma1":idata_ma1, "model_ma2":idata_ma2})
    cmp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 8.14
        """
    )
    return


@app.cell
def _(
    autocov_1,
    idata_ma1,
    idata_ma2,
    model_ma1,
    model_ma2_1,
    select_model,
    y_obs_1,
):
    from functools import partial
    select_model([(model_ma1, idata_ma1), (model_ma2_1, idata_ma2)], statistics=[partial(autocov_1, n=6)], n_samples=10000, observations=y_obs_1)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
