import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 2: Exploratory Analysis of Bayesian Models
        """
    )
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pymc as pm
    from scipy import stats
    return az, np, plt, pm, stats


@app.cell
def _(az, np, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams['figure.dpi'] = 300
    np.random.seed(5201)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding Your Assumptions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.2
        """
    )
    return


@app.cell
def _(np, pm):
    half_length = 3.66
    penalty_point = 11

    def Phi(x):
        """Calculates the standard normal cumulative distribution function."""
        return 0.5 + 0.5 * pm.math.erf(x / 2.0 ** 0.5)
    ppss = []
    sigmas_deg = [5, 20, 60]
    sigmas_rad = np.deg2rad(sigmas_deg)
    for _sigma in sigmas_rad:
        with pm.Model() as _model:
            _σ = pm.HalfNormal('σ', _sigma)
            α = pm.Normal('α', 0, _σ)
            p_goal = pm.Deterministic('p_goal', 2 * Phi(pm.math.arctan(half_length / penalty_point) / _σ) - 1)
            _pps = pm.sample_prior_predictive(250)
            ppss.append(_pps)
    return half_length, penalty_point, ppss, sigmas_deg


@app.cell
def _(half_length, np, penalty_point, plt, ppss, sigmas_deg):
    _fig, _axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'), figsize=(10, 4))
    max_angle = np.arctan(half_length / penalty_point)
    for _sigma, _pps, _ax in zip(sigmas_deg, ppss, _axes):
        cutoff = _pps.prior['p_goal'] > 0.1
        values = _pps.prior['α'].where(cutoff)
        cax = _ax.scatter(values, np.ones_like(values), c=_pps.prior['p_goal'].where(cutoff), marker='.', cmap='viridis_r', vmin=0.1)
        _ax.fill_between(np.linspace(-max_angle, max_angle, 100), 0, 1.01, alpha=0.25)
        _ax.set_yticks([])
        _ax.set_title(f'$\\sigma = \\mathcal{{HN}}({_sigma})$')
        _ax.plot(0, 0, 'o')
    _fig.colorbar(cax, extend='min', ticks=[1, 0.5, 0.1], shrink=0.7, aspect=40)
    plt.savefig('img/chp02/prior_predictive_distributions_00.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.3
        """
    )
    return


@app.cell
def _(az, np, plt):
    from scipy.special import expit
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
    _axes = np.ravel(_axes)
    for dim, _ax in zip([2, 5, 20], _axes):
        β = np.random.normal(0, 1, size=(10000, dim))
        X = np.random.binomial(n=1, p=0.75, size=(dim, 500))
        az.plot_kde(expit(β @ X).mean(1), ax=_ax)
        _ax.set_title(f'{dim} predictors')
        _ax.set_xticks([0, 0.5, 1])
        _ax.set_yticks([0, 1, 2])
    _fig.text(0.34, -0.075, size=18, s='mean of the simulated data')
    plt.savefig('img/chp02/prior_predictive_distributions_01.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding Your Predictions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.4
        """
    )
    return


@app.cell
def _(pm, stats):
    Y = stats.bernoulli(0.7).rvs(100)
    with pm.Model() as _model:
        θ = pm.Beta('θ', 1, 1)
        y_obs = pm.Binomial('y_obs', n=1, p=θ, observed=Y)
        idata_b = pm.sample(1000)
        idata_b.extend(pm.sample_posterior_predictive(idata_b))
    return Y, idata_b


@app.cell
def _(az, idata_b):
    pred_dist = az.extract(idata_b, group="posterior_predictive", num_samples=1000)["y_obs"].values
    pred_dist.sum(0).shape
    return (pred_dist,)


@app.cell
def _(Y, az, np, plt, pred_dist):
    _, _ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    az.plot_dist(pred_dist.sum(0), hist_kwargs={'color': 'C2'}, ax=_ax[0])
    _ax[0].axvline(Y.sum(), color='C4', lw=2.5)
    _ax[0].axvline(pred_dist.sum(0).mean(), color='k', ls='--')
    _ax[0].set_yticks([])
    _ax[0].set_xlabel('number of success')
    pps_ = pred_dist.mean(1)
    _ax[1].plot((np.zeros_like(pps_), np.ones_like(pps_)), (1 - pps_, pps_), 'C1', alpha=0.05)
    _ax[1].plot((0, 1), (1 - Y.mean(), Y.mean()), 'C4', lw=2.5)
    _ax[1].plot((0, 1), (1 - pps_.mean(), pps_.mean()), 'k--')
    _ax[1].set_xticks((0, 1))
    _ax[1].set_xlabel('observed values')
    _ax[1].set_ylabel('probability')
    plt.savefig('img/chp02/posterior_predictive_check.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.5
        """
    )
    return


@app.cell
def _(Y, az, idata_b, plt, pred_dist):
    _, _ax = plt.subplots(1, 2, figsize=(10, 4))
    az.plot_bpv(idata_b, kind='p_value', ax=_ax[0])
    _ax[0].legend([f'bpv={(Y.mean() > pred_dist.mean(1)).mean():.2f}'], handlelength=0)
    az.plot_bpv(idata_b, kind='u_value', ax=_ax[1])
    _ax[1].set_yticks([])
    _ax[1].set_xticks([0.0, 0.5, 1.0])
    plt.savefig('img/chp02/posterior_predictive_check_pu_values.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.6
        """
    )
    return


@app.cell
def _(az, idata_b, plt):
    _, _ax = plt.subplots(1, 2, figsize=(10, 4))
    az.plot_bpv(idata_b, kind='t_stat', t_stat='mean', ax=_ax[0])
    _ax[0].set_title('mean')
    az.plot_bpv(idata_b, kind='t_stat', t_stat='std', ax=_ax[1])
    _ax[1].set_title('standard deviation')
    _ax[1].set_xticks([0.32, 0.41, 0.5])
    plt.savefig('img/chp02/posterior_predictive_check_tstat.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.7
        """
    )
    return


@app.cell
def _(az, np, plt):
    n_obs = 500
    samples = 2000
    y_obs_1 = np.random.normal(0, 1, size=n_obs)
    idata1 = az.from_dict(posterior_predictive={'y': np.random.normal(0.5, 1, size=(1, samples, n_obs))}, observed_data={'y': y_obs_1})
    idata2 = az.from_dict(posterior_predictive={'y': np.random.normal(0, 2, size=(1, samples, n_obs))}, observed_data={'y': y_obs_1})
    idata3 = az.from_dict(posterior_predictive={'y': np.random.normal(0, 0.5, size=(1, samples, n_obs))}, observed_data={'y': y_obs_1})
    idata4 = az.from_dict(posterior_predictive={'y': np.concatenate([np.random.normal(-0.25, 1, size=(1, samples // 2, n_obs)), np.random.normal(0.25, 1, size=(1, samples // 2, n_obs))])}, observed_data={'y': y_obs_1})
    idatas = [idata1, idata2, idata3, idata4]
    _, _axes = plt.subplots(len(idatas), 3, figsize=(10, 10), sharex='col')
    for _idata, _ax in zip(idatas, _axes):
        az.plot_ppc(_idata, ax=_ax[0], color='C1', alpha=0.01, mean=False, legend=False)
        az.plot_kde(_idata.observed_data['y'].values, ax=_ax[0], plot_kwargs={'color': 'C4', 'zorder': 3})
        _ax[0].set_xlabel('')
        az.plot_bpv(_idata, kind='p_value', ax=_ax[1])
        az.plot_bpv(_idata, kind='u_value', ax=_ax[2])
        _ax[2].set_yticks([])
        _ax[2].set_xticks([0.0, 0.5, 1.0])
        for _ax_ in _ax:
            _ax_.set_title('')
    plt.savefig('img/chp02/posterior_predictive_many_examples.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Diagnosing Numerical Inference
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.1
        """
    )
    return


@app.cell
def _(np):
    np.random.seed(5201)
    return


@app.cell
def _(np, stats):
    good_chains = stats.beta.rvs(2, 5,size=(2, 2000))
    bad_chains0 = np.random.normal(np.sort(good_chains, axis=None), 0.05,
                                   size=4000).reshape(2, -1)

    bad_chains1 = good_chains.copy()
    for i in np.random.randint(1900, size=4):
        bad_chains1[i%2:,i:i+100] = np.random.beta(i, 950, size=100)

    chains = {"good_chains":good_chains,
              "bad_chains0":bad_chains0,
              "bad_chains1":bad_chains1}
    return (chains,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.2
        """
    )
    return


@app.cell
def _(az, chains):
    az.ess(chains)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.3 and Figure 2.8
        """
    )
    return


@app.cell
def _(az, chains, plt):
    _, _axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex=True)
    az.plot_ess(chains, kind='local', ax=_axes[0])
    az.plot_ess(chains, kind='quantile', ax=_axes[1])
    for _ax_ in _axes[0]:
        _ax_.set_xlabel('')
    for _ax_ in _axes[1]:
        _ax_.set_title('')
    for _ax_ in _axes[:, 1:].ravel():
        _ax_.set_ylabel('')
    plt.ylim(-100, 5000)
    plt.savefig('img/chp02/plot_ess.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.4
        """
    )
    return


@app.cell
def _(az, chains):
    az.rhat(chains)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.5
        """
    )
    return


@app.cell
def _(az, chains):
    az.mcse(chains)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.6 and Figure 2.9
        """
    )
    return


@app.cell
def _(az, chains, plt):
    _, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    az.plot_mcse(chains, ax=_axes)
    for _ax_ in _axes[1:]:
        _ax_.set_ylabel('')
        _ax_.set_ylim(0, 0.15)
    plt.savefig('img/chp02/plot_mcse.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.7
        """
    )
    return


@app.cell
def _(az, chains):
    az.summary(chains, kind="diagnostics")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.8 and Figure 2.10
        """
    )
    return


@app.cell
def _(az, chains, plt):
    az.plot_trace(chains)
    plt.savefig("img/chp02/trace_plots.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.9 and Figure 2.11
        """
    )
    return


@app.cell
def _(az, chains, plt):
    az.plot_autocorr(chains, combined=True, figsize=(12, 4))
    plt.savefig('img/chp02/autocorrelation_plot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.10 and Figure 2.12
        """
    )
    return


@app.cell
def _(az, chains, plt):
    _, _axes = plt.subplots(1, 3, figsize=(12, 4))
    az.plot_rank(chains, kind='bars', ax=_axes)
    for _ax_ in _axes[1:]:
        _ax_.set_ylabel('')
        _ax_.set_yticks([])
    plt.savefig('img/chp02/rank_plot_bars.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.11 and Figure 2.13
        """
    )
    return


@app.cell
def _(az, chains, plt):
    az.plot_rank(chains, kind="vlines", figsize=(12, 4))
    plt.savefig('img/chp02/rank_plot_vlines.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.12, 2.13, and 2.14
        """
    )
    return


@app.cell
def _(np, pm):
    with pm.Model() as model_0:
        θ1 = pm.Normal('θ1', 0, 1, testval=0.1)
        θ2 = pm.Uniform('θ2', -θ1, θ1)
        idata_0 = pm.sample(return_inferencedata=True)
    with pm.Model() as model_1:
        θ1 = pm.HalfNormal('θ1', 1 / (2 / np.pi) ** 0.5)
        θ2 = pm.Uniform('θ2', -θ1, θ1)
        idata_1 = pm.sample(return_inferencedata=True)
    with pm.Model() as model_1bis:
        θ1 = pm.HalfNormal('θ1', 1 / (2 / np.pi) ** 0.5)
        θ2 = pm.Uniform('θ2', -θ1, θ1)
        idata_1bis = pm.sample(return_inferencedata=True, target_accept=0.95)
    idatas_1 = [idata_0, idata_1, idata_1bis]
    return (idatas_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.14
        """
    )
    return


@app.cell
def _(az, idatas_1, plt):
    _fig, _axes = plt.subplots(6, 2, figsize=(10, 10))
    _axes = _axes.reshape(3, 2, 2)
    for _idata, _ax, color in zip(idatas_1, _axes, ['0.95', '1', '0.95']):
        az.plot_trace(_idata, kind='rank_vlines', axes=_ax)
        [_ax_.set_facecolor(color) for _ax_ in _ax.ravel()]
    _fig.text(0.45, 1, s='model 0', fontsize=16)
    _fig.text(0.45, 0.67, s='model 1', fontsize=16)
    _fig.text(0.45, 0.33, s='model 1bis', fontsize=16)
    plt.savefig('img/chp02/divergences_trace.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.15
        """
    )
    return


@app.cell
def _(az, idatas_1, plt):
    _, _axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
    for _idata, _ax, _model in zip(idatas_1, _axes, ['model 0', 'model 1', 'model 1bis']):
        az.plot_pair(_idata, divergences=True, scatter_kwargs={'color': 'C1'}, divergences_kwargs={'color': 'C4'}, ax=_ax)
        _ax.set_xlabel('')
        _ax.set_ylabel('')
        _ax.set_title(_model)
    _axes[0].set_ylabel('θ2', rotation=0, labelpad=15)
    _axes[1].set_xlabel('θ1', labelpad=10)
    plt.savefig('img/chp02/divergences_pair.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Model Comparison
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 2.15
        """
    )
    return


@app.cell
def _(np):
    np.random.seed(90210)
    return


@app.cell
def _(np, pm):
    y_obs_2 = np.random.normal(0, 1, size=100)
    idatas_cmp = {}
    with pm.Model() as mA:
        _σ = pm.HalfNormal('σ', 1)
        y = pm.SkewNormal('y', mu=0, sigma=_σ, alpha=1, observed=y_obs_2)
        idataA = pm.sample(idata_kwargs={'log_likelihood': True})
        idataA.extend(pm.sample_posterior_predictive(idataA))
        idatas_cmp['mA'] = idataA
    with pm.Model() as mB:
        _σ = pm.HalfNormal('σ', 1)
        y = pm.Normal('y', 0, _σ, observed=y_obs_2)
        idataB = pm.sample(idata_kwargs={'log_likelihood': True})
        idataB.extend(pm.sample_posterior_predictive(idataB))
        idatas_cmp['mB'] = idataB
    with pm.Model() as mC:
        μ = pm.Normal('μ', 0, 1)
        _σ = pm.HalfNormal('σ', 1)
        y = pm.Normal('y', μ, _σ, observed=y_obs_2)
        idataC = pm.sample(idata_kwargs={'log_likelihood': True})
        idataC.extend(pm.sample_posterior_predictive(idataC))
        idatas_cmp['mC'] = idataC
    return idatas_cmp, y_obs_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 2.1
        """
    )
    return


@app.cell
def _(az, idatas_cmp):
    cmp = az.compare(idatas_cmp)
    cmp.round(2)
    return (cmp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.16
        """
    )
    return


@app.cell
def _(az, cmp, plt):
    az.plot_compare(cmp, figsize=(9, 3))
    plt.savefig("img/chp02/compare_dummy.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.17
        """
    )
    return


@app.cell
def _(az, idatas_cmp, plt):
    az.plot_elpd(idatas_cmp, figsize=(10, 5), plot_kwargs={"marker":"."}, threshold=2);
    plt.savefig("img/chp02/elpd_dummy.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.18
        """
    )
    return


@app.cell
def _(az, idatas_cmp, plt):
    _, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for idx, (_model, _ax) in enumerate(zip(('mA', 'mB', 'mC'), _axes)):
        loo_ = az.loo(idatas_cmp[_model], pointwise=True)
        az.plot_khat(loo_, ax=_ax, threshold=0.09, show_hlines=True, hlines_kwargs={'hlines': 0.09, 'ls': '--'})
        _ax.set_title(_model)
        if idx:
            _axes[idx].set_ylabel('')
        if not idx % 2:
            _axes[idx].set_xlabel('')
    plt.savefig('img/chp02/loo_k_dummy.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.19
        """
    )
    return


@app.cell
def _(az, plt, y_obs_2):
    az.plot_kde(y_obs_2, rug=True)
    plt.yticks([])
    for da, loc in zip([34, 49, 72, 75, 95], [-0.065, -0.05, -0.065, -0.065, -0.065]):
        plt.text(y_obs_2[da], loc, f'{da}')
    plt.text(y_obs_2[78], loc, '78', fontweight='bold')
    plt.savefig('img/chp02/elpd_and_khat.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 2.20
        """
    )
    return


@app.cell
def _(az, idatas_cmp, plt):
    _, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for _model, _ax in zip(('mA', 'mB', 'mC'), _axes):
        az.plot_loo_pit(idatas_cmp[_model], y='y', legend=False, use_hdi=True, ax=_ax)
        _ax.set_title(_model)
        _ax.set_xticks([0, 0.5, 1])
        _ax.set_yticks([0, 1, 2])
    plt.savefig('img/chp02/loo_pit_dummy.png')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
