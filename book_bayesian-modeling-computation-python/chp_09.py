import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 9: End to End Bayesian Workflows
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import arviz as az

    import matplotlib.pyplot as plt
    import pymc as pm
    import numpy as np


    np.random.seed(seed=233423)
    sampling_random_seed = 0
    return az, np, pd, plt, pm


@app.cell
def _(az, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams['figure.dpi'] = 300
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Making a Model and Probably More Than One
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.1 and Figure 9.2
        """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("../data/948363589_T_ONTIME_MARKETING.zip", low_memory=False)
    return (df,)


@app.cell
def _(az, df, plt):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    msn_arrivals = df[(df['DEST'] == 'MSN') & df['ORIGIN'].isin(['MSP', 'DTW'])]['ARR_DELAY']
    az.plot_kde(msn_arrivals.values, ax=_ax)
    _ax.set_yticks([])
    _ax.set_xlabel('Minutes late')
    plt.savefig('img/chp09/arrivaldistributions.png')
    return (msn_arrivals,)


@app.cell
def _(msn_arrivals):
    msn_arrivals.notnull().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.2
        """
    )
    return


@app.cell
def _(msn_arrivals, pm):
    try:
        with pm.Model() as normal_model:
            _normal_mu = ...
            _normal_sd = ...
            _normal_delay = pm.SkewNormal('delays', mu=_normal_mu, sigma=_normal_sd, observed=msn_arrivals)
        with pm.Model() as skew_normal_model:
            skew_normal_alpha = ...
            skew_normal_mu = ...
            skew_normal_sd = ...
            skew_normal_delays = pm.SkewNormal('delays', mu=skew_normal_mu, sigma=skew_normal_sd, alpha=skew_normal_alpha, observed=msn_arrivals)
        with pm.Model() as gumbel_model:
            _gumbel_beta = ...
            _gumbel_mu = ...
            _gumbel_delays = pm.Gumbel('delays', mu=_gumbel_mu, beta=_gumbel_beta, observed=msn_arrivals)
    except:
        pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Choosing Priors and Predictive Priors
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.3
        """
    )
    return


@app.cell
def _(msn_arrivals, pm):
    samples = 1000
    with pm.Model() as normal_model_1:
        _normal_sd = pm.HalfStudentT('sd', sigma=60, nu=5)
        _normal_mu = pm.Normal('mu', 0, 30)
        _normal_delay = pm.Normal('delays', mu=_normal_mu, sigma=_normal_sd, observed=msn_arrivals)
        normal_prior_predictive = pm.sample_prior_predictive()
    with pm.Model() as gumbel_model_1:
        _gumbel_beta = pm.HalfStudentT('beta', sigma=60, nu=5)
        _gumbel_mu = pm.Normal('mu', 0, 20)
        _gumbel_delays = pm.Gumbel('delays', mu=_gumbel_mu, beta=_gumbel_beta, observed=msn_arrivals)
        gumbel_predictive = pm.sample_prior_predictive()
    return (
        gumbel_model_1,
        gumbel_predictive,
        normal_model_1,
        normal_prior_predictive,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.3
        """
    )
    return


@app.cell
def _(az, gumbel_predictive, normal_prior_predictive, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    prior_predictives = {'normal': normal_prior_predictive, 'gumbel': gumbel_predictive}
    for i, (_label, prior_predictive) in enumerate(prior_predictives.items()):
        data = prior_predictive.prior_predictive['delays']
        az.plot_dist(data, ax=_axes[i])
        _axes[i].set_yticks([])
        _axes[i].set_xlim(-300, 300)
        _axes[i].set_title(_label)
    _fig.savefig('img/chp09/Airline_Prior_Predictive.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Inference and Inference Diagnostics
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.4
        """
    )
    return


@app.cell
def _(az, normal_model_1, plt, pm):
    with normal_model_1:
        normal_data = pm.sample(random_seed=0, chains=2)
        az.plot_rank(normal_data)
    plt.savefig('img/chp09/rank_plot_bars_normal.png')
    return (normal_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.5
        """
    )
    return


@app.cell
def _(az, gumbel_model_1, plt, pm):
    with gumbel_model_1:
        gumbel_data = pm.sample(random_seed=0, chains=2, draws=10000, idata_kwargs={'log_likelihood': True})
        az.plot_rank(gumbel_data)
    plt.savefig('img/chp09/rank_plot_bars_gumbel.png')
    return (gumbel_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Posterior Plots
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.6
        """
    )
    return


@app.cell
def _(az, normal_data, plt):
    az.plot_posterior(normal_data, figsize=(10, 3))
    plt.savefig('img/chp09/posterior_plot_delays_normal.png');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.7
        """
    )
    return


@app.cell
def _(az, gumbel_data, plt):
    az.plot_posterior(gumbel_data, figsize=(10, 3))
    plt.savefig('img/chp09/posterior_plot_delays_gumbel.png');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Evaluating Posterior Predictive Distributions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.5
        """
    )
    return


@app.cell
def _(normal_model_1, pm):
    with normal_model_1:
        normal_data_1 = pm.sample(random_seed=0, idata_kwargs={'log_likelihood': True})
        normal_data_1.extend(pm.sample_posterior_predictive(normal_data_1, random_seed=0))
    return (normal_data_1,)


@app.cell
def _(az, normal_data_1, plt):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    az.plot_ppc(normal_data_1, observed=True, num_pp_samples=20, ax=_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gumbel Posterior Predictive
        """
    )
    return


@app.cell
def _(gumbel_data, gumbel_model_1, pm):
    with gumbel_model_1:
        gumbel_data.extend(pm.sample_posterior_predictive(gumbel_data, random_seed=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.8
        """
    )
    return


@app.cell
def _(az, gumbel_data, msn_arrivals, normal_data_1, plt):
    _fig, _ax = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    az.plot_ppc(normal_data_1, observed=False, num_pp_samples=20, ax=_ax[0], color='C4')
    az.plot_kde(msn_arrivals.values, ax=_ax[0], label='Observed')
    az.plot_ppc(gumbel_data, observed=False, num_pp_samples=20, ax=_ax[1], color='C4')
    az.plot_kde(msn_arrivals.values, ax=_ax[1], label='Observed')
    _ax[0].set_title('Normal')
    _ax[0].set_xlabel('')
    _ax[1].set_title('Gumbel')
    _ax[1].legend().remove()
    plt.savefig('img/chp09/delays_model_posterior_predictive.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.9
        """
    )
    return


@app.cell
def _(gumbel_data):
    _gumbel_late = gumbel_data.posterior_predictive['delays'].values.reshape(-1, 336).copy()
    _dist_of_late = (_gumbel_late > 0).sum(axis=1) / 336
    return


@app.cell
def _(az, gumbel_data, msn_arrivals, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _gumbel_late = gumbel_data.posterior_predictive['delays'].values.reshape(-1, 336).copy()
    _dist_of_late = (_gumbel_late > 0).sum(axis=1) / 336
    az.plot_dist(_dist_of_late, ax=_axes[0])
    percent_observed_late = (msn_arrivals > 0).sum() / 336
    _axes[0].axvline(percent_observed_late, c='gray')
    _axes[0].set_title('Test Statistic of On Time Proportion')
    _axes[0].set_yticks([])
    _gumbel_late[_gumbel_late < 0] = np.nan
    median_lateness = np.nanmedian(_gumbel_late, axis=1)
    az.plot_dist(median_lateness, ax=_axes[1])
    median_time_observed_late = msn_arrivals[msn_arrivals >= 0].median()
    _axes[1].axvline(median_time_observed_late, c='gray')
    _axes[1].set_title('Test Statistic of Median Minutes Late')
    _axes[1].set_yticks([])
    plt.savefig('img/chp09/arrival_test_statistics_for_gumbel_posterior_predictive.png')
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
        ### Code 9.6
        """
    )
    return


@app.cell
def _(az, gumbel_data, normal_data_1):
    compare_dict = {'normal': normal_data_1, 'gumbel': gumbel_data}
    comp = az.compare(compare_dict)
    comp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.10
        """
    )
    return


@app.cell
def _(az, gumbel_data, normal_data_1, plt):
    _, _axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    for _label, _model, _ax in zip(('gumbel', 'normal'), (gumbel_data, normal_data_1), _axes):
        az.plot_loo_pit(_model, y='delays', legend=False, use_hdi=True, ax=_ax)
        _ax.set_title(_label)
    plt.savefig('img/chp09/loo_pit_delays.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 9.1
        """
    )
    return


@app.cell
def _(az, gumbel_data, normal_data_1):
    cmp_dict = {'gumbel': gumbel_data, 'normal': normal_data_1}
    cmp = az.compare(cmp_dict)
    cmp
    return cmp, cmp_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.7 and Figure 9.12
        """
    )
    return


@app.cell
def _(az, cmp, plt):
    az.plot_compare(cmp, figsize=(10, 2))
    plt.savefig("img/chp09/model_comparison_airlines.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.12
        """
    )
    return


@app.cell
def _(az, gumbel_data, normal_data_1):
    gumbel_loo = az.loo(gumbel_data, pointwise=True)
    normal_loo = az.loo(normal_data_1, pointwise=True)
    return gumbel_loo, normal_loo


@app.cell
def _(az, cmp_dict, gumbel_loo, normal_loo, np, plt):
    _fig = plt.figure(figsize=(10, 6))
    gs = _fig.add_gridspec(2, 2)
    _ax = _fig.add_subplot(gs[0, :])
    ax1 = _fig.add_subplot(gs[1, 0])
    ax2 = _fig.add_subplot(gs[1, 1])
    diff = gumbel_loo.loo_i - normal_loo.loo_i
    idx = np.abs(diff) > 4
    x_values = np.where(idx)[0]
    y_values = diff[idx].values
    az.plot_elpd(cmp_dict, ax=_ax)
    for x, y in zip(x_values, y_values):
        if x != 158:
            x_pos = x + 4
        else:
            x_pos = x - 15
        _ax.text(x_pos, y - 1, x)
    for _label, elpd_data, _ax in zip(('gumbel', 'normal'), (gumbel_loo, normal_loo), (ax1, ax2)):
        az.plot_khat(elpd_data, ax=_ax)
        _ax.set_title(_label)
        idx = elpd_data.pareto_k > 0.7
        x_values = np.where(idx)[0]
        y_values = elpd_data.pareto_k[idx].values
        for x, y in zip(x_values, y_values):
            if x != 158:
                x_pos = x + 10
            else:
                x_pos = x - 30
            _ax.text(x_pos, y, x)
    ax1.set_ylim(ax2.get_ylim())
    ax2.set_ylabel('')
    ax2.set_yticks([])
    plt.savefig('img/chp09/elpd_plot_delays.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Reward Functions and Decisions
        """
    )
    return


@app.cell
def _(gumbel_data):
    posterior_pred = gumbel_data.posterior_predictive["delays"].values.reshape(-1, 336).copy()
    return (posterior_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.8
        """
    )
    return


@app.cell
def _(np):
    @np.vectorize
    def current_revenue(delay):
        """Calculates revenue """
        if delay >= 0:
            return 300.*delay
        return np.nan
    return (current_revenue,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.9
        """
    )
    return


@app.cell
def _(np):
    def revenue_calculator(posterior_pred, revenue_func):    
        revenue_per_flight = revenue_func(posterior_pred)
        average_revenue = np.nanmean(revenue_per_flight)
        return revenue_per_flight, average_revenue
    return (revenue_calculator,)


@app.cell
def _(current_revenue, posterior_pred, revenue_calculator):
    revenue_per_flight, average_revenue = revenue_calculator(posterior_pred, current_revenue)
    average_revenue
    return (revenue_per_flight,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.13
        """
    )
    return


@app.cell
def _(plt, revenue_per_flight):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.hist(revenue_per_flight.flatten(), bins=30, rwidth=0.9, color='C2')
    _ax.set_yticks([])
    _ax.set_title('Late fee revenue per flight under current fee structure')
    _ax.xaxis.set_major_formatter('${x:1.0f}')
    plt.savefig('img/chp09/late_fee_current_structure_hist.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.10
        """
    )
    return


@app.cell
def _(np):
    @np.vectorize
    def proposed_revenue(delay):
        """Calculates revenue """
        if delay >= 100:
            return 30000.
        elif delay >= 10:
            return 5000.
        elif delay >= 0:
            return 1000.
        else:
            return np.nan
    return (proposed_revenue,)


@app.cell
def _(posterior_pred, proposed_revenue, revenue_calculator):
    revenue_per_flight_proposed, average_revenue_proposed = revenue_calculator(posterior_pred, proposed_revenue)
    average_revenue_proposed
    return (revenue_per_flight_proposed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sharing the Results With a Particular Audience
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 9.14
        """
    )
    return


@app.cell
def _(pd, plt, revenue_per_flight_proposed):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    counts = pd.Series(revenue_per_flight_proposed.flatten()).value_counts()
    counts.index = counts.index.astype(int)
    counts.plot(kind='bar', ax=_ax, color='C2')
    _ax.set_title('Late fee revenue per flight under proposed fee structure')
    _ax.set_yticks([])
    _ax.tick_params(axis='x', labelrotation=0)
    _ax.set_xticklabels([f'${i}' for i in counts.index])
    plt.savefig('img/chp09/late_fee_proposed_structure_hist.png')
    return (counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 9.2
        """
    )
    return


@app.cell
def _(counts):
    counts
    return


@app.cell
def _(counts):
    counts/counts.sum()*100
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Experimental Example: Comparing Between Two Groups
        """
    )
    return


@app.cell
def _(pd):
    composites_df = pd.read_csv("../data/CompositeTensileTest.csv")
    return (composites_df,)


@app.cell
def _(composites_df):
    unidirectional = composites_df["Unidirectional Ultimate Strength (ksi)"].values
    bidirectional = composites_df["Bidirectional Ultimate Strength (ksi)"].values
    return bidirectional, unidirectional


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.11
        """
    )
    return


@app.cell
def _(pm, unidirectional):
    with pm.Model() as unidirectional_model:
        sd = pm.HalfStudentT("sd_uni", 20)
        mu = pm.Normal("mu_uni", 120, 30)
    
        uni_ksi = pm.Normal("uni_ksi", mu=mu, sigma=sd, observed=unidirectional)
    
        uni_data = pm.sample(draws=5000)
    return (uni_data,)


@app.cell
def _(az, plt, uni_data):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    az.plot_dist(uni_data.posterior['mu_uni'], ax=_axes[0])
    az.plot_dist(uni_data.posterior['sd_uni'], ax=_axes[1])
    _axes[0].set_yticks([])
    _axes[1].set_yticks([])
    _fig.savefig('img/chp09/kde_uni.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.12 and Figure 9.15
        """
    )
    return


@app.cell
def _(az, plt, uni_data):
    az.plot_posterior(uni_data, figsize=(12, 3));
    plt.savefig("img/chp09/posterior_uni.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.13
        """
    )
    return


@app.cell
def _(bidirectional, np, pm, unidirectional):
    μ_m = 120
    μ_s = 30
    σ_low = 10
    σ_high = 30
    with pm.Model() as _model:
        uni_mean = pm.Normal('uni_mean', mu=μ_m, sigma=μ_s)
        bi_mean = pm.Normal('bi_mean', mu=μ_m, sigma=μ_s)
        uni_std = pm.Uniform('uni_std', lower=σ_low, upper=σ_high)
        bi_std = pm.Uniform('bi_std', lower=σ_low, upper=σ_high)
        ν = pm.Exponential('ν_minus_one', 1 / 29.0) + 1
        λ1 = uni_std ** (-2)
        λ2 = bi_std ** (-2)
        group1 = pm.StudentT('uni', nu=ν, mu=uni_mean, lam=λ1, observed=unidirectional)
        group2 = pm.StudentT('bi', nu=ν, mu=bi_mean, lam=λ2, observed=bidirectional)
        diff_of_means = pm.Deterministic('Difference of Means', uni_mean - bi_mean)
        diff_of_stds = pm.Deterministic('Difference of Stds', uni_std - bi_std)
        effect_size = pm.Deterministic('Effect Size', diff_of_means / np.sqrt((uni_std ** 2 + bi_std ** 2) / 2))
        t_idata = pm.sample(draws=10000)
    return (t_idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.14
        """
    )
    return


@app.cell
def _(az, plt, t_idata):
    _axes = az.plot_forest(t_idata, var_names=['uni_mean', 'bi_mean'], figsize=(10, 2))
    _axes[0].set_title('Mean Ultimate Strength Estimate: 94.0% HDI')
    plt.savefig('img/chp09/Posterior_Forest_Plot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.15 and Figure 9.17
        """
    )
    return


@app.cell
def _(az, plt, t_idata):
    az.plot_posterior(t_idata, var_names=['Difference of Means','Effect Size'], hdi_prob=.95, ref_val=0, figsize=(12, 3));
    plt.savefig("img/chp09/composite_difference_of_means.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 9.16
        """
    )
    return


@app.cell
def _(az, t_idata):
    az.summary(t_idata, kind="stats")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
