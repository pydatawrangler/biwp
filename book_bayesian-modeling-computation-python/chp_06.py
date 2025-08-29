import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 6: Time Series

        Note: This notebook is failing execution for us at the moment. Use it as a reference. We will have it updated shortly
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import arviz as az
    import pandas as pd
    import numpy as np

    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    tfb = tfp.bijectors
    root = tfd.JointDistributionCoroutine.Root

    import datetime
    print(f"Last Run {datetime.datetime.now()}")
    return az, np, pd, plt, tf, tfb, tfp


@app.cell
def _(az, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams["figure.dpi"] = 300
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Time Series Analysis as a Regression Problem
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.1
        """
    )
    return


@app.cell
def _(np, pd):
    co2_by_month = pd.read_csv("../data/monthly_mauna_loa_co2.csv")
    co2_by_month["date_month"] = pd.to_datetime(co2_by_month["date_month"])
    co2_by_month["CO2"] = co2_by_month["CO2"].astype(np.float32)
    co2_by_month.set_index("date_month", drop=True, inplace=True)

    num_forecast_steps = 12 * 10  # Forecast the final ten years, given previous data
    co2_by_month_training_data = co2_by_month[:-num_forecast_steps]
    co2_by_month_testing_data = co2_by_month[-num_forecast_steps:]
    return (
        co2_by_month,
        co2_by_month_testing_data,
        co2_by_month_training_data,
        num_forecast_steps,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.1
        """
    )
    return


@app.cell
def _(co2_by_month_testing_data, co2_by_month_training_data, plt):
    def plot_co2_data(fig_ax=None):
        if not fig_ax:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        else:
            fig, ax = fig_ax
        ax.plot(co2_by_month_training_data, label="training data")
        ax.plot(co2_by_month_testing_data, color="C4", label="testing data")
        ax.legend()
        ax.set(
            ylabel="Atmospheric CO₂ concentration (ppm)",
            xlabel="Year"
        )
        ax.text(0.99, .02,
                """Source: Scripps Institute for Oceanography CO₂ program
                http://scrippsco2.ucsd.edu/data/atmospheric_co2/primary_mlo_co2_record""",
                transform=ax.transAxes,
                horizontalalignment="right",
                alpha=0.5)
        fig.autofmt_xdate()
        return fig, ax


    _ = plot_co2_data()
    plt.savefig("img/chp06/fig1_co2_by_month.png")
    return (plot_co2_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.2 and Figure 6.2
        """
    )
    return


@app.cell
def _(co2_by_month, np, num_forecast_steps, pd, plt):
    trend_all = np.linspace(0., 1., len(co2_by_month))[..., None]
    trend_all = trend_all.astype(np.float32)
    trend = trend_all[:-num_forecast_steps, :]

    seasonality_all = pd.get_dummies(
       co2_by_month.index.month).values.astype(np.float32)
    seasonality = seasonality_all[:-num_forecast_steps, :]

    fig, ax = plt.subplots(figsize=(10, 4))
    X_subset = np.concatenate([trend, seasonality], axis=-1)[-50:]
    im = ax.imshow(X_subset.T, cmap="cet_gray_r")

    label_loc = np.arange(1, 50, 12)
    ax.set_xticks(label_loc)
    ax.set_yticks([])
    ax.set_xticklabels(co2_by_month.index.year[-50:][label_loc])
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=.6)

    plt.savefig("img/chp06/fig2_sparse_design_matrix.png")
    return seasonality, seasonality_all, trend, trend_all


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.3
        """
    )
    return


@app.cell
def _(seasonality, tf, tfp, trend):
    tfd_1 = tfp.distributions
    root_1 = tfd_1.JointDistributionCoroutine.Root

    @tfd_1.JointDistributionCoroutine
    def ts_regression_model():
        intercept = (yield root_1(tfd_1.Normal(0.0, 100.0, name='intercept')))
        trend_coeff = (yield root_1(tfd_1.Normal(0.0, 10.0, name='trend_coeff')))
        seasonality_coeff = (yield root_1(tfd_1.Sample(tfd_1.Normal(0.0, 1.0), sample_shape=seasonality.shape[-1], name='seasonality_coeff')))
        noise = (yield root_1(tfd_1.HalfCauchy(loc=0.0, scale=5.0, name='noise_sigma')))
        y_hat = intercept[..., None] + tf.einsum('ij,...->...i', trend, trend_coeff) + tf.einsum('ij,...j->...i', seasonality, seasonality_coeff)
        observed = (yield tfd_1.Independent(tfd_1.Normal(y_hat, noise[..., None]), reinterpreted_batch_ndims=1, name='observed'))
    ts_regression_model.log_prob_parts(ts_regression_model.sample(4))
    return root_1, tfd_1, ts_regression_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.4 and Figure 6.3
        """
    )
    return


@app.cell
def _(co2_by_month, num_forecast_steps, plt, tf, ts_regression_model):
    prior_samples = ts_regression_model.sample(100)
    prior_predictive_timeseries = prior_samples[-1]
    fig_1, ax_1 = plt.subplots(figsize=(10, 5))
    ax_1.plot(co2_by_month.index[:-num_forecast_steps], tf.transpose(prior_predictive_timeseries), alpha=0.5)
    ax_1.set_xlabel('Year')
    fig_1.autofmt_xdate()
    plt.savefig('img/chp06/fig3_prior_predictive1.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.5
        """
    )
    return


@app.cell
def _(tf, tfp):
    run_mcmc = tf.function(
        tfp.experimental.mcmc.windowed_adaptive_nuts,
        autograph=False, jit_compile=True)
    return (run_mcmc,)


@app.cell
def _(co2_by_month_training_data, run_mcmc, ts_regression_model):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    mcmc_samples, sampler_stats = run_mcmc(
        1000, ts_regression_model, n_chains=4, num_adaptation_steps=1000,
        observed=co2_by_month_training_data["CO2"].values[None, ...])
    return mcmc_samples, sampler_stats


@app.cell
def _(az, mcmc_samples, np, sampler_stats):
    regression_idata = az.from_dict(
        posterior={
            k:np.swapaxes(v.numpy(), 1, 0)
            for k, v in mcmc_samples._asdict().items()},
        sample_stats={
            k:np.swapaxes(sampler_stats[k], 1, 0)
            for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]}
    )

    az.summary(regression_idata)
    return (regression_idata,)


@app.cell
def _(az, regression_idata):
    axes = az.plot_trace(regression_idata, compact=True);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.6
        """
    )
    return


@app.cell
def _(
    mcmc_samples,
    regression_idata,
    seasonality_all,
    tf,
    tfd_1,
    trend_all,
    ts_regression_model,
):
    posterior_dist, posterior_and_predictive = ts_regression_model.sample_distributions(value=mcmc_samples)
    posterior_predictive_samples = posterior_and_predictive[-1]
    posterior_predictive_dist = posterior_dist[-1]
    nchains = regression_idata.posterior.dims['chain']
    trend_posterior = mcmc_samples.intercept + tf.einsum('ij,...->i...', trend_all, mcmc_samples.trend_coeff)
    seasonality_posterior = tf.einsum('ij,...j->i...', seasonality_all, mcmc_samples.seasonality_coeff)
    y_hat = trend_posterior + seasonality_posterior
    posterior_predictive_dist = tfd_1.Normal(y_hat, mcmc_samples.noise_sigma)
    posterior_predictive_samples = posterior_predictive_dist.sample()
    return (
        nchains,
        posterior_predictive_samples,
        seasonality_posterior,
        trend_posterior,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.4
        """
    )
    return


@app.cell
def _(
    co2_by_month,
    nchains,
    num_forecast_steps,
    plt,
    seasonality_posterior,
    trend_posterior,
):
    fig_2, ax_2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for i in range(nchains):
        ax_2[0].plot(co2_by_month.index[:-num_forecast_steps], trend_posterior[:-num_forecast_steps, -100:, i], alpha=0.05)
        ax_2[1].plot(co2_by_month.index[:-num_forecast_steps], seasonality_posterior[:-num_forecast_steps, -100:, i], alpha=0.05)
    ax_2[0].set_title('Trend (Linear)')
    ax_2[1].set_title('Seasonality (Month of the year effect)')
    ax_2[1].set_xlabel('Year')
    fig_2.autofmt_xdate()
    plt.savefig('img/chp06/fig4_posterior_predictive_components1.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.5
        """
    )
    return


@app.cell
def _(co2_by_month, plot_co2_data, plt, posterior_predictive_samples, tf):
    fig_3, ax_3 = plt.subplots(1, 1, figsize=(10, 5))
    sample_shape = posterior_predictive_samples.shape[1:]
    ax_3.plot(co2_by_month.index, tf.reshape(posterior_predictive_samples, [-1, tf.math.reduce_prod(sample_shape)])[:, :500], color='gray', alpha=0.01)
    plot_co2_data((fig_3, ax_3))
    plt.savefig('img/chp06/fig5_posterior_predictive1.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.7
        """
    )
    return


@app.cell
def _(np):
    n_changepoints = 8
    n_tp = 500
    t = np.linspace(0, 1, n_tp)
    s = np.linspace(0, 1, n_changepoints + 2)[1:-1]
    A = t[:, None] > s
    k, m = (2.5, 40)
    delta = np.random.laplace(0.1, size=n_changepoints)
    growth = (k + A @ delta) * t
    offset = m + A @ (-s * delta)
    trend_1 = growth + offset
    return A, growth, offset, s, t, trend_1


@app.cell
def _(A, growth, np, offset, plt, s, t, trend_1):
    _, ax_4 = plt.subplots(4, 1, figsize=(10, 10))
    ax_4[0].imshow(A.T, cmap='cet_gray_r', aspect='auto', interpolation='none')
    ax_4[0].axis('off')
    ax_4[0].set_title('$\\mathbf{A}$')
    ax_4[1].plot(t, growth, lw=2)
    ax_4[1].set_title('$(k + \\mathbf{A}\\delta) t$')
    ax_4[2].plot(t, offset, lw=2)
    ax_4[2].set_title('$m + \\mathbf{A} \\gamma$')
    ax_4[3].plot(t, trend_1, lw=2)
    ax_4[3].set_title('Step linear function as trend')
    lines = [np.where(t > s_)[0][0] for s_ in s]
    for ax_ in ax_4[1:]:
        ax_.vlines(t[lines], *ax_.get_ylim(), linestyles='--')
    plt.savefig('img/chp06/fig6_step_linear_function.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.8 and Figure 6.7
        """
    )
    return


@app.cell
def _(np, plt):
    def gen_fourier_basis(t, p=365.25, n=3):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)
    n_tp_1 = 500
    p = 12
    t_monthly = np.asarray([i % p for i in range(n_tp_1)])
    monthly_X = gen_fourier_basis(t_monthly, p=p, n=3)
    fig_4, ax_5 = plt.subplots(figsize=(10, 3))
    ax_5.plot(monthly_X[:p * 2, 0])
    ax_5.plot(monthly_X[:p * 2, 1:], alpha=0.25)
    plt.savefig('img/chp06/fig7_fourier_basis.png')
    return (gen_fourier_basis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.9
        """
    )
    return


@app.cell
def _(
    co2_by_month_training_data,
    gen_fourier_basis,
    np,
    root_1,
    seasonality_all,
    tf,
    tfd_1,
):
    n_changepoints_1 = 12
    n_tp_2 = seasonality_all.shape[0]
    t_1 = np.linspace(0, 1, n_tp_2, dtype=np.float32)
    s_1 = np.linspace(0, max(t_1), n_changepoints_1 + 2, dtype=np.float32)[1:-1]
    A_1 = (t_1[:, None] > s_1).astype(np.float32)
    X_pred = gen_fourier_basis(np.where(seasonality_all)[1], p=seasonality_all.shape[-1], n=6)
    n_pred = X_pred.shape[-1]

    def gen_gam_jd(training=True):

        @tfd_1.JointDistributionCoroutine
        def gam():
            beta = (yield root_1(tfd_1.Sample(tfd_1.Normal(0.0, 1.0), sample_shape=n_pred, name='beta')))
            seasonality = tf.einsum('ij,...j->...i', X_pred, beta)
            k = (yield root_1(tfd_1.HalfNormal(10.0, name='k')))
            m = (yield root_1(tfd_1.Normal(co2_by_month_training_data['CO2'].mean(), scale=5.0, name='m')))
            tau = (yield root_1(tfd_1.HalfNormal(10.0, name='tau')))
            delta = (yield tfd_1.Sample(tfd_1.Laplace(0.0, tau), sample_shape=n_changepoints_1, name='delta'))
            growth_rate = k[..., None] + tf.einsum('ij,...j->...i', A_1, delta)
            offset = m[..., None] + tf.einsum('ij,...j->...i', A_1, -s_1 * delta)
            trend = growth_rate * t_1 + offset
            y_hat = seasonality + trend
            if training:
                y_hat = y_hat[..., :co2_by_month_training_data.shape[0]]
            noise_sigma = (yield root_1(tfd_1.HalfNormal(scale=5.0, name='noise_sigma')))
            observed = (yield tfd_1.Independent(tfd_1.Normal(y_hat, noise_sigma[..., None]), reinterpreted_batch_ndims=1, name='observed'))
        return gam
    gam = gen_gam_jd()
    return A_1, X_pred, gam, gen_gam_jd, n_changepoints_1, n_pred, s_1, t_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Autoregressive Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.8
        """
    )
    return


@app.cell
def _(co2_by_month, gam, num_forecast_steps, plt, tf):
    prior_samples_1 = gam.sample(100)
    prior_predictive_timeseries_1 = prior_samples_1[-1]
    fig_5, ax_6 = plt.subplots(figsize=(10, 5))
    ax_6.plot(co2_by_month.index[:-num_forecast_steps], tf.transpose(prior_predictive_timeseries_1), alpha=0.5)
    ax_6.set_xlabel('Year')
    fig_5.autofmt_xdate()
    plt.savefig('img/chp06/fig8_prior_predictive2.png')
    return


@app.cell
def _(co2_by_month_training_data, gam, run_mcmc, tf):
    mcmc_samples_1, sampler_stats_1 = run_mcmc(1000, gam, n_chains=4, num_adaptation_steps=1000, seed=tf.constant([-12341, 62345], dtype=tf.int32), observed=co2_by_month_training_data.T)
    return mcmc_samples_1, sampler_stats_1


@app.cell
def _(az, mcmc_samples_1, np, sampler_stats_1):
    gam_idata = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_1._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_1[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    axes_1 = az.plot_trace(gam_idata, compact=True)
    return (gam_idata,)


@app.cell
def _(gen_gam_jd, mcmc_samples_1):
    gam_full = gen_gam_jd(training=False)
    posterior_dists, _ = gam_full.sample_distributions(value=mcmc_samples_1)
    return (posterior_dists,)


@app.cell
def _(A_1, X_pred, co2_by_month, mcmc_samples_1, nchains, plt, s_1, t_1, tf):
    _, ax_7 = plt.subplots(2, 1, figsize=(10, 5))
    k_1, m_1, tau, delta_1 = mcmc_samples_1[1:5]
    growth_rate = k_1[..., None] + tf.einsum('ij,...j->...i', A_1, delta_1)
    offset_1 = m_1[..., None] + tf.einsum('ij,...j->...i', A_1, -s_1 * delta_1)
    trend_posterior_1 = growth_rate * t_1 + offset_1
    seasonality_posterior_1 = tf.einsum('ij,...j->...i', X_pred, mcmc_samples_1[0])
    for i_1 in range(nchains):
        ax_7[0].plot(co2_by_month.index, trend_posterior_1[-100:, i_1, :].numpy().T, alpha=0.05)
        ax_7[1].plot(co2_by_month.index, seasonality_posterior_1[-100:, i_1, :].numpy().T, alpha=0.05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.9
        """
    )
    return


@app.cell
def _(co2_by_month, np, plot_co2_data, plt, posterior_dists):
    fig_6, ax_8 = plt.subplots(1, 1, figsize=(10, 5))
    fitted_with_forecast = posterior_dists[-1].distribution.sample().numpy()
    ax_8.plot(co2_by_month.index, fitted_with_forecast[-100:, 0, :].T, color='gray', alpha=0.1)
    ax_8.plot(co2_by_month.index, fitted_with_forecast[-100:, 1, :].T, color='gray', alpha=0.1)
    plot_co2_data((fig_6, ax_8))
    average_forecast = np.mean(fitted_with_forecast, axis=(0, 1)).T
    ax_8.plot(co2_by_month.index, average_forecast, ls='--', label='GAM forecast', alpha=0.5)
    plt.savefig('img/chp06/fig9_posterior_predictive2.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.10 and Figure 6.10
        """
    )
    return


@app.cell
def _(np, plt, root_1, tf, tfd_1):
    n_t = 200

    @tfd_1.JointDistributionCoroutine
    def ar1_with_forloop():
        sigma = (yield root_1(tfd_1.HalfNormal(1.0)))
        rho = (yield root_1(tfd_1.Uniform(-1.0, 1.0)))
        x0 = (yield tfd_1.Normal(0.0, sigma))
        x = [x0]
        for i in range(1, n_t):
            x_i = (yield tfd_1.Normal(x[i - 1] * rho, sigma))
            x.append(x_i)
    nplot = 4
    fig_7, axes_2 = plt.subplots(nplot, 1)
    for ax_9, rho in zip(axes_2, np.linspace(-1.01, 1.01, nplot)):
        test_samples = ar1_with_forloop.sample(value=(1.0, rho))
        ar1_samples = tf.stack(test_samples[2:])
        ax_9.plot(ar1_samples, alpha=0.5, label='$\\rho$=%.2f' % rho)
        ax_9.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.0, fontsize=10)
        ax_9.get_xaxis().set_visible(False)
    fig_7.suptitle('AR(1) process with varies autoregressive coefficient ($\\rho$)')
    plt.savefig('img/chp06/fig10_ar1_process.png')
    return (n_t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.11
        """
    )
    return


@app.cell
def _(n_t, np, plt, root_1, tf, tfd_1):
    @tfd_1.JointDistributionCoroutine
    def ar1_without_forloop():
        sigma = (yield root_1(tfd_1.HalfNormal(1.0)))
        rho = (yield root_1(tfd_1.Uniform(-1.0, 1.0)))

        def ar1_fun(x):
            x_tm1 = tf.concat([tf.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)
            loc = x_tm1 * rho[..., None]
            return tfd_1.Independent(tfd_1.Normal(loc=loc, scale=sigma[..., None]), reinterpreted_batch_ndims=1)
        dist = (yield tfd_1.Autoregressive(distribution_fn=ar1_fun, sample0=tf.zeros([n_t], dtype=rho.dtype), num_steps=n_t))
    seed = [1000, 5234]
    _, ax_10 = plt.subplots(figsize=(10, 5))
    rho_1 = np.linspace(-1.01, 1.01, 5)
    sigma = np.ones(5)
    test_samples_1 = ar1_without_forloop.sample(value=(sigma, rho_1), seed=seed)
    ar1_samples_1 = tf.transpose(test_samples_1[-1])
    ax_10.plot(ar1_samples_1, alpha=0.5)
    ax_10.set_title('AR(1) process with varies autoregressive coefficient (rho)')
    return (seed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A more general way to implement this is to use a Lag operator $B$
        """
    )
    return


@app.cell
def _(np):
    B = np.diag(np.ones(5 - 1), -1)
    B @ B
    return


@app.cell
def _(n_t, np, plt, root_1, seed, tf, tfd_1):
    B_1 = np.diag(np.ones(n_t - 1), -1)

    @tfd_1.JointDistributionCoroutine
    def ar1_lag_operator():
        sigma = (yield root_1(tfd_1.HalfNormal(1.0, name='sigma')))
        rho = (yield root_1(tfd_1.Uniform(-1.0, 1.0, name='rho')))

        def ar1_fun(x):
            loc = tf.einsum('ij,...j->...i', B_1, x) * rho[..., None]
            return tfd_1.Independent(tfd_1.Normal(loc=loc, scale=sigma[..., None]), reinterpreted_batch_ndims=1)
        dist = (yield tfd_1.Autoregressive(distribution_fn=ar1_fun, sample0=tf.zeros([n_t], dtype=rho.dtype), num_steps=n_t, name='ar1'))
    _, ax_11 = plt.subplots(figsize=(10, 5))
    rho_2 = np.linspace(-1.01, 1.01, 5)
    sigma_1 = np.ones(5)
    test_samples_2 = ar1_lag_operator.sample(value=(sigma_1, rho_2), seed=seed)
    ar1_samples_2 = tf.transpose(test_samples_2[-1])
    ax_11.plot(ar1_samples_2, alpha=0.5)
    ax_11.set_title('AR(1) process with varies autoregressive coefficient (rho)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that since we are using a stateless RNG seeding, we got the same result (yay!)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.12
        """
    )
    return


@app.cell
def _(
    A_1,
    X_pred,
    co2_by_month_training_data,
    n_changepoints_1,
    n_pred,
    root_1,
    s_1,
    t_1,
    tf,
    tfd_1,
):
    def gam_trend_seasonality():
        beta = (yield root_1(tfd_1.Sample(tfd_1.Normal(0.0, 1.0), sample_shape=n_pred, name='beta')))
        seasonality = tf.einsum('ij,...j->...i', X_pred, beta)
        k = (yield root_1(tfd_1.HalfNormal(10.0, name='k')))
        m = (yield root_1(tfd_1.Normal(co2_by_month_training_data['CO2'].mean(), scale=5.0, name='m')))
        tau = (yield root_1(tfd_1.HalfNormal(10.0, name='tau')))
        delta = (yield tfd_1.Sample(tfd_1.Laplace(0.0, tau), sample_shape=n_changepoints_1, name='delta'))
        growth_rate = k[..., None] + tf.einsum('ij,...j->...i', A_1, delta)
        offset = m[..., None] + tf.einsum('ij,...j->...i', A_1, -s_1 * delta)
        trend = growth_rate * t_1 + offset
        noise_sigma = (yield root_1(tfd_1.HalfNormal(scale=5.0, name='noise_sigma')))
        return (seasonality, trend, noise_sigma)

    def generate_gam(training=True):

        @tfd_1.JointDistributionCoroutine
        def gam():
            seasonality, trend, noise_sigma = (yield from gam_trend_seasonality())
            y_hat = seasonality + trend
            if training:
                y_hat = y_hat[..., :co2_by_month_training_data.shape[0]]
            observed = (yield tfd_1.Independent(tfd_1.Normal(y_hat, noise_sigma[..., None]), reinterpreted_batch_ndims=1, name='observed'))
        return gam
    gam_1 = generate_gam()
    return (gam_trend_seasonality,)


@app.cell
def _(
    co2_by_month_training_data,
    gam_trend_seasonality,
    plt,
    root_1,
    tf,
    tfd_1,
):
    def generate_gam_ar_likelihood(training=True):

        @tfd_1.JointDistributionCoroutine
        def gam_with_ar_likelihood():
            seasonality, trend, noise_sigma = (yield from gam_trend_seasonality())
            y_hat = seasonality + trend
            if training:
                y_hat = y_hat[..., :co2_by_month_training_data.shape[0]]
            rho = (yield root_1(tfd_1.Uniform(-1.0, 1.0, name='rho')))

            def ar_fun(y):
                loc = tf.concat([tf.zeros_like(y[..., :1]), y[..., :-1]], axis=-1) * rho[..., None] + y_hat
                return tfd_1.Independent(tfd_1.Normal(loc=loc, scale=noise_sigma[..., None]), reinterpreted_batch_ndims=1)
            observed = (yield tfd_1.Autoregressive(distribution_fn=ar_fun, sample0=tf.zeros_like(y_hat), num_steps=1, name='observed'))
        return gam_with_ar_likelihood
    gam_with_ar_likelihood = generate_gam_ar_likelihood()
    plt.plot(tf.transpose(gam_with_ar_likelihood.sample(50)[-1]))
    return gam_with_ar_likelihood, generate_gam_ar_likelihood


@app.cell
def _(co2_by_month_training_data, gam_with_ar_likelihood, run_mcmc, tf):
    mcmc_samples_2, sampler_stats_2 = run_mcmc(1000, gam_with_ar_likelihood, n_chains=4, num_adaptation_steps=1000, seed=tf.constant([-234272345, 73564234], dtype=tf.int32), observed=co2_by_month_training_data.T)
    return mcmc_samples_2, sampler_stats_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.13
        """
    )
    return


@app.cell
def _(az, mcmc_samples_2, np, sampler_stats_2):
    gam_ar_likelihood_idata = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_2._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_2[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    axes_3 = az.plot_trace(gam_ar_likelihood_idata, compact=True)
    return (gam_ar_likelihood_idata,)


@app.cell
def _(A_1, X_pred, co2_by_month, mcmc_samples_2, nchains, plt, s_1, t_1, tf):
    _, ax_12 = plt.subplots(2, 1, figsize=(10, 5))
    k_2, m_2, tau_1, delta_2 = mcmc_samples_2[1:5]
    growth_rate_1 = k_2[..., None] + tf.einsum('ij,...j->...i', A_1, delta_2)
    offset_2 = m_2[..., None] + tf.einsum('ij,...j->...i', A_1, -s_1 * delta_2)
    trend_posterior_2 = growth_rate_1 * t_1 + offset_2
    seasonality_posterior_2 = tf.einsum('ij,...j->...i', X_pred, mcmc_samples_2[0])
    for i_2 in range(nchains):
        ax_12[0].plot(co2_by_month.index, trend_posterior_2[-100:, i_2, :].numpy().T, alpha=0.05)
        ax_12[1].plot(co2_by_month.index, seasonality_posterior_2[-100:, i_2, :].numpy().T, alpha=0.05)
    return


@app.cell
def _(
    co2_by_month,
    generate_gam_ar_likelihood,
    mcmc_samples_2,
    np,
    plot_co2_data,
    plt,
):
    gam_with_ar_likelihood_full = generate_gam_ar_likelihood(training=False)
    _, values = gam_with_ar_likelihood_full.sample_distributions(value=mcmc_samples_2)
    fitted_with_forecast_1 = values[-1].numpy()
    fig_8, ax_13 = plt.subplots(1, 1, figsize=(10, 5))
    ax_13.plot(co2_by_month.index, fitted_with_forecast_1[-100:, 0, :].T, color='gray', alpha=0.1)
    ax_13.plot(co2_by_month.index, fitted_with_forecast_1[-100:, 1, :].T, color='gray', alpha=0.1)
    plot_co2_data((fig_8, ax_13))
    average_forecast_1 = np.mean(fitted_with_forecast_1, axis=(0, 1)).T
    ax_13.plot(co2_by_month.index, average_forecast_1, ls='--', label='GAM forecast', alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.11
        """
    )
    return


@app.cell
def _(az, gam_ar_likelihood_idata, gam_idata, plt):
    fig_9, axes_4 = plt.subplots(1, 3, figsize=(4 * 3, 4))
    az.plot_posterior(gam_idata, var_names=['noise_sigma'], alpha=0.5, lw=2.5, ax=axes_4[0])
    axes_4[0].set_title('$\\sigma_{noise}$ (Normal)')
    az.plot_posterior(gam_ar_likelihood_idata, var_names=['noise_sigma', 'rho'], alpha=0.5, lw=2.5, ax=axes_4[1:])
    axes_4[1].set_title('$\\sigma_{noise}$ (AR(1))')
    axes_4[2].set_title('$\\rho$')
    plt.savefig('img/chp06/fig11_ar1_likelihood_rho.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.14
        """
    )
    return


@app.cell
def _(
    co2_by_month_training_data,
    gam_trend_seasonality,
    plt,
    root_1,
    tf,
    tfd_1,
):
    def generate_gam_ar_latent(training=True):

        @tfd_1.JointDistributionCoroutine
        def gam_with_latent_ar():
            seasonality, trend, noise_sigma = (yield from gam_trend_seasonality())
            ar_sigma = (yield root_1(tfd_1.HalfNormal(0.1, name='ar_sigma')))
            rho = (yield root_1(tfd_1.Uniform(-1.0, 1.0, name='rho')))

            def ar_fun(y):
                loc = tf.concat([tf.zeros_like(y[..., :1]), y[..., :-1]], axis=-1) * rho[..., None]
                return tfd_1.Independent(tfd_1.Normal(loc=loc, scale=ar_sigma[..., None]), reinterpreted_batch_ndims=1)
            temporal_error = (yield tfd_1.Autoregressive(distribution_fn=ar_fun, sample0=tf.zeros_like(trend), num_steps=trend.shape[-1], name='temporal_error'))
            y_hat = seasonality + trend + temporal_error
            if training:
                y_hat = y_hat[..., :co2_by_month_training_data.shape[0]]
            observed = (yield tfd_1.Independent(tfd_1.Normal(y_hat, noise_sigma[..., None]), reinterpreted_batch_ndims=1, name='observed'))
        return gam_with_latent_ar
    gam_with_latent_ar = generate_gam_ar_latent()
    plt.plot(tf.transpose(gam_with_latent_ar.sample(50)[-1]))
    return gam_with_latent_ar, generate_gam_ar_latent


@app.cell
def _(co2_by_month_training_data, gam_with_latent_ar, run_mcmc, tf):
    mcmc_samples_3, sampler_stats_3 = run_mcmc(1000, gam_with_latent_ar, n_chains=4, num_adaptation_steps=1000, seed=tf.constant([36245, 734565], dtype=tf.int32), observed=co2_by_month_training_data.T)
    return mcmc_samples_3, sampler_stats_3


@app.cell
def _(az, mcmc_samples_3, np, sampler_stats_3):
    nuts_trace_ar_latent = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_3._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_3[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    axes_5 = az.plot_trace(nuts_trace_ar_latent, var_names=['beta', 'tau', 'ar_sigma', 'rho', 'noise_sigma'], compact=True)
    return (nuts_trace_ar_latent,)


@app.cell
def _(generate_gam_ar_latent, mcmc_samples_3):
    gam_with_latent_ar_full = generate_gam_ar_latent(training=False)
    posterior_dists_1, ppc_samples = gam_with_latent_ar_full.sample_distributions(value=mcmc_samples_3)
    return (ppc_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.12
        """
    )
    return


@app.cell
def _(A_1, X_pred, co2_by_month, mcmc_samples_3, nchains, plt, s_1, t_1, tf):
    fig_10, ax_14 = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    beta, k_3, m_3, tau_2, delta_3 = mcmc_samples_3[:5]
    growth_rate_2 = k_3[..., None] + tf.einsum('ij,...j->...i', A_1, delta_3)
    offset_3 = m_3[..., None] + tf.einsum('ij,...j->...i', A_1, -s_1 * delta_3)
    trend_posterior_3 = growth_rate_2 * t_1 + offset_3
    seasonality_posterior_3 = tf.einsum('ij,...j->...i', X_pred, beta)
    temporal_error = mcmc_samples_3[-1]
    for i_3 in range(nchains):
        ax_14[0].plot(co2_by_month.index, trend_posterior_3[-100:, i_3, :].numpy().T, alpha=0.05)
        ax_14[1].plot(co2_by_month.index, seasonality_posterior_3[-100:, i_3, :].numpy().T, alpha=0.05)
        ax_14[2].plot(co2_by_month.index, temporal_error[-100:, i_3, :].numpy().T, alpha=0.05)
    ax_14[0].set_title('Trend (Step Linear)')
    ax_14[1].set_title('Seasonality (Month of the year effect)')
    ax_14[2].set_title('AR(1)')
    ax_14[2].set_xlabel('Year')
    fig_10.autofmt_xdate()
    plt.savefig('img/chp06/fig12_posterior_predictive_ar1.png')
    return


@app.cell
def _(co2_by_month, np, plot_co2_data, plt, ppc_samples):
    fig_11, ax_15 = plt.subplots(1, 1, figsize=(10, 5))
    fitted_with_forecast_2 = ppc_samples[-1].numpy()
    ax_15.plot(co2_by_month.index, fitted_with_forecast_2[-100:, 0, :].T, color='gray', alpha=0.1)
    ax_15.plot(co2_by_month.index, fitted_with_forecast_2[-100:, 1, :].T, color='gray', alpha=0.1)
    plot_co2_data((fig_11, ax_15))
    average_forecast_2 = np.mean(fitted_with_forecast_2, axis=(0, 1)).T
    ax_15.plot(co2_by_month.index, average_forecast_2, ls='--', label='GAM forecast', alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.13
        """
    )
    return


@app.cell
def _(az, nuts_trace_ar_latent, plt):
    axes_6 = az.plot_posterior(nuts_trace_ar_latent, var_names=['noise_sigma', 'ar_sigma', 'rho'], alpha=0.5, lw=2.5, figsize=(4 * 3, 4))
    axes_6[0].set_title('$\\sigma_{noise}$')
    axes_6[1].set_title('$\\sigma_{AR}$')
    axes_6[2].set_title('$\\rho$')
    plt.savefig('img/chp06/fig13_ar1_likelihood_rho2.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Reflection on autoregressive and smoothing
        """
    )
    return


@app.cell
def _(np, plt):
    num_steps = 100

    x = np.linspace(0, 50, num_steps)
    f = np.exp(1.0 + np.power(x, 0.5) - np.exp(x/15.0))
    y = f + np.random.normal(scale=1.0, size=x.shape)

    plt.plot(x, y, 'ok', label='Observed')
    plt.plot(x, f, 'r', label='f(x)')
    plt.legend()
    plt.xlabel('x');
    plt.ylabel('y');
    return f, num_steps, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.15
        """
    )
    return


@app.cell
def _(num_steps, root_1, tf, tfd_1):
    @tfd_1.JointDistributionCoroutine
    def smoothing_grw():
        alpha = (yield root_1(tfd_1.Beta(5, 1.0)))
        variance = (yield root_1(tfd_1.HalfNormal(10.0)))
        sigma0 = tf.sqrt(variance * alpha)
        sigma1 = tf.sqrt(variance * (1.0 - alpha))
        z = (yield tfd_1.Sample(tfd_1.Normal(0.0, sigma0), num_steps))
        observed = (yield tfd_1.Independent(tfd_1.Normal(tf.math.cumsum(z, axis=-1), sigma1[..., None]), name='observed'))
    return (smoothing_grw,)


@app.cell
def _(run_mcmc, smoothing_grw, tf, y):
    mcmc_samples_4, sampler_stats_4 = run_mcmc(1000, smoothing_grw, n_chains=4, num_adaptation_steps=1000, observed=tf.constant(y[None, ...], dtype=tf.float32))
    return (mcmc_samples_4,)


@app.cell
def _(mcmc_samples_4, plt):
    _, ax_16 = plt.subplots(2, 1, figsize=(10, 5))
    ax_16[0].plot(mcmc_samples_4[0], alpha=0.5)
    ax_16[1].plot(mcmc_samples_4[1], alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.14
        """
    )
    return


@app.cell
def _(f, mcmc_samples_4, np, plt, tf, x, y):
    nsample, nchain = mcmc_samples_4[-1].shape[:2]
    z = tf.reshape(tf.math.cumsum(mcmc_samples_4[-1], axis=-1), [nsample * nchain, -1])
    lower, upper = np.percentile(z, [5, 95], axis=0)
    _, ax_17 = plt.subplots(figsize=(10, 4))
    ax_17.plot(x, y, 'o', label='Observed')
    ax_17.plot(x, f, label='f(x)')
    ax_17.fill_between(x, lower, upper, color='C1', alpha=0.25)
    ax_17.plot(x, tf.reduce_mean(z, axis=0), color='C4', ls='--', label='z')
    ax_17.legend()
    ax_17.set_xlabel('x')
    ax_17.set_ylabel('y')
    plt.savefig('img/chp06/fig14_smoothing_with_gw.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### SARIMA
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.16 and Figure 6.15
        """
    )
    return


@app.cell
def _(pd, plt):
    us_monthly_birth = pd.read_csv("../data/monthly_birth_usa.csv")
    us_monthly_birth["date_month"] = pd.to_datetime(us_monthly_birth["date_month"])
    us_monthly_birth.set_index("date_month", drop=True, inplace=True)

    def plot_birth_data(fig_ax=None):
        if not fig_ax:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        else:
            fig, ax = fig_ax
    
        ax.plot(us_monthly_birth, lw=2)
        ax.set_ylabel("Birth (thousands)")
        ax.set_xlabel("Year")
        fig.suptitle("Monthly live birth U.S.A",
                     fontsize=15)
        ax.text(0.99, .02,
                "Source: Stoffer D (2019). “astsa: Applied Statistical Time Series Analysis.”",
                transform=ax.transAxes,
                horizontalalignment="right",
                alpha=0.5)
        fig.autofmt_xdate()
        return fig, ax


    _ = plot_birth_data()

    plt.savefig("img/chp06/fig15_birth_by_month.png")
    return (us_monthly_birth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### y ~ Sarima(1,1,1)(1,1,1)[12]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.17
        """
    )
    return


@app.cell
def _(np, tf, tfd_1, us_monthly_birth):
    p_1, d, q = (1, 1, 1)
    P, D, Q, period = (1, 1, 1, 12)
    observed = us_monthly_birth['birth_in_thousands'].values
    for _ in range(D):
        observed = observed[period:] - observed[:-period]
    observed = tf.constant(np.diff(observed, n=d), tf.float32)
    r = max(p_1, q, P * period, Q * period)

    def likelihood(mu0, sigma, phi, theta, sphi, stheta):
        batch_shape = tf.shape(mu0)
        y_extended = tf.concat([tf.zeros(tf.concat([[r], batch_shape], axis=0), dtype=mu0.dtype), tf.einsum('...,j->j...', tf.ones_like(mu0, dtype=observed.dtype), observed)], axis=0)
        eps_t = tf.zeros_like(y_extended, dtype=observed.dtype)

        def arma_onestep(t, eps_t):
            t_shift = t + r
            y_past = tf.gather(y_extended, t_shift - (np.arange(p_1) + 1))
            ar = tf.einsum('...p,p...->...', phi, y_past)
            eps_past = tf.gather(eps_t, t_shift - (np.arange(q) + 1))
            ma = tf.einsum('...q,q...->...', theta, eps_past)
            sy_past = tf.gather(y_extended, t_shift - (np.arange(P) + 1) * period)
            sar = tf.einsum('...p,p...->...', sphi, sy_past)
            seps_past = tf.gather(eps_t, t_shift - (np.arange(Q) + 1) * period)
            sma = tf.einsum('...q,q...->...', stheta, seps_past)
            mu_at_t = ar + ma + sar + sma + mu0
            eps_update = tf.gather(y_extended, t_shift) - mu_at_t
            epsilon_t_next = tf.tensor_scatter_nd_update(eps_t, [[t_shift]], eps_update[None, ...])
            return (t + 1, epsilon_t_next)
        t, eps_output_ = tf.while_loop(lambda t, *_: t < observed.shape[-1], arma_onestep, loop_vars=(0, eps_t), maximum_iterations=observed.shape[-1])
        eps_output = eps_output_[r:]
        return tf.reduce_sum(tfd_1.Normal(0, sigma[None, ...]).log_prob(eps_output), axis=0)
    return P, Q, likelihood, observed, p_1, q


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.18
        """
    )
    return


@app.cell
def _(P, Q, p_1, q, root_1, tfb, tfd_1):
    @tfd_1.JointDistributionCoroutine
    def sarima_priors():
        mu0 = (yield root_1(tfd_1.StudentT(df=6, loc=0, scale=2.5, name='mu0')))
        sigma = (yield root_1(tfd_1.HalfStudentT(df=7, loc=0, scale=1.0, name='sigma')))
        phi = (yield root_1(tfd_1.Sample(tfd_1.TransformedDistribution(tfd_1.Beta(concentration1=2.0, concentration0=2.0), tfb.Shift(-1.0)(tfb.Scale(2.0))), p_1, name='phi')))
        theta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), q, name='theta')))
        sphi = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), P, name='sphi')))
        stheta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), Q, name='stheta')))
    return (sarima_priors,)


@app.cell
def _(tf, tfd_1, tfp):
    from tensorflow_probability.python.internal import unnest
    from tensorflow_probability.python.internal import samplers

    def run_mcmc_simple(n_draws, joint_dist, n_chains=4, num_adaptation_steps=1000, return_compiled_function=False, target_log_prob_fn=None, bijector=None, init_state=None, seed=None, **pins):
        joint_dist_pinned = joint_dist.experimental_pin(**pins) if pins else joint_dist
        if bijector is None:
            bijector = joint_dist_pinned.experimental_default_event_space_bijector()
        if target_log_prob_fn is None:
            target_log_prob_fn = joint_dist_pinned.unnormalized_log_prob
        if seed is None:
            seed = 26401
        run_mcmc_seed = samplers.sanitize_seed(seed, salt='run_mcmc_seed')
        if init_state is None:
            if pins:
                init_state_ = joint_dist_pinned.sample_unpinned(n_chains)
            else:
                init_state_ = joint_dist_pinned.sample(n_chains)
            ini_state_unbound = bijector.inverse(init_state_)
            run_mcmc_seed, *init_seed = samplers.split_seed(run_mcmc_seed, n=len(ini_state_unbound) + 1)
            init_state = bijector.forward(tf.nest.map_structure(lambda x, seed: tfd_1.Uniform(-1.0, tf.constant(1.0, x.dtype)).sample(x.shape, seed=seed), ini_state_unbound, tf.nest.pack_sequence_as(ini_state_unbound, init_seed)))

        @tf.function(autograph=False, jit_compile=True)
        def run_inference_nuts(init_state, draws, tune, seed):
            seed, tuning_seed, sample_seed = samplers.split_seed(seed, n=3)

            def gen_kernel(step_size):
                hmc = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=target_log_prob_fn, step_size=step_size)
                hmc = tfp.mcmc.TransformedTransitionKernel(hmc, bijector=bijector)
                tuning_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(hmc, tune // 2, target_accept_prob=0.85)
                return tuning_hmc

            def tuning_trace_fn(_, pkr):
                return (pkr.inner_results.transformed_state, pkr.new_step_size)

            def get_tuned_stepsize(samples, step_size):
                return tf.math.reduce_std(samples, axis=0) * step_size[-1]
            step_size = tf.nest.map_structure(tf.ones_like, bijector.inverse(init_state))
            tuning_hmc = gen_kernel(step_size)
            init_samples, (sample_unbounded, tuning_step_size) = tfp.mcmc.sample_chain(num_results=200, num_burnin_steps=tune // 2 - 200, current_state=init_state, kernel=tuning_hmc, trace_fn=tuning_trace_fn, seed=tuning_seed)
            tuning_step_size = tf.nest.pack_sequence_as(sample_unbounded, tuning_step_size)
            step_size_new = tf.nest.map_structure(get_tuned_stepsize, sample_unbounded, tuning_step_size)
            sample_hmc = gen_kernel(step_size_new)

            def sample_trace_fn(_, pkr):
                energy_diff = unnest.get_innermost(pkr, 'log_accept_ratio')
                return {'target_log_prob': unnest.get_innermost(pkr, 'target_log_prob'), 'n_steps': unnest.get_innermost(pkr, 'leapfrogs_taken'), 'diverging': unnest.get_innermost(pkr, 'has_divergence'), 'energy': unnest.get_innermost(pkr, 'energy'), 'accept_ratio': tf.minimum(1.0, tf.exp(energy_diff)), 'reach_max_depth': unnest.get_innermost(pkr, 'reach_max_depth')}
            current_state = tf.nest.map_structure(lambda x: x[-1], init_samples)
            return tfp.mcmc.sample_chain(num_results=draws, num_burnin_steps=tune // 2, current_state=current_state, kernel=sample_hmc, trace_fn=sample_trace_fn, seed=sample_seed)
        mcmc_samples, mcmc_diagnostic = run_inference_nuts(init_state, n_draws, num_adaptation_steps, run_mcmc_seed)
        if return_compiled_function:
            return (mcmc_samples, mcmc_diagnostic, run_inference_nuts)
        else:
            return (mcmc_samples, mcmc_diagnostic)
    return (run_mcmc_simple,)


@app.cell
def _(likelihood, run_mcmc_simple, sarima_priors, tf):
    target_log_prob_fn = lambda *x: sarima_priors.log_prob(*x) + likelihood(*x)
    mcmc_samples_5, sampler_stats_5 = run_mcmc_simple(1000, sarima_priors, n_chains=4, num_adaptation_steps=1000, target_log_prob_fn=target_log_prob_fn, seed=tf.constant([623453, 456345], dtype=tf.int32))
    return mcmc_samples_5, sampler_stats_5


@app.cell
def _(az, mcmc_samples_5, np, sampler_stats_5):
    nuts_trace_arima = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_5._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_5[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    axes_7 = az.plot_trace(nuts_trace_arima)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### SARIMAX Class
        """
    )
    return


@app.cell
def _(np, tf, tfd_1):
    import warnings
    from statsmodels.tsa.statespace.tools import diff as tsa_diff
    from tensorflow_probability.python.internal import distribution_util
    from tensorflow_probability.python.internal import prefer_static as ps

    class SARIMAX:

        def __init__(self, observed, design_matrix=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), dtype=np.float32):
            """
            SARIMAX Likelihood for univariate time series
        
            order - (p,d,q)
            seasonal_order - (P,D,Q,s)
            """
            self.p, self.d, self.q = order
            self.P, self.D, self.Q, self.period = seasonal_order
            if design_matrix is not None:
                assert ps.rank(design_matrix) >= 2
                assert ps.shape(design_matrix)[-2] == observed.shape[-1]
                if self.period > 0:
                    warnings.warn('\n                Can not model seasonal difference with Dynamic regressions,\n                Setting D to 0 ...\n                ')
                    self.period = 0
                self.design_matrix = tf.convert_to_tensor(np.diff(design_matrix, n=self.d, axis=0), dtype=dtype)
            else:
                self.design_matrix = None
            if self.period <= 1:
                self.P, self.D, self.Q = (0, 0, 0)
            self.r = max(self.p, self.q, self.P * self.period, self.Q * self.period)
            observed_diff = tsa_diff(observed, k_diff=self.d, k_seasonal_diff=self.D, seasonal_periods=self.period)
            self.observed = tf.convert_to_tensor(observed_diff, dtype=dtype)
            self.dtype = dtype

        def _log_prob(self, *args):
            mu0 = args[0]
            sigma = args[1]
            i = 2
            if self.design_matrix is not None:
                reg_coeff = args[i]
                mu_t = mu0[None, ...] + tf.einsum('...i,ji->j...', reg_coeff, self.design_matrix)
                i = i + 1
            else:
                mu_t = tf.einsum('...,j->j...', mu0, ps.ones_like(self.observed))
            if self.p > 0:
                phi = args[i]
                i = i + 1
            if self.q > 0:
                theta = args[i]
                i = i + 1
            if self.P > 0:
                sphi = args[i]
                i = i + 1
            if self.Q > 0:
                stheta = args[i]
                i = i + 1
            batch_shape = ps.shape(mu0)
            y_extended = ps.concat([ps.zeros(tf.concat([[self.r], batch_shape], axis=0), dtype=mu0.dtype), tf.einsum('...,j->j...', ps.ones_like(mu0, dtype=self.observed.dtype), self.observed)], axis=0)
            eps_t = ps.zeros_like(y_extended, dtype=self.observed.dtype)

            def body_fn(t, mu_t, eps_t):
                mu_temp = []
                t_switch = t + self.r
                if self.p > 0:
                    y_past = tf.gather(y_extended, t_switch - (np.arange(self.p) + 1))
                    ar = tf.einsum('...p,p...->...', phi, y_past)
                    mu_temp.append(ar)
                if self.q > 0:
                    eps_past = tf.gather(eps_t, t_switch - (np.arange(self.q) + 1))
                    ma = tf.einsum('...q,q...->...', theta, eps_past)
                    mu_temp.append(ma)
                if self.P > 0:
                    y_past = tf.gather(y_extended, t_switch - (np.arange(self.P) + 1) * self.period)
                    sar = tf.einsum('...p,p...->...', sphi, y_past)
                    mu_temp.append(sar)
                if self.Q > 0:
                    eps_past = tf.gather(eps_t, t_switch - (np.arange(self.Q) + 1) * self.period)
                    sma = tf.einsum('...q,q...->...', stheta, eps_past)
                    mu_temp.append(sma)
                mu_update = sum(mu_temp) + tf.gather(mu_t, t)
                mu_t_next = tf.tensor_scatter_nd_update(mu_t, [[t]], mu_update[None, ...])
                eps_update = tf.gather(y_extended, t_switch) - mu_update
                epsilon_t_next = tf.tensor_scatter_nd_update(eps_t, [[t_switch]], eps_update[None, ...])
                return (t + 1, mu_t_next, epsilon_t_next)
            t, mu_output, eps_output_ = tf.while_loop(lambda t, *_: t < self.observed.shape[-1], body_fn, loop_vars=(0, mu_t, eps_t), maximum_iterations=self.observed.shape[-1])
            eps_output = eps_output_[self.r:]
            return (tfd_1.Normal(0, sigma[None, ...]).log_prob(eps_output), mu_output)

        def log_prob(self, *args):
            log_prob_val, _ = self._log_prob(*args)
            return ps.reduce_sum(log_prob_val, axis=0)

        def log_prob_elementwise(self, *args):
            sigma = args[1]
            _, mu_output = self._log_prob(*args)
            mu = distribution_util.move_dimension(mu_output, 0, -1)
            return tfd_1.Normal(mu, sigma[..., None]).log_prob(self.observed)
    return (SARIMAX,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### y ~ Sarima(1,1,1)(1,1,1)[12]
        """
    )
    return


@app.cell
def _(SARIMAX, root_1, tfd_1, us_monthly_birth):
    p_2, q_1, P_1, Q_1 = (1, 1, 1, 1)

    @tfd_1.JointDistributionCoroutine
    def sarima_priors_1():
        mu0 = (yield root_1(tfd_1.StudentT(df=6, loc=0, scale=2.5, name='mu0')))
        sigma = (yield root_1(tfd_1.HalfStudentT(df=7, loc=0, scale=1.0, name='sigma')))
        phi = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), p_2, name='phi')))
        theta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), q_1, name='theta')))
        sphi = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), P_1, name='sphi')))
        stheta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), Q_1, name='stheta')))
    sarima_1 = SARIMAX(us_monthly_birth['birth_in_thousands'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    return sarima_1, sarima_priors_1


@app.cell
def _(run_mcmc_simple, sarima_1, sarima_priors_1, tf):
    target_log_prob_fn_1 = lambda *x: sarima_priors_1.log_prob(*x) + sarima_1.log_prob(*x)
    mcmc_samples_6, sampler_stats_6 = run_mcmc_simple(1000, sarima_priors_1, n_chains=4, num_adaptation_steps=1000, target_log_prob_fn=target_log_prob_fn_1, seed=tf.constant([623453, 456345], dtype=tf.int32))
    return mcmc_samples_6, sampler_stats_6


@app.cell
def _(az, mcmc_samples_6, np, sampler_stats_6, sarima_1):
    data_likelihood = np.swapaxes(sarima_1.log_prob_elementwise(*mcmc_samples_6), 1, 0)
    sarima_0_idata = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_6._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_6[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']}, observed_data={'observed': sarima_1.observed}, log_likelihood={'observed': data_likelihood})
    axes_8 = az.plot_trace(sarima_0_idata)
    return (sarima_0_idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.8
        """
    )
    return


@app.cell
def _(SARIMAX, np, root_1, tfd_1, us_monthly_birth):
    def gen_fourier_basis_1(t, p=365.25, n=3):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)
    p_3 = 12
    t_monthly_1 = np.asarray([i % p_3 for i in range(len(us_monthly_birth))]) + 1
    monthly_X_1 = gen_fourier_basis_1(t_monthly_1, p=p_3, n=2)

    @tfd_1.JointDistributionCoroutine
    def sarima_priors_2():
        mu0 = (yield root_1(tfd_1.StudentT(df=6, loc=0, scale=2.5, name='mu0')))
        sigma = (yield root_1(tfd_1.HalfStudentT(df=7, loc=0, scale=1.0, name='sigma')))
        beta = (yield root_1(tfd_1.Sample(tfd_1.StudentT(df=6, loc=0, scale=2.5), monthly_X_1.shape[-1], name='beta')))
        phi = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), 1, name='phi')))
        theta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), 1, name='theta')))
    sarima_2 = SARIMAX(us_monthly_birth['birth_in_thousands'], design_matrix=monthly_X_1, order=(1, 1, 1))
    return sarima_2, sarima_priors_2


@app.cell
def _(run_mcmc_simple, sarima_2, sarima_priors_2, tf):
    target_log_prob_fn_2 = lambda *x: sarima_priors_2.log_prob(*x) + sarima_2.log_prob(*x)
    mcmc_samples_7, sampler_stats_7 = run_mcmc_simple(1000, sarima_priors_2, n_chains=4, num_adaptation_steps=1000, target_log_prob_fn=target_log_prob_fn_2, seed=tf.constant([623453, 456345], dtype=tf.int32))
    return mcmc_samples_7, sampler_stats_7


@app.cell
def _(az, mcmc_samples_7, np, sampler_stats_7, sarima_2):
    data_likelihood_1 = np.swapaxes(sarima_2.log_prob_elementwise(*mcmc_samples_7), 1, 0)
    arimax_idata = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_7._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_7[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']}, observed_data={'observed': sarima_2.observed}, log_likelihood={'observed': data_likelihood_1})
    axes_9 = az.plot_trace(arimax_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### y ~ Sarima(0,1,2)(1,1,1)[12]
        """
    )
    return


@app.cell
def _(SARIMAX, root_1, tfd_1, us_monthly_birth):
    @tfd_1.JointDistributionCoroutine
    def sarima_priors_3():
        mu0 = (yield root_1(tfd_1.StudentT(df=6, loc=0, scale=2.5, name='mu0')))
        sigma = (yield root_1(tfd_1.HalfStudentT(df=7, loc=0, scale=1.0, name='sigma')))
        theta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), 2, name='theta')))
        sphi = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), 1, name='sphi')))
        stheta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), 1, name='stheta')))
    sarima_3 = SARIMAX(us_monthly_birth['birth_in_thousands'], order=(0, 1, 2), seasonal_order=(1, 1, 1, 12))
    return sarima_3, sarima_priors_3


@app.cell
def _(run_mcmc_simple, sarima_3, sarima_priors_3, tf):
    target_log_prob_fn_3 = lambda *x: sarima_priors_3.log_prob(*x) + sarima_3.log_prob(*x)
    mcmc_samples_8, sampler_stats_8 = run_mcmc_simple(1000, sarima_priors_3, n_chains=4, num_adaptation_steps=1000, target_log_prob_fn=target_log_prob_fn_3, seed=tf.constant([934563, 12356], dtype=tf.int32))
    return mcmc_samples_8, sampler_stats_8


@app.cell
def _(az, mcmc_samples_8, np, sampler_stats_8, sarima_2, sarima_3):
    data_likelihood_2 = np.swapaxes(sarima_3.log_prob_elementwise(*mcmc_samples_8), 1, 0)
    sarima_1_idata = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_8._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_8[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']}, observed_data={'observed': sarima_2.observed}, log_likelihood={'observed': data_likelihood_2})
    axes_10 = az.plot_trace(sarima_1_idata)
    return (sarima_1_idata,)


@app.cell
def _(az, sarima_1_idata):
    az.summary(sarima_1_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 6.1
        """
    )
    return


@app.cell
def _(az, sarima_0_idata, sarima_1_idata):
    compare_dict = {"SARIMA(1,1,1)(1,1,1)[12]": sarima_0_idata,
    #                 "ARIMAX(1,1,1)X[4]": arimax_idata,
                    "SARIMA(0,1,2)(1,1,1)[12]": sarima_1_idata}
    cmp = az.compare(compare_dict, ic='loo')
    cmp.round(2)
    return


@app.cell
def _():
    # print(cmp.round(2).to_latex())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## State Space Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.19
        """
    )
    return


@app.cell
def _(tf, tfd_1):
    theta0, theta1 = (1.2, 2.6)
    sigma_2 = 0.4
    num_timesteps = 100
    time_stamp = tf.linspace(0.0, 1.0, num_timesteps)[..., None]
    yhat = theta0 + theta1 * time_stamp
    y_1 = tfd_1.Normal(yhat, sigma_2).sample()
    return num_timesteps, sigma_2, time_stamp, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.20
        """
    )
    return


@app.cell
def _(num_timesteps, sigma_2, tf, tfd_1, time_stamp):
    initial_state_prior = tfd_1.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[5.0, 5.0])
    transition_matrix = lambda _: tf.linalg.LinearOperatorIdentity(2)
    transition_noise = lambda _: tfd_1.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[0.0, 0.0])
    H = tf.concat([tf.ones_like(time_stamp), time_stamp], axis=-1)
    observation_matrix = lambda t: tf.linalg.LinearOperatorFullMatrix([tf.gather(H, t)])
    observation_noise = lambda _: tfd_1.MultivariateNormalDiag(loc=[0.0], scale_diag=[sigma_2])
    linear_growth_model = tfd_1.LinearGaussianStateSpaceModel(num_timesteps=num_timesteps, transition_matrix=transition_matrix, transition_noise=transition_noise, observation_matrix=observation_matrix, observation_noise=observation_noise, initial_state_prior=initial_state_prior)
    return H, initial_state_prior, linear_growth_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.21
        """
    )
    return


@app.cell
def _(linear_growth_model, y_1):
    log_likelihoods, mt_filtered, Pt_filtered, mt_predicted, Pt_predicted, observation_means, observation_cov = linear_growth_model.forward_filter(y_1)
    return Pt_filtered, mt_filtered, observation_means


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.16
        """
    )
    return


@app.cell
def _(
    H,
    Pt_filtered,
    initial_state_prior,
    mt_filtered,
    observation_means,
    plt,
    sigma_2,
    tf,
    time_stamp,
    y_1,
):
    m0 = initial_state_prior.mean()
    P0 = initial_state_prior.covariance()
    P0_inv = tf.linalg.inv(P0)
    P_t = tf.linalg.inv(P0_inv + 1 / sigma_2 ** 2 * tf.matmul(H, H, transpose_a=True))
    m_t = tf.matmul(P_t, 1 / sigma_2 ** 2 * tf.matmul(H, y_1, transpose_a=True) + tf.matmul(P0_inv, m0[..., None]))
    filtered_vars = tf.linalg.diag_part(Pt_filtered)
    _, ax_18 = plt.subplots(1, 3, figsize=(10, 4))
    ax_18[0].plot(time_stamp, y_1, '--o', alpha=0.5)
    ax_18[0].plot(time_stamp, observation_means, lw=1.5, color='k')
    ax_18[0].set_title('Observed time series')
    ax_18[0].legend(['Observed', 'Predicted'])
    ax_18[0].set_xlabel('time')
    color = ['C4', 'C1']
    for i_4 in range(2):
        ax_18[1].plot(time_stamp, tf.transpose(mt_filtered[..., i_4]), color=color[i_4])
        ax_18[2].semilogy(time_stamp, tf.transpose(filtered_vars[..., i_4]), color=color[i_4])
    for i_4 in range(2):
        ax_18[i_4 + 1].set_xlabel('time')
        ax_18[i_4 + 1].legend(['theta0', 'theta1'])
    ax_18[1].hlines(m_t, time_stamp[0], time_stamp[-1], ls='--')
    ax_18[1].set_title('$m_{t \\mid t}$')
    ax_18[2].hlines(tf.linalg.diag_part(P_t), time_stamp[0], time_stamp[-1], ls='--')
    ax_18[2].set_title('diag($P_{t \\mid t}$)')
    ax_18[2].grid()
    plt.tight_layout()
    plt.savefig('img/chp06/fig16_linear_growth_lgssm.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### ARMA as LGSSM
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.22
        """
    )
    return


@app.cell
def _(tf, tfd_1):
    num_timesteps_1 = 300
    phi1 = -0.1
    phi2 = 0.5
    theta1_1 = -0.25
    sigma_3 = 1.25
    initial_state_prior_1 = tfd_1.MultivariateNormalDiag(scale_diag=[sigma_3, sigma_3])
    transition_matrix_1 = lambda _: tf.linalg.LinearOperatorFullMatrix([[phi1, 1], [phi2, 0]])
    R_t = tf.constant([[sigma_3], [sigma_3 * theta1_1]])
    Q_t_tril = tf.concat([R_t, tf.zeros_like(R_t)], axis=-1)
    transition_noise_1 = lambda _: tfd_1.MultivariateNormalTriL(scale_tril=Q_t_tril)
    observation_matrix_1 = lambda t: tf.linalg.LinearOperatorFullMatrix([[1.0, 0.0]])
    observation_noise_1 = lambda _: tfd_1.MultivariateNormalDiag(loc=[0.0], scale_diag=[0.0])
    arma = tfd_1.LinearGaussianStateSpaceModel(num_timesteps=num_timesteps_1, transition_matrix=transition_matrix_1, transition_noise=transition_noise_1, observation_matrix=observation_matrix_1, observation_noise=observation_noise_1, initial_state_prior=initial_state_prior_1)
    sim_ts = arma.sample()
    return (
        Q_t_tril,
        arma,
        num_timesteps_1,
        phi1,
        phi2,
        sigma_3,
        sim_ts,
        theta1_1,
    )


@app.cell
def _(Q_t_tril, np, tf):
    np.linalg.eigvals(Q_t_tril @ tf.transpose(Q_t_tril)) >= 0
    return


@app.cell
def _(plt, sim_ts):
    plt.plot(sim_ts);
    return


@app.cell
def _(arma, sim_ts):
    arma.log_prob(sim_ts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.23
        """
    )
    return


@app.cell
def _(num_timesteps_1, root_1, tf, tfd_1):
    @tfd_1.JointDistributionCoroutine
    def arma_lgssm():
        sigma = (yield root_1(tfd_1.HalfStudentT(df=7, loc=0, scale=1.0, name='sigma')))
        phi = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), 2, name='phi')))
        theta = (yield root_1(tfd_1.Sample(tfd_1.Normal(loc=0, scale=0.5), 1, name='theta')))
        init_scale_diag = tf.concat([sigma[..., None], sigma[..., None]], axis=-1)
        initial_state_prior = tfd_1.MultivariateNormalDiag(scale_diag=init_scale_diag)
        F_t = tf.concat([phi[..., None], tf.concat([tf.ones_like(phi[..., 0, None]), tf.zeros_like(phi[..., 0, None])], axis=-1)[..., None]], axis=-1)

        def transition_matrix(_):
            return tf.linalg.LinearOperatorFullMatrix(F_t)
        transition_scale_tril = tf.concat([sigma[..., None], theta * sigma[..., None]], axis=-1)[..., None]
        scale_tril = tf.concat([transition_scale_tril, tf.zeros_like(transition_scale_tril)], axis=-1)

        def transition_noise(_):
            return tfd_1.MultivariateNormalTriL(scale_tril=scale_tril)

        def observation_matrix(t):
            return tf.linalg.LinearOperatorFullMatrix([[1.0, 0.0]])

        def observation_noise(t):
            return tfd_1.MultivariateNormalDiag(loc=[0], scale_diag=[0.0])
        arma = (yield tfd_1.LinearGaussianStateSpaceModel(num_timesteps=num_timesteps_1, transition_matrix=transition_matrix, transition_noise=transition_noise, observation_matrix=observation_matrix, observation_noise=observation_noise, initial_state_prior=initial_state_prior, name='arma'))
    return (arma_lgssm,)


@app.cell
def _(arma_lgssm, run_mcmc_simple, sim_ts, tf):
    mcmc_samples_9, sampler_stats_9 = run_mcmc_simple(1000, arma_lgssm, n_chains=4, num_adaptation_steps=1000, seed=tf.constant([23453, 94567], dtype=tf.int32), arma=sim_ts)
    return mcmc_samples_9, sampler_stats_9


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.17
        """
    )
    return


@app.cell
def _(
    az,
    mcmc_samples_9,
    np,
    phi1,
    phi2,
    plt,
    sampler_stats_9,
    sigma_3,
    theta1_1,
):
    test_trace = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_9._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_9[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    lines_1 = (('sigma', {}, sigma_3), ('phi', {}, [phi1, phi2]), ('theta', {}, theta1_1))
    axes_11 = az.plot_trace(test_trace, lines=lines_1)
    plt.savefig('img/chp06/fig17_arma_lgssm_inference_result.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Bayesian Structural Time Series Models on Monthly Live Birth Data
        """
    )
    return


app._unparsable_cell(
    r"""
    def generate_bsts_model(observed=None):
        \"\"\"
        Args:
        observed: Observed time series, tfp.sts use it to generate data informed prior.
        \"\"\"
        # Trend
        trend = tfp.sts.LocalLinearTrend(observed_time_series=observed)
        # Seasonal
        seasonal = tfp.sts.Seasonal(num_seasons=12, observed_time_series=observed)
        # Full model
        return tfp.sts.Sum([trend, seasonal], observed_time_series=observed)

    observed = tf.constant(us_monthly_birth[\"birth_in_thousands\"], dtype=tf.float32)
    birth_model = generate_bsts_model(observed=observed)

    # Generate the posterior distribution conditioned on the observed
    # target_log_prob_fn = birth_model.joint_log_prob(observed_time_series=observed)

    birth_model_jd = birth_model.joint_distribu2.15.0.post1tion(observed_time_series=observed)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.25
        """
    )
    return


@app.cell
def _(birth_model):
    birth_model.components
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.26
        """
    )
    return


@app.cell
def _(birth_model):
    birth_model.components[1].parameters
    return


@app.cell
def _(birth_model_jd, run_mcmc, tf):
    mcmc_samples_10, sampler_stats_10 = run_mcmc(1000, birth_model_jd, n_chains=4, num_adaptation_steps=1000, seed=tf.constant([745678, 562345], dtype=tf.int32))
    return mcmc_samples_10, sampler_stats_10


@app.cell
def _(az, mcmc_samples_10, np, sampler_stats_10):
    birth_model_idata = az.from_dict(posterior={k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_10.items()}, sample_stats={k: np.swapaxes(sampler_stats_10[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    axes_12 = az.plot_trace(birth_model_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 6.27
        """
    )
    return


@app.cell
def _(birth_model, mcmc_samples_10, observed, tf, tfp):
    parameter_samples = tf.nest.map_structure(lambda x: x[-100:, 0, ...], mcmc_samples_10)
    component_dists = tfp.sts.decompose_by_component(birth_model, observed_time_series=observed, parameter_samples=parameter_samples)
    n_steps = 36
    forecast_dist = tfp.sts.forecast(birth_model, observed_time_series=observed, parameter_samples=parameter_samples, num_steps_forecast=n_steps)
    return component_dists, forecast_dist, n_steps


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Other Time Series Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 6.19
        """
    )
    return


@app.cell
def _(component_dists, forecast_dist, n_steps, np, pd, plt, us_monthly_birth):
    birth_dates = us_monthly_birth.index
    forecast_date = pd.date_range(start=birth_dates[-1] + np.timedelta64(1, 'M'), end=birth_dates[-1] + np.timedelta64(1 + n_steps, 'M'), freq='M')
    fig_12, axes_13 = plt.subplots(1 + len(component_dists.keys()), 1, figsize=(10, 9), sharex=True)
    ax_19 = axes_13[0]
    ax_19.plot(us_monthly_birth, lw=1.5, label='observed')
    forecast_mean = np.squeeze(forecast_dist.mean())
    line = ax_19.plot(forecast_date, forecast_mean, lw=1.5, label='forecast mean', color='C4')
    forecast_std = np.squeeze(forecast_dist.stddev())
    ax_19.fill_between(forecast_date, forecast_mean - 2 * forecast_std, forecast_mean + 2 * forecast_std, color=line[0].get_color(), alpha=0.2)
    for ax__1, (key, dist) in zip(axes_13[1:], component_dists.items()):
        comp_mean, comp_std = (np.squeeze(dist.mean()), np.squeeze(dist.stddev()))
        line = ax__1.plot(birth_dates, dist.mean(), lw=2.0)
        ax__1.fill_between(birth_dates, comp_mean - 2 * comp_std, comp_mean + 2 * comp_std, alpha=0.2)
        ax__1.set_title(key.name[:-1])
    ax_19.legend()
    ax_19.set_ylabel('Birth (thousands)')
    ax__1.set_xlabel('Year')
    ax_19.set_title('Monthly live birth U.S.A', fontsize=15)
    ax_19.text(0.99, 0.02, 'Source: Stoffer D (2019). “astsa: Applied Statistical Time Series Analysis.”', transform=ax_19.transAxes, horizontalalignment='right', alpha=0.5)
    fig_12.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('img/chp06/fig19_bsts_lgssm_result.png')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
