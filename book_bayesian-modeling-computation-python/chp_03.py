import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 3: Linear Models and Probabilistic Programming Languages
        """
    )
    return


@app.cell
def _():
    import pymc as pm
    import matplotlib.pyplot as plt
    import arviz as az
    import xarray as xr
    import pandas as pd
    from scipy import special, stats
    import numpy as np
    return az, np, pd, plt, pm, special, stats, xr


@app.cell
def _(az, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams['figure.dpi'] = 300
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Comparing Two (or More) Groups
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.1
        """
    )
    return


@app.cell
def _(pd):
    penguins = pd.read_csv("../data/penguins.csv")

    # Subset to the columns needed
    missing_data = penguins.isnull()[
        ["bill_length_mm", "flipper_length_mm", "sex", "body_mass_g"]
    ].any(axis=1)

    # Drop rows with any missing data
    penguins = penguins.loc[~missing_data]

    penguins.head()
    return (penguins,)


@app.cell
def _(penguins):
    penguins.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 3.1 and Code 3.2
        """
    )
    return


@app.cell
def _(penguins):
    summary_stats = (penguins.loc[:, ["species", "body_mass_g"]]
                             .groupby("species")
                             .agg(["mean", "std", "count"]))
    summary_stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.3
        """
    )
    return


@app.cell
def _(penguins, pm):
    adelie_mask = penguins['species'] == 'Adelie'
    adelie_mass_obs = penguins.loc[adelie_mask, 'body_mass_g'].values
    with pm.Model() as model_adelie_penguin_mass:
        _σ = pm.HalfStudentT('σ', 100, 2000)
        _μ = pm.Normal('μ', 4000, 3000)
        _mass = pm.Normal('mass', mu=_μ, sigma=_σ, observed=adelie_mass_obs)
        idata_adelie_mass = pm.sample(chains=4)
        idata_adelie_mass.extend(pm.sample_prior_predictive(samples=5000))
    return adelie_mask, adelie_mass_obs, idata_adelie_mass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.1
        """
    )
    return


@app.cell
def _(az, idata_adelie_mass, plt):
    _axes = az.plot_posterior(idata_adelie_mass.prior, var_names=['σ', 'μ'], figsize=(10, 4))
    plt.savefig('img/chp03/single_species_prior_predictive.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.2
        """
    )
    return


@app.cell
def _(az, idata_adelie_mass, plt):
    _axes = az.plot_trace(idata_adelie_mass, divergences='bottom', kind='rank_bars', figsize=(10, 4))
    plt.savefig('img/chp03/single_species_KDE_rankplot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 3.2
        """
    )
    return


@app.cell
def _(az, idata_adelie_mass):
    az.summary(idata_adelie_mass)
    return


@app.cell
def _():
    #print(az.summary(idata_adelie_mass).round(1).to_latex())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.3
        """
    )
    return


@app.cell
def _(az, idata_adelie_mass, plt):
    _axes = az.plot_posterior(idata_adelie_mass, hdi_prob=0.94, figsize=(10, 4))
    _axes[0].axvline(3706, linestyle='--')
    _axes[1].axvline(459, linestyle='--')
    plt.savefig('img/chp03/single_species_mass_posteriorplot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.4
        """
    )
    return


@app.cell
def _(pd, penguins, pm):
    all_species = pd.Categorical(penguins['species'])
    coords = {'species': all_species.categories}
    with pm.Model(coords=coords) as model_penguin_mass_all_species:
        _σ = pm.HalfStudentT('σ', 100, 2000, dims='species')
        _μ = pm.Normal('μ', 4000, 3000, dims='species')
        _mass = pm.Normal('mass', mu=_μ[all_species.codes], sigma=_σ[all_species.codes], observed=penguins['body_mass_g'])
        idata_penguin_mass_all_species = pm.sample()
    return all_species, idata_penguin_mass_all_species


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.4
        """
    )
    return


@app.cell
def _(az, idata_penguin_mass_all_species, plt):
    _axes = az.plot_trace(idata_penguin_mass_all_species, compact=False, divergences='bottom', kind='rank_bars', figsize=(10, 10))
    plt.savefig('img/chp03/all_species_KDE_rankplot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.5 and Figure 3.5
        """
    )
    return


@app.cell
def _(az, idata_penguin_mass_all_species, plt):
    _axes = az.plot_forest(idata_penguin_mass_all_species, var_names=['μ'], figsize=(8, 2.5))
    _axes[0].set_title('μ Mass Estimate: 94.0% HDI')
    plt.savefig('img/chp03/independent_model_forestplotmeans.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.6 and Figure 3.6
        """
    )
    return


@app.cell
def _(az, idata_penguin_mass_all_species, plt):
    _axes = az.plot_forest(idata_penguin_mass_all_species, var_names=['σ'], figsize=(8, 3))
    _axes[0].set_title('σ Mass Estimate: 94.0% HDI')
    plt.savefig('img/chp03/independent_model_forestplotsigma.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.7
        """
    )
    return


@app.cell
def _(all_species, penguins):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    root = tfd.JointDistributionCoroutine.Root
    species_idx = tf.constant(all_species.codes, tf.int32)
    body_mass_g = tf.constant(penguins['body_mass_g'], tf.float32)

    @tfd.JointDistributionCoroutine
    def jd_penguin_mass_all_species():
        _σ = (yield root(tfd.Sample(tfd.HalfStudentT(df=100, loc=0, scale=2000), sample_shape=3, name='sigma')))
        _μ = (yield root(tfd.Sample(tfd.Normal(loc=4000, scale=3000), sample_shape=3, name='mu')))
        _mass = (yield tfd.Independent(tfd.Normal(loc=tf.gather(_μ, species_idx, axis=-1), scale=tf.gather(_σ, species_idx, axis=-1)), reinterpreted_batch_ndims=1, name='mass'))
    return body_mass_g, jd_penguin_mass_all_species, root, tf, tfd, tfp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.8
        """
    )
    return


@app.cell
def _(jd_penguin_mass_all_species):
    _prior_predictive_samples = jd_penguin_mass_all_species.sample(1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.9
        """
    )
    return


@app.cell
def _(jd_penguin_mass_all_species, tf):
    jd_penguin_mass_all_species.sample(sigma=tf.constant([.1, .2, .3]))
    jd_penguin_mass_all_species.sample(mu=tf.constant([.1, .2, .3]));
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.10
        """
    )
    return


@app.cell
def _(body_mass_g, jd_penguin_mass_all_species):
    target_density_function = lambda *x: jd_penguin_mass_all_species.log_prob(*x, mass=body_mass_g)

    jd_penguin_mass_observed = jd_penguin_mass_all_species.experimental_pin(mass=body_mass_g)
    target_density_function = jd_penguin_mass_observed.unnormalized_log_prob

    # init_state = jd_penguin_mass_observed.sample_unpinned(10)
    # target_density_function1(*init_state), target_density_function2(*init_state)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.11
        """
    )
    return


@app.cell
def _(az, body_mass_g, jd_penguin_mass_all_species, np, tf, tfp):
    run_mcmc = tf.function(
        tfp.experimental.mcmc.windowed_adaptive_nuts,
        autograph=False, jit_compile=True)
    mcmc_samples, sampler_stats = run_mcmc(
        1000, jd_penguin_mass_all_species, n_chains=4, num_adaptation_steps=1000,
        mass=body_mass_g)

    idata_penguin_mass_all_species2 = az.from_dict(
        posterior={
            # TFP mcmc returns (num_samples, num_chains, ...), we swap
            # the first and second axis below for each RV so the shape
            # is what ArviZ expected.
            k:np.swapaxes(v, 1, 0)
            for k, v in mcmc_samples._asdict().items()},
        sample_stats={
            k:np.swapaxes(sampler_stats[k], 1, 0)
            for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]}
    )
    return idata_penguin_mass_all_species2, mcmc_samples, run_mcmc


@app.cell
def _(az, idata_penguin_mass_all_species2):
    az.plot_trace(idata_penguin_mass_all_species2, divergences="bottom", kind="rank_bars", figsize=(10,4));
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.12
        """
    )
    return


@app.cell
def _(
    body_mass_g,
    idata_penguin_mass_all_species2,
    jd_penguin_mass_all_species,
    mcmc_samples,
    np,
):
    _prior_predictive_samples = jd_penguin_mass_all_species.sample([1, 1000])
    dist, samples = jd_penguin_mass_all_species.sample_distributions(value=mcmc_samples)
    _ppc_samples = samples[-1]
    ppc_distribution = dist[-1].distribution
    data_log_likelihood = ppc_distribution.log_prob(body_mass_g)
    idata_penguin_mass_all_species2.add_groups(prior=_prior_predictive_samples[:-1]._asdict(), prior_predictive={'mass': _prior_predictive_samples[-1]}, posterior_predictive={'mass': np.swapaxes(_ppc_samples, 1, 0)}, log_likelihood={'mass': np.swapaxes(data_log_likelihood, 1, 0)}, observed_data={'mass': body_mass_g})
    return


@app.cell
def _(az, idata_penguin_mass_all_species2):
    az.plot_ppc(idata_penguin_mass_all_species2, num_pp_samples=50, figsize=(10, 3));
    return


@app.cell
def _(az, idata_penguin_mass_all_species2):
    az.loo(idata_penguin_mass_all_species2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.7
        """
    )
    return


@app.cell
def _(np, plt, stats):
    _fig = plt.figure(figsize=(10, 8))
    _ax = _fig.add_subplot(projection='3d')
    x = np.linspace(-16, 12, 500)
    z = np.array([0, 4, 8])
    for _i, zi in enumerate(z):
        dens = stats.norm(-zi, 3).pdf(x)
        _ax.plot(x, dens, zs=zi + 1, zdir='y', c='k')
        _ax.plot([-zi, -zi], [0, max(dens)], zs=zi + 1, c='k', ls=':', zdir='y')
        _ax.text(-zi, zi - 1, max(dens) * 1.03 + _i / 100, f'$\\mathcal{{N}}(\\beta_0 + \\beta_1 x_{_i}, \\sigma)$', zdir='y', fontsize=18)
    _ax.plot(-z, z + 1, 'C4-', lw=3)
    _ax.set_xlabel('y', fontsize=20)
    _ax.set_ylabel('x', fontsize=24, labelpad=20)
    _ax.set_yticks([zi + 1 for zi in z])
    _ax.set_yticklabels([f'$x_{_i}$' for _i in range(len(z))], fontsize=22)
    _ax.grid(False)
    _ax.set_xticks([])
    _ax.set_zticks([])
    _ax.yaxis.pane.fill = False
    _ax.xaxis.pane.fill = False
    _ax.xaxis.pane.set_edgecolor('None')
    _ax.yaxis.pane.set_edgecolor('None')
    _ax.zaxis.pane.set_facecolor('C3')
    _ax.zaxis.line.set_linewidth(0)
    _ax.view_init(elev=10, azim=-25)
    plt.savefig('img/chp03/3d_linear_regression.png', bbox_inches='tight', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.13
        """
    )
    return


@app.cell
def _(adelie_mask, adelie_mass_obs, penguins, pm):
    adelie_flipper_length_obs = penguins.loc[adelie_mask, 'flipper_length_mm']
    with pm.Model() as model_adelie_flipper_regression:
        adelie_flipper_length = pm.MutableData('adelie_flipper_length', adelie_flipper_length_obs)
        _σ = pm.HalfStudentT('σ', 100, 2000)
        β_0 = pm.Normal('β_0', 0, 4000)
        β_1 = pm.Normal('β_1', 0, 4000)
        _μ = pm.Deterministic('μ', β_0 + β_1 * adelie_flipper_length)
        _mass = pm.Normal('mass', mu=_μ, sigma=_σ, observed=adelie_mass_obs)
        idata_adelie_flipper_regression = pm.sample()
    return (
        adelie_flipper_length_obs,
        idata_adelie_flipper_regression,
        model_adelie_flipper_regression,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.8
        """
    )
    return


@app.cell
def _(az, idata_adelie_flipper_regression, plt):
    _axes = az.plot_posterior(idata_adelie_flipper_regression, var_names=['β_0', 'β_1'], figsize=(10, 3))
    plt.savefig('img/chp03/adelie_coefficient_posterior_plots')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.9
        """
    )
    return


@app.cell
def _(az, idata_adelie_flipper_regression, idata_adelie_mass, plt):
    _axes = az.plot_forest([idata_adelie_mass, idata_adelie_flipper_regression], model_names=['mass_only', 'flipper_regression'], var_names=['σ'], combined=True, figsize=(10, 3.5))
    _axes[0].set_title('σ Comparison 94.0 HDI')
    plt.savefig('img/chp03/SingleSpecies_SingleRegression_Forest_Sigma_Comparison.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.10
        """
    )
    return


@app.cell
def _(idata_adelie_flipper_regression):
    idata_adelie_flipper_regression.posterior["β_0"].mean().item()
    return


@app.cell
def _(
    adelie_flipper_length_obs,
    adelie_mass_obs,
    az,
    idata_adelie_flipper_regression,
    np,
    plt,
):
    _fig, _ax = plt.subplots()
    alpha_m = idata_adelie_flipper_regression.posterior['β_0'].mean().item()
    beta_m = idata_adelie_flipper_regression.posterior['β_1'].mean().item()
    _flipper_length = np.linspace(adelie_flipper_length_obs.min(), adelie_flipper_length_obs.max(), 100)
    flipper_length_mean = alpha_m + beta_m * _flipper_length
    _ax.plot(_flipper_length, flipper_length_mean, c='C4', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    _ax.scatter(adelie_flipper_length_obs, adelie_mass_obs)
    az.plot_hdi(adelie_flipper_length_obs, idata_adelie_flipper_regression.posterior['μ'], hdi_prob=0.94, color='k', ax=_ax)
    _ax.set_xlabel('Flipper Length')
    _ax.set_ylabel('Mass')
    plt.savefig('img/chp03/flipper_length_mass_regression.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.14
        """
    )
    return


@app.cell
def _(
    adelie_flipper_length_obs,
    idata_adelie_flipper_regression,
    model_adelie_flipper_regression,
    pm,
):
    with model_adelie_flipper_regression:
        # Change the underlying value to the mean observed flipper length
        # for our posterior predictive samples
        pm.set_data({"adelie_flipper_length": [adelie_flipper_length_obs.mean()]})
        posterior_predictions = pm.sample_posterior_predictive(
            idata_adelie_flipper_regression.posterior, var_names=["mass", "μ"])
    return (posterior_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.11
        """
    )
    return


@app.cell
def _(az, plt, posterior_predictions):
    _fig, _ax = plt.subplots()
    az.plot_dist(posterior_predictions.posterior_predictive['mass'], label='Posterior Predictive of \nIndividual Penguin Mass', ax=_ax)
    az.plot_dist(posterior_predictions.posterior_predictive['μ'], label='Posterior Predictive of μ', color='C4', ax=_ax)
    _ax.set_xlim(2900, 4500)
    _ax.legend(loc=2)
    _ax.set_xlabel('Mass (grams)')
    _ax.set_yticks([])
    plt.savefig('img/chp03/flipper_length_mass_posterior_predictive.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.15
        """
    )
    return


@app.cell
def _(adelie_mask, penguins):
    adelie_flipper_length_obs_1 = penguins.loc[adelie_mask, 'flipper_length_mm'].values
    adelie_flipper_length_c = adelie_flipper_length_obs_1 - adelie_flipper_length_obs_1.mean()
    return adelie_flipper_length_c, adelie_flipper_length_obs_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### PyMC Centered Model
        """
    )
    return


@app.cell
def _(adelie_flipper_length_c, adelie_mass_obs, pm):
    with pm.Model() as model_adelie_flipper_regression_1:
        _σ = pm.HalfStudentT('σ', 100, 2000)
        β_1_1 = pm.Normal('β_1', 0, 4000)
        β_0_1 = pm.Normal('β_0', 0, 4000)
        _μ = pm.Deterministic('μ', β_0_1 + β_1_1 * adelie_flipper_length_c)
        _mass = pm.Normal('mass', mu=_μ, sigma=_σ, observed=adelie_mass_obs)
        inf_data_adelie_flipper_length_c = pm.sample(random_seed=0)
    return (inf_data_adelie_flipper_length_c,)


@app.cell
def _(az, inf_data_adelie_flipper_length_c, plt):
    az.plot_posterior(inf_data_adelie_flipper_length_c, var_names = ["β_0", "β_1"], figsize=(10, 4));
    plt.savefig("img/chp03/singlespecies_multipleregression_centered.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.16
        """
    )
    return


@app.cell
def _(adelie_flipper_length_obs_1):
    adelie_flipper_length_c_1 = adelie_flipper_length_obs_1 - adelie_flipper_length_obs_1.mean()
    return (adelie_flipper_length_c_1,)


@app.cell
def _(
    adelie_flipper_length_c_1,
    adelie_flipper_length_obs_1,
    adelie_mass_obs,
    az,
    np,
    root,
    run_mcmc,
    tf,
    tfd,
):
    def gen_adelie_flipper_model(adelie_flipper_length):
        adelie_flipper_length = tf.constant(adelie_flipper_length, tf.float32)

        @tfd.JointDistributionCoroutine
        def jd_adelie_flipper_regression():
            _σ = (yield root(tfd.HalfStudentT(df=100, loc=0, scale=2000, name='sigma')))
            β_1 = (yield root(tfd.Normal(loc=0, scale=4000, name='beta_1')))
            β_0 = (yield root(tfd.Normal(loc=0, scale=4000, name='beta_0')))
            _μ = β_0[..., None] + β_1[..., None] * adelie_flipper_length
            _mass = (yield tfd.Independent(tfd.Normal(loc=_μ, scale=_σ[..., None]), reinterpreted_batch_ndims=1, name='mass'))
        return jd_adelie_flipper_regression
    jd_adelie_flipper_regression = gen_adelie_flipper_model(adelie_flipper_length_obs_1)
    jd_adelie_flipper_regression = gen_adelie_flipper_model(adelie_flipper_length_c_1)
    mcmc_samples_1, sampler_stats_1 = run_mcmc(1000, jd_adelie_flipper_regression, n_chains=4, num_adaptation_steps=1000, mass=tf.constant(adelie_mass_obs, tf.float32))
    inf_data_adelie_flipper_length_c_1 = az.from_dict(posterior={k: np.swapaxes(v, 1, 0) for k, v in mcmc_samples_1._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_1[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    return (inf_data_adelie_flipper_length_c_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.12
        """
    )
    return


@app.cell
def _(az, inf_data_adelie_flipper_length_c_1):
    az.plot_posterior(inf_data_adelie_flipper_length_c_1, var_names=['beta_0', 'beta_1'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Multiple Linear Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.17
        """
    )
    return


@app.cell
def _(adelie_flipper_length_obs_1, adelie_mask, adelie_mass_obs, penguins, pm):
    sex_obs = penguins.loc[adelie_mask, 'sex'].replace({'male': 0, 'female': 1})
    with pm.Model() as model_penguin_mass_categorical:
        _σ = pm.HalfStudentT('σ', 100, 2000)
        β_0_2 = pm.Normal('β_0', 0, 3000)
        β_1_2 = pm.Normal('β_1', 0, 3000)
        β_2 = pm.Normal('β_2', 0, 3000)
        _μ = pm.Deterministic('μ', β_0_2 + β_1_2 * adelie_flipper_length_obs_1 + β_2 * sex_obs)
        _mass = pm.Normal('mass', mu=_μ, sigma=_σ, observed=adelie_mass_obs)
        inf_data_penguin_mass_categorical = pm.sample(target_accept=0.9)
    return inf_data_penguin_mass_categorical, sex_obs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.13
        """
    )
    return


@app.cell
def _(az, inf_data_penguin_mass_categorical, plt):
    az.plot_posterior(inf_data_penguin_mass_categorical, var_names =["β_0", "β_1", "β_2"], figsize=(10, 4))
    plt.savefig("img/chp03/adelie_sex_coefficient_posterior.png")
    return


@app.cell
def _(az, inf_data_penguin_mass_categorical):
    az.summary(inf_data_penguin_mass_categorical, var_names=["β_0","β_1","β_2", "σ"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.19 and Figure 3.15
        """
    )
    return


@app.cell
def _(
    az,
    idata_adelie_flipper_regression,
    idata_adelie_mass,
    inf_data_penguin_mass_categorical,
    plt,
):
    _axes = az.plot_forest([idata_adelie_mass, idata_adelie_flipper_regression, inf_data_penguin_mass_categorical], model_names=['mass_only', 'flipper_regression', 'flipper_sex_regression'], var_names=['σ'], combined=True, figsize=(10, 2))
    _axes[0].set_title('σ Comparison 94.0 HDI')
    plt.savefig('img/chp03/singlespecies_multipleregression_forest_sigma_comparison.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.14
        """
    )
    return


@app.cell
def _(
    adelie_flipper_length_obs_1,
    adelie_mass_obs,
    inf_data_penguin_mass_categorical,
    np,
    plt,
    sex_obs,
):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    alpha_1 = inf_data_penguin_mass_categorical.posterior['β_0'].mean().item()
    beta_1 = inf_data_penguin_mass_categorical.posterior['β_1'].mean().item()
    beta_2 = inf_data_penguin_mass_categorical.posterior['β_2'].mean().item()
    _flipper_length = np.linspace(adelie_flipper_length_obs_1.min(), adelie_flipper_length_obs_1.max(), 100)
    mass_mean_male = alpha_1 + beta_1 * _flipper_length
    mass_mean_female = alpha_1 + beta_1 * _flipper_length + beta_2
    _ax.plot(_flipper_length, mass_mean_male, label='Male')
    _ax.plot(_flipper_length, mass_mean_female, c='C4', label='Female')
    _ax.scatter(adelie_flipper_length_obs_1, adelie_mass_obs, c=[{0: 'k', 1: 'b'}[code] for code in sex_obs.values])
    _ax.set_xlabel('Flipper Length')
    _ax.set_ylabel('Mass')
    _ax.legend()
    plt.savefig('img/chp03/single_species_categorical_regression.png')
    return


@app.cell
def _(
    adelie_flipper_length_obs_1,
    adelie_mask,
    adelie_mass_obs,
    penguins,
    root,
    run_mcmc,
    sex_obs,
    tf,
    tfd,
):
    def gen_jd_flipper_bill_sex(flipper_length, sex, bill_length, dtype=tf.float32):
        _flipper_length, sex, _bill_length = tf.nest.map_structure(lambda x: tf.constant(x, dtype), (_flipper_length, sex, _bill_length))

        @tfd.JointDistributionCoroutine
        def jd_flipper_bill_sex():
            _σ = (yield root(tfd.HalfStudentT(df=100, loc=0, scale=2000, name='sigma')))
            β_0 = (yield root(tfd.Normal(loc=0, scale=3000, name='beta_0')))
            β_1 = (yield root(tfd.Normal(loc=0, scale=3000, name='beta_1')))
            β_2 = (yield root(tfd.Normal(loc=0, scale=3000, name='beta_2')))
            β_3 = (yield root(tfd.Normal(loc=0, scale=3000, name='beta_3')))
            _μ = β_0[..., None] + β_1[..., None] * _flipper_length + β_2[..., None] * sex + β_3[..., None] * _bill_length
            _mass = (yield tfd.Independent(tfd.Normal(loc=_μ, scale=_σ[..., None]), reinterpreted_batch_ndims=1, name='mass'))
        return jd_flipper_bill_sex
    bill_length_obs = penguins.loc[adelie_mask, 'bill_length_mm']
    jd_flipper_bill_sex = gen_jd_flipper_bill_sex(adelie_flipper_length_obs_1, sex_obs, bill_length_obs)
    mcmc_samples_2, sampler_stats_2 = run_mcmc(1000, jd_flipper_bill_sex, n_chains=4, num_adaptation_steps=1000, mass=tf.constant(adelie_mass_obs, tf.float32))
    return (
        bill_length_obs,
        gen_jd_flipper_bill_sex,
        mcmc_samples_2,
        sampler_stats_2,
    )


@app.cell
def _(az, mcmc_samples_2, np, sampler_stats_2):
    idata_model_penguin_flipper_bill_sex = az.from_dict(posterior={k: np.swapaxes(v, 1, 0) for k, v in mcmc_samples_2._asdict().items()}, sample_stats={k: np.swapaxes(sampler_stats_2[k], 1, 0) for k in ['target_log_prob', 'diverging', 'accept_ratio', 'n_steps']})
    return (idata_model_penguin_flipper_bill_sex,)


@app.cell
def _(az, idata_model_penguin_flipper_bill_sex):
    az.plot_posterior(idata_model_penguin_flipper_bill_sex, var_names=["beta_1", "beta_2", "beta_3"]);
    return


@app.cell
def _(az, idata_model_penguin_flipper_bill_sex):
    az.summary(idata_model_penguin_flipper_bill_sex, var_names=["beta_1", "beta_2", "beta_3", "sigma"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.21
        """
    )
    return


@app.cell
def _(
    adelie_mask,
    bill_length_obs,
    gen_jd_flipper_bill_sex,
    mcmc_samples_2,
    np,
    penguins,
):
    mean_flipper_length = penguins.loc[adelie_mask, 'flipper_length_mm'].mean()
    counterfactual_flipper_lengths = np.linspace(mean_flipper_length - 20, mean_flipper_length + 20, 21)
    sex_male_indicator = np.zeros_like(counterfactual_flipper_lengths)
    mean_bill_length = np.ones_like(counterfactual_flipper_lengths) * bill_length_obs.mean()
    jd_flipper_bill_sex_counterfactual = gen_jd_flipper_bill_sex(counterfactual_flipper_lengths, sex_male_indicator, mean_bill_length)
    _ppc_samples = jd_flipper_bill_sex_counterfactual.sample(value=mcmc_samples_2)
    estimated_mass = _ppc_samples[-1].numpy().reshape(-1, 21)
    return counterfactual_flipper_lengths, estimated_mass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.16
        """
    )
    return


@app.cell
def _(az, counterfactual_flipper_lengths, estimated_mass, plt):
    _, _ax = plt.subplots(figsize=(10, 3))
    az.plot_hdi(counterfactual_flipper_lengths, estimated_mass, color='C2', plot_kwargs={'ls': '--'}, ax=_ax)
    _ax.plot(counterfactual_flipper_lengths, estimated_mass.mean(axis=0), lw=4, c='blue')
    _ax.set_title('Mass estimates with Flipper Length Counterfactual for \n Male Penguins at Mean Observed Bill Length')
    _ax.set_xlabel('Counterfactual Flipper Length')
    _ax.set_ylabel('Mass')
    plt.savefig('img/chp03/linear_counter_factual.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Generalized Linear Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.17
        """
    )
    return


@app.cell
def _(np, plt, special):
    x_1 = np.linspace(-10, 10, 1000)
    y = special.expit(x_1)
    plt.figure(figsize=(10, 2))
    plt.plot(x_1, y)
    plt.savefig('img/chp03/logistic.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.22
        """
    )
    return


@app.cell
def _(pd, penguins, pm):
    species_filter = penguins['species'].isin(['Adelie', 'Chinstrap'])
    bill_length_obs_1 = penguins.loc[species_filter, 'bill_length_mm'].values
    species = pd.Categorical(penguins.loc[species_filter, 'species'])
    with pm.Model() as model_logistic_penguins_bill_length:
        β_0_3 = pm.Normal('β_0', mu=0, sigma=10)
        β_1_3 = pm.Normal('β_1', mu=0, sigma=10)
        _μ = β_0_3 + pm.math.dot(bill_length_obs_1, β_1_3)
        _θ = pm.Deterministic('θ', pm.math.sigmoid(_μ))
        _bd = pm.Deterministic('bd', -β_0_3 / β_1_3)
        _yl = pm.Bernoulli('yl', p=_θ, observed=species.codes)
        idata_logistic_penguins_bill_length = pm.sample(5000, chains=2, random_seed=0, idata_kwargs={'log_likelihood': True})
        idata_logistic_penguins_bill_length.extend(pm.sample_prior_predictive(samples=10000))
        idata_logistic_penguins_bill_length.extend(pm.sample_posterior_predictive(idata_logistic_penguins_bill_length))
    return (
        bill_length_obs_1,
        idata_logistic_penguins_bill_length,
        species,
        species_filter,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.18
        """
    )
    return


@app.cell
def _(az, idata_logistic_penguins_bill_length, plt):
    _, _ax = plt.subplots(figsize=(10, 2))
    az.plot_dist(idata_logistic_penguins_bill_length.prior_predictive['yl'], color='C2', ax=_ax)
    _ax.set_xticklabels(['Adelie: 0', 'Chinstrap: 1'])
    plt.savefig('img/chp03/prior_predictive_logistic.png')
    return


@app.cell
def _(az, idata_logistic_penguins_bill_length):
    az.plot_trace(idata_logistic_penguins_bill_length, var_names=["β_0", "β_1"], kind="rank_bars");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 3.3
        """
    )
    return


@app.cell
def _(az, idata_logistic_penguins_bill_length):
    az.summary(idata_logistic_penguins_bill_length, var_names=["β_0", "β_1"], kind="stats")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.19
        """
    )
    return


@app.cell
def _(
    az,
    bill_length_obs_1,
    idata_logistic_penguins_bill_length,
    np,
    plt,
    species,
):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _theta = idata_logistic_penguins_bill_length.posterior['θ'].mean(('chain', 'draw'))
    _idx = np.argsort(bill_length_obs_1)
    _ax.vlines(idata_logistic_penguins_bill_length.posterior['bd'].values.mean(), 0, 1, color='k')
    bd_hpd = az.hdi(idata_logistic_penguins_bill_length.posterior['bd'].values.flatten(), ax=_ax)
    plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='C2', alpha=0.5)
    for _i, (_label, _marker) in enumerate(zip(species.categories, ('.', 's'))):
        _filter = species.codes == _i
        x_2 = bill_length_obs_1[_filter]
        y_1 = np.random.normal(_i, 0.02, size=_filter.sum())
        _ax.scatter(bill_length_obs_1[_filter], y_1, marker=_marker, label=_label, alpha=0.8)
    az.plot_hdi(bill_length_obs_1, idata_logistic_penguins_bill_length.posterior['θ'].values, color='C4', ax=_ax, plot_kwargs={'zorder': 10})
    _ax.plot(bill_length_obs_1[_idx], _theta[_idx], color='C4', zorder=10)
    _ax.set_xlabel('Bill Length (mm)')
    _ax.set_ylabel('θ', rotation=0)
    plt.legend()
    plt.savefig('img/chp03/logistic_bill_length.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Code 3.23
        """
    )
    return


@app.cell
def _(penguins, pm, species, species_filter):
    mass_obs = penguins.loc[species_filter, 'body_mass_g'].values
    with pm.Model() as model_logistic_penguins_mass:
        β_0_4 = pm.Normal('β_0', mu=0, sigma=10)
        β_1_4 = pm.Normal('β_1', mu=0, sigma=10)
        _μ = β_0_4 + pm.math.dot(mass_obs, β_1_4)
        _θ = pm.Deterministic('θ', pm.math.sigmoid(_μ))
        _bd = pm.Deterministic('bd', -β_0_4 / β_1_4)
        _yl = pm.Bernoulli('yl', p=_θ, observed=species.codes)
        idata_logistic_penguins_mass = pm.sample(5000, chains=2, target_accept=0.9, random_seed=0, idata_kwargs={'log_likelihood': True})
        idata_logistic_penguins_mass.extend(pm.sample_posterior_predictive(idata_logistic_penguins_mass))
    return idata_logistic_penguins_mass, mass_obs


@app.cell
def _(az, idata_logistic_penguins_mass):
    az.plot_trace(idata_logistic_penguins_mass, var_names=["β_0", "β_1"], kind="rank_bars");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 3.4
        """
    )
    return


@app.cell
def _(az, idata_logistic_penguins_mass):
    az.summary(idata_logistic_penguins_mass, var_names=["β_0", "β_1", "bd"], kind="stats")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.20
        """
    )
    return


@app.cell
def _(species):
    species.codes
    return


@app.cell
def _(az, idata_logistic_penguins_mass, mass_obs, np, plt, species):
    _theta = idata_logistic_penguins_mass.posterior['θ'].mean(('chain', 'draw'))
    _bd = idata_logistic_penguins_mass.posterior['bd']
    _fig, _ax = plt.subplots()
    _idx = np.argsort(mass_obs)
    _ax.plot(mass_obs[_idx], _theta[_idx], color='C4', lw=3)
    for _i, (_label, _marker) in enumerate(zip(species.categories, ('.', 's'))):
        _filter = species.codes == _i
        x_3 = mass_obs[_filter]
        y_2 = np.random.normal(_i, 0.02, size=_filter.sum())
        _ax.scatter(mass_obs[_filter], y_2, marker=_marker, label=_label, alpha=0.8)
    az.plot_hdi(mass_obs, idata_logistic_penguins_mass.posterior['θ'], color='C4', ax=_ax)
    _ax.set_xlabel('Mass (Grams)')
    _ax.set_ylabel('θ', rotation=0)
    plt.legend()
    plt.savefig('img/chp03/logistic_mass.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.24
        """
    )
    return


@app.cell
def _(penguins, pm, species, species_filter):
    X = penguins.loc[species_filter, ['bill_length_mm', 'body_mass_g']]
    X.insert(0, 'Intercept', value=1)
    X = X.values
    with pm.Model() as model_logistic_penguins_bill_length_mass:
        β = pm.Normal('β', mu=0, sigma=20, shape=3)
        _μ = pm.math.dot(X, β)
        _θ = pm.Deterministic('θ', pm.math.sigmoid(_μ))
        _bd = pm.Deterministic('bd', -β[0] / β[2] - β[1] / β[2] * X[:, 1])
        _yl = pm.Bernoulli('yl', p=_θ, observed=species.codes)
        idata_logistic_penguins_bill_length_mass = pm.sample(5000, chains=2, random_seed=0, target_accept=0.9, idata_kwargs={'log_likelihood': True})
        idata_logistic_penguins_bill_length_mass.extend(pm.sample_posterior_predictive(idata_logistic_penguins_bill_length_mass))
    return X, idata_logistic_penguins_bill_length_mass


@app.cell
def _(az, idata_logistic_penguins_bill_length_mass):
    az.plot_trace(idata_logistic_penguins_bill_length_mass, compact=False, var_names=["β"], kind="rank_bars");
    return


@app.cell
def _(az, idata_logistic_penguins_bill_length_mass):
    az.summary(idata_logistic_penguins_bill_length_mass, var_names=["β"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.21
        """
    )
    return


@app.cell
def _(X, az, idata_logistic_penguins_bill_length_mass, np, plt, species):
    _fig, _ax = plt.subplots()
    _idx = np.argsort(X[:, 1])
    _bd = idata_logistic_penguins_bill_length_mass.posterior['bd'].mean(('chain', 'draw'))[_idx]
    species_filter_1 = species.codes.astype(bool)
    _ax.plot(X[:, 1][_idx], _bd, color='C4')
    az.plot_hdi(X[:, 1], idata_logistic_penguins_bill_length_mass.posterior['bd'], color='C4', ax=_ax)
    _ax.scatter(X[~species_filter_1, 1], X[~species_filter_1, 2], alpha=0.8, label='Adelie', zorder=10)
    _ax.scatter(X[species_filter_1, 1], X[species_filter_1, 2], marker='s', label='Chinstrap', zorder=10)
    _ax.set_ylabel('Mass (grams)')
    _ax.set_xlabel('Bill Length (mm)')
    _ax.legend()
    plt.savefig('img/chp03/decision_boundary_logistic_mass_bill_length.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.26
        """
    )
    return


@app.cell
def _(
    az,
    idata_logistic_penguins_bill_length,
    idata_logistic_penguins_bill_length_mass,
    idata_logistic_penguins_mass,
):
    az.compare({"mass": idata_logistic_penguins_mass,
                "bill": idata_logistic_penguins_bill_length,
                "mass_bill": idata_logistic_penguins_bill_length_mass}).round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.25
        """
    )
    return


@app.cell
def _(
    az,
    idata_logistic_penguins_bill_length,
    idata_logistic_penguins_bill_length_mass,
    idata_logistic_penguins_mass,
    plt,
):
    models = {'bill': idata_logistic_penguins_bill_length, 'mass': idata_logistic_penguins_mass, 'mass bill': idata_logistic_penguins_bill_length_mass}
    _, _axes = plt.subplots(3, 1, figsize=(12, 4), sharey=True)
    for (_label, model), _ax in zip(models.items(), _axes):
        az.plot_separation(model, 'yl', ax=_ax, color='C4')
        _ax.set_title(_label)
    plt.savefig('img/chp03/penguins_separation_plot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.27
        """
    )
    return


@app.cell
def _(penguins):
    penguins.loc[:,"species"].value_counts()
    return


@app.cell
def _(penguins):
    counts = penguins["species"].value_counts()
    adelie_count = counts["Adelie"],
    chinstrap_count = counts["Chinstrap"]
    adelie_count / (adelie_count+chinstrap_count)
    return adelie_count, chinstrap_count


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.28
        """
    )
    return


@app.cell
def _(adelie_count, chinstrap_count):
    adelie_count / chinstrap_count
    return


@app.cell
def _(idata_logistic_penguins_bill_length):
    idata_logistic_penguins_bill_length.posterior["β_0"].mean().item()
    return


@app.cell
def _(idata_logistic_penguins_bill_length):
    β_0_5 = idata_logistic_penguins_bill_length.posterior['β_0'].mean().item()
    β_1_5 = idata_logistic_penguins_bill_length.posterior['β_1'].mean().item()
    return β_0_5, β_1_5


@app.cell
def _(β_0_5):
    β_0_5
    return


@app.cell
def _(β_1_5):
    β_1_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.29
        """
    )
    return


@app.cell
def _(special, β_0_5, β_1_5):
    _bill_length = 45
    val_1 = β_0_5 + β_1_5 * _bill_length
    val_2 = β_0_5 + β_1_5 * (_bill_length + 1)
    f'Class Probability change from 45mm Bill Length to 46mm: {(special.expit(val_2) - special.expit(val_1)) * 100:.0f}%'
    return


@app.cell
def _(np, β_0_5, β_1_5):
    _bill_length = np.array([30, 45])
    val_1_1 = β_0_5 + β_1_5 * _bill_length
    val_2_1 = β_0_5 + β_1_5 * (_bill_length + 1)
    return val_1_1, val_2_1


@app.cell
def _(special, val_1_1, val_2_1):
    special.expit(val_2_1) - special.expit(val_1_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Picking Priors in Regression Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.23
        """
    )
    return


@app.cell
def _(np):
    x_4 = np.arange(-2, 3, 1)
    y_3 = [50, 44, 50, 47, 56]
    return x_4, y_3


@app.cell
def _(plt, x_4, y_3):
    import matplotlib.ticker as mtick
    _fig, _ax = plt.subplots()
    _ax.scatter(x_4, y_3)
    _ax.set_xticks(x_4)
    _ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _ax.set_ylim(40, 60)
    _ax.set_xlabel('Attractiveness of Parent')
    _ax.set_ylabel('% of Girl Babies')
    _ax.set_title('Attractiveness of Parent and Sex Ratio')
    plt.savefig('img/chp03/beautyratio.png')
    return (mtick,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.30
        """
    )
    return


@app.cell
def _(pm, x_4, y_3):
    with pm.Model() as model_uninformative_prior_sex_ratio:
        _σ = pm.Exponential('σ', 0.5)
        β_1_6 = pm.Normal('β_1', 0, 20)
        β_0_6 = pm.Normal('β_0', 50, 20)
        _μ = pm.Deterministic('μ', β_0_6 + β_1_6 * x_4)
        _ratio = pm.Normal('ratio', mu=_μ, sigma=_σ, observed=y_3)
        idata_uninformative_prior_sex_ratio = pm.sample(random_seed=0)
        idata_uninformative_prior_sex_ratio.extend(pm.sample_prior_predictive(samples=10000))
    return (idata_uninformative_prior_sex_ratio,)


@app.cell
def _(az, idata_uninformative_prior_sex_ratio, plt):
    az.plot_posterior(idata_uninformative_prior_sex_ratio.prior, var_names=["β_0", "β_1"])
    plt.savefig("img/chp03/priorpredictiveuninformativeKDE.png")
    return


@app.cell
def _(az, idata_uninformative_prior_sex_ratio):
    az.summary(idata_uninformative_prior_sex_ratio, var_names=["β_0", "β_1", "σ"], kind="stats")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.24
        """
    )
    return


@app.cell
def _(az, idata_uninformative_prior_sex_ratio, mtick, np, plt, x_4, xr, y_3):
    _fig, _axes = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
    np.random.seed(0)
    _num_samples = 50
    _subset = az.extract(idata_uninformative_prior_sex_ratio, group='prior', num_samples=50)
    _axes[0].plot(x_4, (_subset['β_0'] + _subset['β_1'] * xr.DataArray(x_4)).T, c='black', alpha=0.3)
    _b_0_hat = idata_uninformative_prior_sex_ratio.prior['β_0'].values.mean()
    _b_1_hat = idata_uninformative_prior_sex_ratio.prior['β_1'].values.mean()
    _axes[0].plot(x_4, _b_0_hat + _b_1_hat * x_4, c='C4', linewidth=4)
    _axes[0].scatter(x_4, y_3)
    _axes[0].set_xticks(x_4)
    _axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _axes[0].set_ylim(40, 60)
    _axes[0].set_ylabel('% of Girl Babies')
    _axes[0].set_title('Prior samples from informative priors')
    np.random.seed(0)
    _num_samples = 50
    _subset = az.extract(idata_uninformative_prior_sex_ratio, group='prior', num_samples=50)
    _axes[1].plot(x_4, (_subset['β_0'] + _subset['β_1'] * xr.DataArray(x_4)).T, c='black', alpha=0.3)
    _b_0_hat = idata_uninformative_prior_sex_ratio.posterior['β_0'].values.mean()
    _b_1_hat = idata_uninformative_prior_sex_ratio.posterior['β_1'].values.mean()
    _axes[1].plot(x_4, _b_0_hat + _b_1_hat * x_4, c='C4', linewidth=4)
    _axes[1].scatter(x_4, y_3)
    _axes[1].set_xticks(x_4)
    _axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _axes[1].set_ylim(40, 60)
    _axes[1].set_xlabel('Attractiveness of Parent')
    _axes[1].set_ylabel('% of Girl Babies')
    _axes[1].set_title('Posterior samples from informative priors')
    plt.savefig('img/chp03/posterioruninformativelinearregression.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 3.31
        """
    )
    return


@app.cell
def _(pm, x_4, y_3):
    with pm.Model() as model_informative_prior_sex_ratio:
        _σ = pm.Exponential('σ', 0.5)
        β_1_7 = pm.Normal('β_1', 0, 0.5)
        β_0_7 = pm.Normal('β_0', 48.5, 0.5)
        _μ = pm.Deterministic('μ', β_0_7 + β_1_7 * x_4)
        _ratio = pm.Normal('ratio', mu=_μ, sigma=_σ, observed=y_3)
        idata_informative_prior_sex_ratio = pm.sample(random_seed=0)
        idata_informative_prior_sex_ratio.extend(pm.sample_prior_predictive(samples=10000))
    return (idata_informative_prior_sex_ratio,)


@app.cell
def _(az, idata_informative_prior_sex_ratio):
    az.summary(idata_informative_prior_sex_ratio, var_names=["β_0", "β_1", "σ"], kind="stats")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 3.25
        """
    )
    return


@app.cell
def _(az, idata_informative_prior_sex_ratio, mtick, np, plt, x_4, xr, y_3):
    _fig, _axes = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
    np.random.seed(0)
    _num_samples = 50
    _subset = az.extract(idata_informative_prior_sex_ratio, group='prior', num_samples=50)
    _axes[0].plot(x_4, (_subset['β_0'] + _subset['β_1'] * xr.DataArray(x_4)).T, c='black', alpha=0.3)
    _b_0_hat = idata_informative_prior_sex_ratio.prior['β_0'].values.mean()
    _b_1_hat = idata_informative_prior_sex_ratio.prior['β_1'].values.mean()
    _axes[0].plot(x_4, _b_0_hat + _b_1_hat * x_4, c='C4', linewidth=4)
    _axes[0].scatter(x_4, y_3)
    _axes[0].set_xticks(x_4)
    _axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _axes[0].set_ylim(40, 60)
    _axes[0].set_ylabel('% of Girl Babies')
    _axes[0].set_title('Prior samples from informative priors')
    np.random.seed(0)
    _num_samples = 50
    _subset = az.extract(idata_informative_prior_sex_ratio, group='prior', num_samples=50)
    _axes[1].plot(x_4, (_subset['β_0'] + _subset['β_1'] * xr.DataArray(x_4)).T, c='black', alpha=0.3)
    _b_0_hat = idata_informative_prior_sex_ratio.posterior['β_0'].values.mean()
    _b_1_hat = idata_informative_prior_sex_ratio.posterior['β_1'].values.mean()
    _axes[1].plot(x_4, _b_0_hat + _b_1_hat * x_4, c='C4', linewidth=4)
    _axes[1].scatter(x_4, y_3)
    _axes[1].set_xticks(x_4)
    _axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _axes[1].set_ylim(40, 60)
    _axes[1].set_xlabel('Attractiveness of Parent')
    _axes[1].set_ylabel('% of Girl Babies')
    _axes[1].set_title('Posterior samples from informative priors')
    (_b_0_hat, _b_1_hat)
    plt.savefig('img/chp03/posteriorinformativelinearregression.png')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
