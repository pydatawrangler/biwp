import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 11: Appendiceal Topics
        """
    )
    return


@app.cell
def _():
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from scipy.special import binom, betaln
    return az, betaln, binom, np, plt, stats


@app.cell
def _(az, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams['figure.dpi'] = 300
    return


@app.cell
def _(np):
    np.random.seed(14067)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Probability Background
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 11.1
        """
    )
    return


@app.cell
def _(np):
    def die():
        outcomes = [1, 2, 3, 4, 5, 6]
        return np.random.choice(outcomes)

    die()
    return (die,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 11.2
        """
    )
    return


@app.cell
def _(die):
    def experiment(N=10):
        sample = [die() for _i in range(N)]
        for _i in range(1, 7):
            print(f'{_i}: {sample.count(_i) / N:.2g}')
    experiment()
    return


@app.cell
def _(np, stats):
    a = 1
    b = 6
    rv = stats.randint(a, b+1)
    x = np.arange(1, b+1)

    x_pmf = rv.pmf(x)  # evaluate the pmf at the x values
    x_cdf = rv.cdf(x)  # evaluate the cdf at the x values
    mean, variance = rv.stats(moments="mv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 11.17
        """
    )
    return


@app.cell
def _(az, np, plt, stats):
    xs = (np.linspace(0, 20, 200), np.linspace(0, 1, 200), np.linspace(-4, 4, 200))
    _dists = (stats.expon(scale=5), stats.beta(0.5, 0.5), stats.norm(0, 1))
    _, _ax = plt.subplots(3, 3)
    for _idx, (_dist, x_1) in enumerate(zip(_dists, xs)):
        draws = _dist.rvs(100000)
        _data = _dist.cdf(draws)
        _ax[_idx, 0].plot(x_1, _dist.pdf(x_1))
        _ax[_idx, 1].plot(np.sort(_data), np.linspace(0, 1, len(_data)))
        az.plot_kde(_data, ax=_ax[_idx, 2])
    return


@app.cell
def _(np, plt, stats):
    x_2 = range(0, 26)
    q_pmf = stats.binom(10, 0.75).pmf(x_2)
    qu_pmf = stats.randint(0, np.max(np.nonzero(q_pmf)) + 1).pmf(x_2)
    r_pmf = (q_pmf + np.roll(q_pmf, 12)) / 2
    ru_pmf = stats.randint(0, np.max(np.nonzero(r_pmf)) + 1).pmf(x_2)
    s_pmf = (q_pmf + np.roll(q_pmf, 15)) / 2
    su_pmf = (qu_pmf + np.roll(qu_pmf, 15)) / 2
    _, _ax = plt.subplots(3, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    _ax = np.ravel(_ax)
    zipped = zip([q_pmf, qu_pmf, r_pmf, ru_pmf, s_pmf, su_pmf], ['q', 'qu', 'r', 'ru', 's', 'su'])
    for _idx, (_dist, label) in enumerate(zipped):
        _ax[_idx].vlines(x_2, 0, _dist, label=f'H = {stats.entropy(_dist):.2f}')
        _ax[_idx].set_title(label)
        _ax[_idx].legend(loc=1, handlelength=0)
    plt.savefig('img/chp11/entropy.png')
    return q_pmf, qu_pmf, r_pmf, ru_pmf, s_pmf, su_pmf, x_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Kullback-Leibler Divergence
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 11.6 and Figure 11.23
        """
    )
    return


@app.cell
def _(np, plt, q_pmf, qu_pmf, r_pmf, ru_pmf, s_pmf, stats, su_pmf):
    _dists = [q_pmf, qu_pmf, r_pmf, ru_pmf, s_pmf, su_pmf]
    names = ['q', 'qu', 'r', 'ru', 's', 'su']
    _fig, _ax = plt.subplots()
    KL_matrix = np.zeros((6, 6))
    for _i, dist_i in enumerate(_dists):
        for _j, dist_j in enumerate(_dists):
            KL_matrix[_i, _j] = stats.entropy(dist_i, dist_j)
    _ax.set_xticks(np.arange(len(names)))
    _ax.set_yticks(np.arange(len(names)))
    _ax.set_xticklabels(names)
    _ax.set_yticklabels(names)
    plt.set_cmap('viridis')
    cmap = plt.cm.get_cmap()
    cmap.set_bad('w', 0.3)
    _im = _ax.imshow(KL_matrix)
    _fig.colorbar(_im, extend='max')
    plt.savefig('img/chp11/KL_heatmap.png')
    return


@app.cell
def _(betaln, binom, np):
    def beta_binom(prior, y):
        """
        Compute the marginal-log-likelihood for a beta-binomial model,
        analytically.

        prior : tuple
            tuple of alpha and beta parameter for the prior (beta distribution)
        y : array
            array with "1" and "0" corresponding to the success and fails respectively
        """
        α, β = prior
        success = np.sum(_y)
        trials = len(_y)
        return np.log(binom(trials, success)) + betaln(α + success, β + trials - success) - betaln(α, β)
    return (beta_binom,)


@app.cell
def _(np, stats):
    def beta_binom_harmonic(prior, y, s=10000):
        """
        Compute the marginal-log-likelihood for a beta-binomial model,
        using the harmonic mean estimator.

        prior : tuple
            tuple of alpha and beta parameter for the prior (beta distribution)
        y : array
            array with "1" and "0" corresponding to the success and fails respectively
        s : int
            number of samples from the posterior
        """
        α, β = prior
        success = np.sum(_y)
        trials = len(_y)
        _posterior_samples = stats.beta(α + success, β + trials - success).rvs(s)
        log_likelihood = stats.binom.logpmf(success, trials, _posterior_samples)
        return 1 / np.mean(1 / log_likelihood)
    return (beta_binom_harmonic,)


@app.cell
def _(beta_binom, beta_binom_harmonic, np, plt):
    _data = [np.repeat([1, 0], rep) for rep in ((1, 0), (1, 1), (5, 5), (100, 100))]
    priors = ((1, 1), (100, 100), (1, 2), (1, 200))
    x_names = [repr((sum(x), len(x) - sum(x))) for x in _data]
    y_names = ['Beta' + repr(x) for x in priors]
    _fig, _ax = plt.subplots()
    error_matrix = np.zeros((len(priors), len(_data)))
    for _i, prior in enumerate(priors):
        for _j, _y in enumerate(_data):
            error_matrix[_i, _j] = 100 * (1 - beta_binom_harmonic(prior, _y) / beta_binom(prior, _y))
    _im = _ax.imshow(error_matrix, cmap='viridis')
    _ax.set_xticks(np.arange(len(x_names)))
    _ax.set_yticks(np.arange(len(y_names)))
    _ax.set_xticklabels(x_names)
    _ax.set_yticklabels(y_names)
    _fig.colorbar(_im)
    plt.savefig('img/chp11/harmonic_mean_heatmap.png')
    return


@app.cell
def _(np, stats, x_2):
    def normal_harmonic(sd_0, sd_1, y, s=10000):
        post_tau = 1 / sd_0 ** 2 + 1 / sd_1 ** 2
        _posterior_samples = stats.norm(loc=_y / sd_1 ** 2 / post_tau, scale=(1 / post_tau) ** 0.5).rvs((s, len(x_2)))
        log_likelihood = stats.norm.logpdf(loc=x_2, scale=sd_1, x=_posterior_samples).sum(1)
        return 1 / np.mean(1 / log_likelihood)
    return


@app.cell
def _(np, stats):
    _σ_0 = 1
    σ_1 = 1
    _y = np.array([0])
    stats.norm.logpdf(loc=0, scale=(_σ_0 ** 2 + σ_1 ** 2) ** 0.5, x=_y).sum()
    return (σ_1,)


@app.cell
def _(az, np, stats):
    def posterior_ml_ic_normal(σ_0=1, σ_1=1, y=[1]):
        n = len(_y)
        var_μ = 1 / (1 / _σ_0 ** 2 + n / σ_1 ** 2)
        μ = var_μ * np.sum(_y) / σ_1 ** 2
        σ_μ = var_μ ** 0.5
        posterior = stats.norm(loc=μ, scale=σ_μ)
        samples = posterior.rvs(size=(2, 1000))
        log_likelihood = stats.norm(loc=samples[:, :, None], scale=σ_1).logpdf(_y)
        idata = az.from_dict(log_likelihood={'o': log_likelihood})
        log_ml = stats.norm.logpdf(loc=0, scale=(_σ_0 ** 2 + σ_1 ** 2) ** 0.5, x=_y).sum()
        x = np.linspace(-5, 6, 300)
        density = posterior.pdf(x)
        return (μ, σ_μ, x, density, log_ml, az.waic(idata).elpd_waic, az.loo(idata, reff=1).elpd_loo)
    return (posterior_ml_ic_normal,)


@app.cell
def _(np, plt, posterior_ml_ic_normal, stats, σ_1):
    _y = np.array([0.65225338, -0.06122589, 0.27745188, 1.38026371, -0.72751008, -1.10323829, 2.07122286, -0.52652711, 0.51528113, 0.71297661])
    _, _ax = plt.subplots(3, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)
    for _i, _σ_0 in enumerate((1, 10, 100)):
        μ_μ, σ_μ, x_3, density, log_ml, waic, loo = posterior_ml_ic_normal(_σ_0, σ_1, _y)
        _ax[_i].plot(x_3, stats.norm(loc=0, scale=(_σ_0 ** 2 + σ_1 ** 2) ** 0.5).pdf(x_3), lw=2)
        _ax[_i].plot(x_3, density, lw=2, color='C4')
        _ax[_i].plot(0, label=f'log_ml {log_ml:.1f}\nwaic {waic:.1f}\nloo {loo:.1f}\n', alpha=0)
        _ax[_i].set_title(f'μ_μ={μ_μ:.2f} σ_μ={σ_μ:.2f}')
        _ax[_i].legend()
    _ax[2].set_yticks([])
    _ax[2].set_xlabel('μ')
    plt.savefig('img/chp11/ml_waic_loo.png')
    return


@app.cell
def _(np, stats):
    _σ_0 = 1
    σ_1_1 = 1
    _y = np.array([0])
    stats.norm.logpdf(loc=0, scale=(_σ_0 ** 2 + σ_1_1 ** 2) ** 0.5, x=_y).sum()
    return


@app.cell
def _(np):
    N = 10000
    x_4, _y = np.random.uniform(-1, 1, size=(2, N))
    _inside = x_4 ** 2 + _y ** 2 <= 1
    pi = _inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return


@app.cell
def _(np, plt):
    total = 100000
    dims = []
    prop = []
    for d in range(2, 15):
        x_5 = np.random.random(size=(d, total))
        _inside = ((x_5 * x_5).sum(axis=0) < 1).sum()
        dims.append(d)
        prop.append(_inside / total)
    plt.plot(dims, prop)
    return


@app.cell
def _(np, stats):
    def posterior_grid(ngrid=10, α=1, β=1, heads=6, trials=9):
        _grid = np.linspace(0, 1, ngrid)
        prior = stats.beta(α, β).pdf(_grid)
        likelihood = stats.binom.pmf(heads, trials, _grid)
        posterior = likelihood * prior
        posterior = posterior / posterior.sum()
        return posterior
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Variational Inference
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        See https://blog.tensorflow.org/2021/02/variational-inference-with-joint-distributions-in-tensorflow-probability.html for a more extended examples
        """
    )
    return


@app.cell
def _(az):
    az.style.use("arviz-colors")

    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    return tf, tfd, tfp


@app.cell
def _(tf, tfp):
    target_logprob = lambda x, y: -(1.0 - x) ** 2 - 1.5 * (_y - x ** 2) ** 2
    _event_shape = [(), ()]
    mean_field_surrogate_posterior = tfp.experimental.vi.build_affine_surrogate_posterior(event_shape=_event_shape, operators='diag')
    full_rank_surrogate_posterior = tfp.experimental.vi.build_affine_surrogate_posterior(event_shape=_event_shape, operators='tril')
    losses = []
    _posterior_samples = []
    for _approx in [mean_field_surrogate_posterior, full_rank_surrogate_posterior]:
        _loss = tfp.vi.fit_surrogate_posterior(target_logprob, _approx, num_steps=200, optimizer=tf.optimizers.Adam(0.1), sample_size=5)
        losses.append(_loss)
        _posterior_samples.append(_approx.sample(10000))
    return (
        full_rank_surrogate_posterior,
        losses,
        mean_field_surrogate_posterior,
        target_logprob,
    )


@app.cell
def _(losses, np, plt):
    plt.plot(np.asarray(losses).T)
    plt.legend(['mean-field', 'full-rank']);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 11.33
        """
    )
    return


@app.cell
def _(
    full_rank_surrogate_posterior,
    mean_field_surrogate_posterior,
    np,
    plt,
    target_logprob,
):
    _grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 5, 100))
    _Z = -target_logprob(*_grid)
    _, _axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for _ax, _approx, _name in zip(_axes, [mean_field_surrogate_posterior, full_rank_surrogate_posterior], ['Mean-field Approximation', 'Full-rank Approximation']):
        _ax.contour(*_grid, _Z, levels=np.arange(7))
        _ax.plot(*_approx.sample(10000), '.', alpha=0.1)
        _ax.set_title(_name)
    plt.tight_layout()
    plt.savefig('img/chp11/vi_in_tfp.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## VI Deep dive
        """
    )
    return


@app.cell
def _(tfp):
    tfb = tfp.bijectors
    return (tfb,)


@app.cell
def _(target_logprob, tf, tfb, tfd, tfp):
    _event_shape = [(), ()]
    full_rank_surrogate_posterior_1 = tfp.experimental.vi.build_affine_surrogate_posterior(event_shape=_event_shape, operators='tril')
    mean_field_surrogate_posterior_1 = tfd.JointDistributionSequential([tfd.Normal(tf.Variable(0.0), tfp.util.TransformedVariable(1.0, bijector=tfb.Exp())), tfd.Normal(tf.Variable(0.0), tfp.util.TransformedVariable(1.0, bijector=tfb.Exp()))])
    made = tfb.AutoregressiveNetwork(params=2, hidden_units=[10, 10])
    flow_surrogate_posterior = tfd.TransformedDistribution(distribution=tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), sample_shape=[2]), bijector=tfb.Chain([tfb.JointMap([tfb.Reshape([]), tfb.Reshape([])]), tfb.Split([1, 1]), tfb.MaskedAutoregressiveFlow(made)]))
    losses_1 = []
    _posterior_samples = []
    for _approx in [mean_field_surrogate_posterior_1, full_rank_surrogate_posterior_1, flow_surrogate_posterior]:
        _loss = tfp.vi.fit_surrogate_posterior(target_logprob, _approx, num_steps=200, optimizer=tf.optimizers.Adam(0.1), sample_size=5)
        losses_1.append(_loss)
        _posterior_samples.append(_approx.sample(10000))
    return (
        flow_surrogate_posterior,
        full_rank_surrogate_posterior_1,
        losses_1,
        mean_field_surrogate_posterior_1,
    )


@app.cell
def _(losses_1, np, plt):
    plt.plot(np.asarray(losses_1).T)
    plt.legend(['mean-field', 'full-rank', 'flow'])
    return


@app.cell
def _(
    flow_surrogate_posterior,
    full_rank_surrogate_posterior_1,
    mean_field_surrogate_posterior_1,
    np,
    plt,
    target_logprob,
):
    _grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 5, 100))
    _Z = -target_logprob(*_grid)
    _, _axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for _ax, _approx, _name in zip(_axes, [mean_field_surrogate_posterior_1, full_rank_surrogate_posterior_1, flow_surrogate_posterior], ['Mean-field Approximation', 'Full-rank Approximation', 'Flow Approximation']):
        _ax.contour(*_grid, _Z, levels=np.arange(7))
        _ax.plot(*_approx.sample(10000), '.', alpha=0.1)
        _ax.set_title(_name)
    plt.tight_layout()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
