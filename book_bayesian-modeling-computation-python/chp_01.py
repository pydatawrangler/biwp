import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 1: Bayesian Inference
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
    from scipy.stats import entropy
    from scipy.optimize import minimize
    return az, entropy, minimize, np, plt, pm, stats


@app.cell
def _(az, np, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams['figure.dpi'] = 300
    np.random.seed(521)
    viridish = [(0.2823529411764706, 0.11372549019607843, 0.43529411764705883, 1.0),
                (0.1450980392156863, 0.6705882352941176, 0.5098039215686274, 1.0),
                (0.6901960784313725, 0.8666666666666667, 0.1843137254901961, 1.0)]
    return (viridish,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A DIY Sampler, Do Not Try This at Home
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.1
        """
    )
    return


@app.cell
def _(np, plt, stats):
    grid = np.linspace(0, 1, 5000)
    prior = stats.triang.pdf(grid, 0.5)
    likelihood = stats.triang.pdf(0.2, grid)
    posterior = prior * likelihood
    log_prior = np.log(prior)
    log_likelihood = np.log(likelihood)
    log_posterior = log_prior + log_likelihood
    _, _ax = plt.subplots(1, 2, figsize=(10, 4))
    _ax[0].plot(grid, prior, label='prior', lw=2)
    _ax[0].plot(grid, likelihood, label='likelihood', lw=2, color='C2')
    _ax[0].plot(grid, posterior, label='posterior', lw=2, color='C4')
    _ax[0].set_xlabel('θ')
    _ax[0].legend()
    _ax[0].set_yticks([])
    _ax[1].plot(grid, log_prior, label='log-prior', lw=2)
    _ax[1].plot(grid, log_likelihood, label='log-likelihood', lw=2, color='C2')
    _ax[1].plot(grid, log_posterior, label='log-posterior', lw=2, color='C4')
    _ax[1].set_xlabel('θ')
    _ax[1].legend()
    _ax[1].set_yticks([])
    plt.savefig('img/chp01/bayesian_triad.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.1
        """
    )
    return


@app.cell
def _(np, stats):
    def post(θ, Y, α=1, β=1):
        if 0 <= θ <= 1:
            prior = stats.beta(_α, _β).pdf(θ)
            like = stats.bernoulli(θ).pmf(Y).prod()
            prop = like * prior
        else:
            prop = -np.inf
        return prop
    return (post,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.2
        """
    )
    return


@app.cell
def _(stats):
    Y = stats.bernoulli(0.7).rvs(20)
    return (Y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.3
        """
    )
    return


@app.cell
def _(Y, np, post, stats):
    n_iters = 1000
    can_sd = 0.05
    _α = _β = 1
    θ = 0.5
    trace = {'θ': np.zeros(n_iters)}
    p2 = post(θ, Y, _α, _β)
    for iter in range(n_iters):
        θ_can = stats.norm(θ, can_sd).rvs(1)
        p1 = post(θ_can, Y, _α, _β)
        pa = p1 / p2
        if pa > stats.uniform(0, 1).rvs(1):
            θ = θ_can
            p2 = p1
        trace['θ'][iter] = θ
    return (trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.5
        """
    )
    return


@app.cell
def _(az, trace):
    az.summary(trace, kind='stats', round_to=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.4 and Figure 1.2
        """
    )
    return


@app.cell
def _(plt, trace):
    _, _axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharey=True)
    _axes[1].hist(trace['θ'], color='0.5', orientation='horizontal', density=True)
    _axes[1].set_xticks([])
    _axes[0].plot(trace['θ'], '0.5')
    _axes[0].set_ylabel('θ', rotation=0, labelpad=15)
    plt.savefig('img/chp01/traceplot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Say Yes to Automating Inference, Say No to Automated Model Building
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.3
        """
    )
    return


@app.cell
def _(az, plt, trace):
    az.plot_posterior(trace)
    plt.savefig("img/chp01/plot_posterior.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.6
        """
    )
    return


@app.cell
def _(Y, pm):
    with pm.Model() as model:
        θ_1 = pm.Beta('θ', alpha=1, beta=1)
        y_obs = pm.Binomial('y_obs', n=1, p=θ_1, observed=Y)
        idata = pm.sample(1000)
    return idata, model, θ_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.7
        """
    )
    return


@app.cell
def _(model, pm):
    graphviz = pm.model_to_graphviz(model)
    graphviz
    return (graphviz,)


@app.cell
def _(graphviz):
    graphviz.graph_attr.update(dpi="300")
    graphviz.render("img/chp01/BetaBinomModelGraphViz", format="png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A Few Options to Quantify Your Prior Information
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.5
        """
    )
    return


@app.cell
def _(idata, model, pm):
    pred_dists = (pm.sample_prior_predictive(1000, model).prior_predictive["y_obs"].values,
                  pm.sample_posterior_predictive(idata, model).posterior_predictive["y_obs"].values)
    return (pred_dists,)


@app.cell
def _(az, idata, plt, pm, pred_dists, θ_1):
    fig, _axes = plt.subplots(4, 1, figsize=(9, 9))
    for _idx, n_d, dist in zip((1, 3), ('Prior', 'Posterior'), pred_dists):
        az.plot_dist(dist.sum(-1), hist_kwargs={'color': '0.5', 'bins': range(0, 22)}, ax=_axes[_idx])
        _axes[_idx].set_title(f'{n_d} predictive distribution', fontweight='bold')
        _axes[_idx].set_xlim(-1, 21)
        _axes[_idx].set_ylim(0, 0.15)
        _axes[_idx].set_xlabel('number of success')
    az.plot_dist(pm.draw(θ_1, 1000), plot_kwargs={'color': '0.5'}, fill_kwargs={'alpha': 1}, ax=_axes[0])
    _axes[0].set_title('Prior distribution', fontweight='bold')
    _axes[0].set_xlim(0, 1)
    _axes[0].set_ylim(0, 4)
    _axes[0].tick_params(axis='both', pad=7)
    _axes[0].set_xlabel('θ')
    az.plot_dist(idata.posterior['θ'], plot_kwargs={'color': '0.5'}, fill_kwargs={'alpha': 1}, ax=_axes[2])
    _axes[2].set_title('Posterior distribution', fontweight='bold')
    _axes[2].set_xlim(0, 1)
    _axes[2].set_ylim(0, 5)
    _axes[2].tick_params(axis='both', pad=7)
    _axes[2].set_xlabel('θ')
    plt.savefig('img/chp01/Bayesian_quartet_distributions.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.6
        """
    )
    return


@app.cell
def _(Y, az, idata, plt, pred_dists, stats):
    predictions = (stats.binom(n=1, p=idata.posterior['θ'].mean()).rvs((4000, len(Y))), pred_dists[1])
    for d, _c, l in zip(predictions, ('C0', 'C4'), ('posterior mean', 'posterior predictive')):
        _ax = az.plot_dist(d.sum(-1), label=l, figsize=(10, 5), hist_kwargs={'alpha': 0.5, 'color': _c, 'bins': range(0, 22)})
        _ax.set_yticks([])
        _ax.set_xlabel('number of success')
    plt.savefig('img/chp01/predictions_distributions.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.8 and Figure 1.7
        """
    )
    return


@app.cell
def _(np, plt, stats, viridish):
    _, _axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex=True, constrained_layout=True)
    _axes = np.ravel(_axes)
    n_trials = [0, 1, 2, 3, 12, 180]
    success = [0, 1, 1, 1, 6, 59]
    data = zip(n_trials, success)
    beta_params = [(0.5, 0.5), (1, 1), (10, 10)]
    θ_2 = np.linspace(0, 1, 1500)
    for _idx, (N, _y) in enumerate(data):
        s_n = 's' if N > 1 else ''
        for jdx, (a_prior, b_prior) in enumerate(beta_params):
            p_theta_given_y = stats.beta.pdf(θ_2, a_prior + _y, b_prior + N - _y)
            _axes[_idx].plot(θ_2, p_theta_given_y, lw=4, color=viridish[jdx])
            _axes[_idx].set_yticks([])
            _axes[_idx].set_ylim(0, 12)
            _axes[_idx].plot(np.divide(_y, N), 0, color='k', marker='o', ms=12)
            _axes[_idx].set_title(f'{N:4d} trial{s_n} {_y:4d} success')
    plt.savefig('img/chp01/beta_binomial_update.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.8
        """
    )
    return


@app.cell
def _(np, plt):
    θ_3 = np.linspace(0, 1, 100)
    κ = θ_3 / (1 - θ_3)
    _y = 2
    n = 7
    _, _axes = plt.subplots(2, 2, figsize=(10, 5), sharex='col', sharey='row', constrained_layout=False)
    _axes[0, 0].set_title("Jeffreys' prior for Alice")
    _axes[0, 0].plot(θ_3, θ_3 ** (-0.5) * (1 - θ_3) ** (-0.5))
    _axes[1, 0].set_title("Jeffreys' posterior for Alice")
    _axes[1, 0].plot(θ_3, θ_3 ** (_y - 0.5) * (1 - θ_3) ** (n - _y - 0.5))
    _axes[1, 0].set_xlabel('θ')
    _axes[0, 1].set_title("Jeffreys' prior for Bob")
    _axes[0, 1].plot(κ, κ ** (-0.5) * (1 + κ) ** (-1))
    _axes[1, 1].set_title("Jeffreys' posterior for Bob")
    _axes[1, 1].plot(κ, κ ** (_y - 0.5) * (1 + κ) ** (-n - 1))
    _axes[1, 1].set_xlim(-0.5, 10)
    _axes[1, 1].set_xlabel('κ')
    _axes[1, 1].text(-4.0, 0.03, size=18, s='$p(\\theta \\mid Y) \\, \\frac{d\\theta}{d\\kappa}$')
    _axes[1, 1].annotate('', xy=(-0.5, 0.025), xytext=(-4.5, 0.025), arrowprops=dict(facecolor='black', shrink=0.05))
    _axes[1, 1].text(-4.0, 0.007, size=18, s='$p(\\kappa \\mid Y) \\, \\frac{d\\kappa}{d\\theta}$')
    _axes[1, 1].annotate('', xy=(-4.5, 0.015), xytext=(-0.5, 0.015), arrowprops=dict(facecolor='black', shrink=0.05), annotation_clip=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.savefig('img/chp01/Jeffrey_priors.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.9
        """
    )
    return


@app.cell
def _(entropy, minimize, np, plt, viridish):
    cons = [[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}], [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: 1.5 - np.sum(x * np.arange(1, 7))}], [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: np.sum(x[[2, 3]]) - 0.8}]]
    max_ent = []
    for _i, _c in enumerate(cons):
        val = minimize(lambda x: -entropy(x), x0=[1 / 6] * 6, bounds=[(0.0, 1.0)] * 6, constraints=_c)['x']
        max_ent.append(entropy(val))
        plt.plot(np.arange(1, 7), val, 'o--', color=viridish[_i], lw=2.5)
    plt.xlabel('$t$')
    plt.ylabel('$p(t)$')
    plt.savefig('img/chp01/max_entropy.png')
    return (max_ent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 1.10
        """
    )
    return


@app.cell
def _(entropy, np):
    ite = 100000
    entropies = np.zeros((3, ite))
    for _idx in range(ite):
        rnds = np.zeros(6)
        total = 0
        x_ = np.random.choice(np.arange(1, 7), size=6, replace=False)
        for _i in x_[:-1]:
            rnd = np.random.uniform(0, 1 - total)
            rnds[_i - 1] = rnd
            total = rnds.sum()
        rnds[-1] = 1 - rnds[:-1].sum()
        H = entropy(rnds)
        entropies[0, _idx] = H
        if abs(1.5 - np.sum(rnds * x_)) < 0.01:
            entropies[1, _idx] = H
        prob_34 = sum(rnds[np.argwhere((x_ == 3) | (x_ == 4)).ravel()])
        if abs(0.8 - prob_34) < 0.01:
            entropies[2, _idx] = H
    return (entropies,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.10
        """
    )
    return


@app.cell
def _(az, entropies, max_ent, np, plt, viridish):
    _, _ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, constrained_layout=True)
    for _i in range(3):
        az.plot_kde(entropies[_i][np.nonzero(entropies[_i])], ax=_ax[_i], plot_kwargs={'color': viridish[_i], 'lw': 4})
        _ax[_i].axvline(max_ent[_i], 0, 1, ls='--')
        _ax[_i].set_yticks([])
        _ax[_i].set_xlabel('entropy')
    plt.savefig('img/chp01/max_entropy_vs_random_dist.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 1.11
        """
    )
    return


@app.cell
def _(np, plt, stats):
    x = np.linspace(0, 1, 500)
    params = [(0.5, 0.5), (1, 1), (3, 3), (100, 25)]
    labels = ['Jeffreys', 'MaxEnt', 'Weakly  Informative', 'Informative']
    _, _ax = plt.subplots()
    for (_α, _β), label, _c in zip(params, labels, (0, 1, 4, 2)):
        pdf = stats.beta.pdf(x, _α, _β)
        _ax.plot(x, pdf, label=f'{label}', c=f'C{_c}', lw=3)
        _ax.set(yticks=[], xlabel='θ', title='Priors')
        _ax.legend()
    plt.savefig('img/chp01/prior_informativeness_spectrum.png')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
