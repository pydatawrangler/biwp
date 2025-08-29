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
    from scipy.special import binom, beta
    import preliz as pz
    return az, beta, binom, np, plt, pz


@app.function
def P(S, A):
    if set(A).issubset(set(S)):
        return len(A)/len(S)
    else:
        return 0


@app.cell
def _(az, np, plt):
    az.style.use("arviz-grayscale")
    np.random.seed(314)
    from cycler import cycler
    default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('figure', dpi=300)
    return


@app.cell
def _(beta, binom, np, plt):
    n = 5
    x = np.arange(0, 6)

    fig, axes = plt.subplots(2, 2, figsize=(10, 4), sharey=True, sharex=True)
    axes = np.ravel(axes)

    for ax, (α, β) in zip(axes, ((1, 1), (5, 2), (0.5, 0.5), (20, 20))): 
        dist_pmf = binom(n, x) * (beta(x+α, n-x+β) / beta(α, β))
        ax.vlines(x, 0, dist_pmf, colors='C0', lw=4)
        ax.set_title(f"α={α}, β={β}")
        ax.set_xticks(x)
        ax.set_xticklabels(x+1)
    fig.text(0.52, -0.04, "x", fontsize=18)
    fig.text(-0.04, 0.4, "P(X=x)", fontsize=18, rotation="vertical")
    # plt.savefig("fig/dice_distribution.png", bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(pz):
    pmfs = pz.BetaBinomial(alpha=10, beta=10, n=5).pdf(range(6))
    return (pmfs,)


@app.cell
def _(pmfs):
    [(i, f"{x:0.3f}") for i, x in enumerate(pmfs)]
    return


@app.cell
def _(pz):
    pz.BetaBinomial(alpha=10, beta=10, n=6).rvs()
    return


@app.cell
def _(plt, pz):
    plt.hist(pz.BetaBinomial(alpha=2, beta=5, n=5).rvs(1000),
            bins=[0, 1, 2, 3, 4, 5, 6], density=True, align="left", color="C2")
    pz.BetaBinomial(alpha=2, beta=5, n=5).plot_pdf()
    plt.show();
    return


@app.cell
def _(plt, pz):
    mus = [0., 0., -2.]
    sigmas = [1, 0.5, 1]
    for mu, sigma in zip(mus, sigmas):
        ax2 = pz.Normal(mu, sigma).plot_pdf()
    [line.set_linewidth(3.) for line in ax2.get_lines()[1::2]]
    plt.show()
    return


@app.cell
def _(np, plt, pz):
    dist = pz.Normal(0, 1)
    ax3 = dist.plot_pdf()
    x_s = np.linspace(-2, 0)
    ax3.fill_between(x_s, dist.pdf(x_s), color="C2")
    # plt.savefig("fig/gauss_prob.png")
    plt.show()
    dist.cdf(0) - dist.cdf(-2)
    return


@app.cell
def _(plt, pz):
    _, ax4 = plt.subplots(2, 2, figsize=(12, 5), sharex="col")
    pz.BetaBinomial(alpha=10, beta=10, n=5).plot_pdf(ax=ax4[0, 0], legend="title")
    pz.BetaBinomial(alpha=10, beta=10, n=5).plot_cdf(ax=ax4[1, 0], legend=None)
    pz.Normal(0, 1).plot_pdf(ax=ax4[0, 1], legend="title")
    pz.Normal(0, 1).plot_cdf(ax=ax4[1, 1], legend=None)
    plt.show()
    return


@app.cell
def _(az, np, pz):
    np.random.seed(1)
    az.plot_posterior({'theta': pz.Beta(4, 12).rvs(1000)})
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
