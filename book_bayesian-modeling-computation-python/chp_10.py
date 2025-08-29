import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 10: Probabilistic Programming Languages
        """
    )
    return


@app.cell
def _():
    from scipy import stats
    import pymc as pm
    import pytensor
    import numpy as np
    import matplotlib.pyplot as plt
    import arviz as az

    import datetime
    print(f"Last Run {datetime.datetime.now()}")
    return az, np, plt, pm, pytensor, stats


@app.cell
def _(az, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams["figure.dpi"] = 300
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Posterior Computation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.1
        """
    )
    return


@app.cell
def _():
    from jax import grad

    simple_grad = grad(lambda x: x**2)
    print(simple_grad(4.0))
    return (grad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.2
        """
    )
    return


@app.cell
def _(grad):
    from jax.scipy.stats import norm

    def model(test_point, observed):
        z_pdf = norm.logpdf(test_point, loc=0, scale=5)
        x_pdf = norm.logpdf(observed, loc=test_point, scale=1)
        logpdf = z_pdf + x_pdf
        return logpdf
    model_grad = grad(model)
    observed, test_point = (5.0, 2.5)
    logp_val = model(test_point, observed)
    grad_1 = model_grad(test_point, observed)
    print(f'log_p_val: {logp_val}')
    print(f'grad: {grad_1}')
    return observed, test_point


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.3
        """
    )
    return


@app.cell
def _(np, observed, pm, test_point):
    with pm.Model() as model_1:
        z = pm.Normal('z', 0.0, 5.0)
        x = pm.Normal('x', mu=z, sigma=1.0, observed=observed)
    func = model_1.logp_dlogp_function()
    func.set_extra_values({})
    print(func(np.array([test_point])))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.4
        """
    )
    return


@app.cell
def _():
    def fraud_detector(fraud_observations, non_fraud_observations, fraud_prior=8, non_fraud_prior=6):
        """Conjugate Beta Binomial model for fraud detection"""
        expectation = (fraud_prior + fraud_observations) / (
            fraud_prior + fraud_observations + non_fraud_prior + non_fraud_observations)
    
        if expectation > .5:
            return {"suspend_card":True}

    # magic command not supported in marimo; please file an issue to add support
    # %timeit fraud_detector(2, 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## PPL Driven Transformations
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.9
        """
    )
    return


@app.cell
def _(np, stats):
    observed_1 = np.repeat(2, 2)
    pdf = stats.norm(0, 1).pdf(observed_1)
    np.prod(pdf, axis=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.10
        """
    )
    return


@app.cell
def _(np, stats):
    observed_2 = np.repeat(2, 1000)
    pdf_1 = stats.norm(0, 1).pdf(observed_2)
    np.prod(pdf_1, axis=0)
    return observed_2, pdf_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.11
        """
    )
    return


@app.cell
def _():
    1.2 - 1
    return


@app.cell
def _(np, pdf_1):
    (pdf_1[0], np.prod(pdf_1, axis=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.12
        """
    )
    return


@app.cell
def _(np, observed_2, pdf_1, stats):
    logpdf = stats.norm(0, 1).logpdf(observed_2)
    (np.log(pdf_1[0]), logpdf[0], logpdf.sum())
    return


@app.cell
def _(np, pdf_1):
    np.log(pdf_1[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Distribution Transforms
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.13
        """
    )
    return


@app.cell
def _(np):
    lower, upper = -1, 2
    domain = np.linspace(lower, upper, 5)
    transform = np.log(domain - lower) - np.log(upper - domain)
    print(f"Original domain: {domain}")
    print(f"Transformed domain: {transform}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.14
        """
    )
    return


@app.cell
def _(pm):
    with pm.Model() as model_2:
        x_1 = pm.Uniform('x', -1.0, 2.0)
    model_2.values_to_rvs
    return (model_2,)


@app.cell
def _(model_2):
    model_2.varlogp_nojac.eval
    return


@app.cell
def _(model_2):
    print(model_2.varlogp.eval({'x_interval__': -2}), model_2.varlogp_nojac.eval({'x_interval__': -2}))
    print(model_2.varlogp.eval({'x_interval__': 1}), model_2.varlogp_nojac.eval({'x_interval__': 1}))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.16
        """
    )
    return


@app.cell
def _():
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    return tfd, tfp


@app.cell
def _(np, tfd, tfp):
    tfb = tfp.bijectors
    lognormal0 = tfd.LogNormal(0.0, 1.0)
    lognormal1 = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tfb.Exp())
    x_2 = lognormal0.sample(100)
    np.testing.assert_array_equal(lognormal0.log_prob(x_2), lognormal1.log_prob(x_2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.17
        """
    )
    return


@app.cell
def _(pm, stats):
    y_observed = stats.norm(0, 0.01).rvs(20)
    with pm.Model() as model_transform:
        _sd = pm.HalfNormal('sd', 5)
        y = pm.Normal('y', mu=0, sigma=_sd, observed=y_observed)
        idata_transform = pm.sample(chains=1, draws=100000)
    print(model_transform.values_to_rvs)
    print(f"Diverging: {idata_transform.sample_stats['diverging'].sum().item()}")
    return (y_observed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.18
        """
    )
    return


@app.cell
def _(pm, y_observed):
    with pm.Model() as model_no_transform:
        _sd = pm.HalfNormal('sd', 5, transform=None)
        y_1 = pm.Normal('y', mu=0, sigma=_sd, observed=y_observed)
        idata_no_transform = pm.sample(chains=1, draws=100000)
    print(model_no_transform.values_to_rvs)
    print(f"Diverging: {idata_no_transform.sample_stats['diverging'].sum().item()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Operation Graphs and Automatic Reparameterization
        """
    )
    return


@app.cell
def _():
    x_3 = 3
    y_2 = 1
    x_3 * y_2 / x_3 + 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.19
        """
    )
    return


@app.cell
def _(pytensor):
    pytensor.config.compute_test_value = 'ignore'
    return


@app.cell
def _(pytensor):
    x_4 = pytensor.tensor.vector('x')
    y_3 = pytensor.tensor.vector('y')
    out = x_4 * (y_3 / x_4) + 0
    pytensor.printing.debugprint(out)
    return out, x_4, y_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.20
        """
    )
    return


@app.cell
def _(out, pytensor, x_4, y_3):
    fgraph = pytensor.function([x_4, y_3], [out])
    pytensor.printing.debugprint(fgraph)
    return (fgraph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.21
        """
    )
    return


@app.cell
def _(fgraph):
    fgraph([1],[3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 10.1 and Figure 10.2
        """
    )
    return


@app.cell
def _(fgraph, out, pytensor):
    pytensor.printing.pydotprint(
        out, outfile="img/chp10/symbolic_graph_unopt.png",
        var_with_name_simple=False, high_contrast=False, with_ids=True)
    pytensor.printing.pydotprint(
        fgraph, 
        outfile="img/chp10/symbolic_graph_opt.png", 
        var_with_name_simple=False, high_contrast=False, with_ids=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.22
        """
    )
    return


@app.cell
def _(pm, pytensor):
    with pm.Model() as model_normal:
        x_5 = pm.Normal('x', 0.0, 1.0)
    pytensor.printing.debugprint(model_normal.logp())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Effect handling
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.23
        """
    )
    return


@app.cell
def _():
    import jax
    import numpyro
    from tensorflow_probability.substrates import jax as tfp_jax

    tfp_dist = tfp_jax.distributions
    numpyro_dist = numpyro.distributions

    root = tfp_dist.JointDistributionCoroutine.Root
    def tfp_model():
        x = yield root(tfp_dist.Normal(loc=1.0, scale=2.0, name="x"))
        z = yield root(tfp_dist.HalfNormal(scale=1., name="z"))
        y = yield tfp_dist.Normal(loc=x, scale=z, name="y")
    
    def numpyro_model():
        x = numpyro.sample("x", numpyro_dist.Normal(loc=1.0, scale=2.0))
        z = numpyro.sample("z", numpyro_dist.HalfNormal(scale=1.0))
        y = numpyro.sample("y", numpyro_dist.Normal(loc=x, scale=z))
    return jax, numpyro, numpyro_model, tfp_dist, tfp_model


@app.cell
def _(tfp_model):
    try:
        print(tfp_model())
    except:
        pass
    return


@app.cell
def _(numpyro_model):
    try:
        print(numpyro_model())
    except:
        pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.24
        """
    )
    return


@app.cell
def _(jax, np, numpyro, numpyro_model, tfp_dist, tfp_model):
    sample_key = jax.random.PRNGKey(52346)
    jd = tfp_dist.JointDistributionCoroutine(tfp_model)
    tfp_sample = jd.sample(1, seed=sample_key)
    _predictive = numpyro.infer.Predictive(numpyro_model, num_samples=1)
    numpyro_sample = _predictive(sample_key)
    log_likelihood_tfp = jd.log_prob(tfp_sample)
    log_likelihood_numpyro = numpyro.infer.util.log_density(numpyro_model, [], {}, params=tfp_sample._asdict())
    np.testing.assert_allclose(log_likelihood_tfp, log_likelihood_numpyro[0], rtol=1e-06)
    return jd, sample_key


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.25
        """
    )
    return


@app.cell
def _(jd, sample_key):
    # Condition z to .01 in TFP and sample
    jd.sample(z=.01, seed=sample_key)
    return


@app.cell
def _(np, numpyro, numpyro_model, sample_key):
    _predictive = numpyro.infer.Predictive(numpyro_model, num_samples=1, params={'z': np.asarray(0.01)})
    _predictive(sample_key)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.26
        """
    )
    return


@app.cell
def _(jd, numpyro, numpyro_model, sample_key):
    dist, value = jd.sample_distributions(z=0.01, seed=sample_key)
    assert dist.y.loc == value.x
    assert dist.y.scale == value.z
    model_3 = numpyro.handlers.substitute(numpyro_model, data={'z': 0.01})
    with numpyro.handlers.seed(rng_seed=sample_key):
        model_trace = numpyro.handlers.trace(numpyro_model).get_trace()
    assert model_trace['y']['fn'].loc == model_trace['x']['value']
    assert model_trace['y']['fn'].scale == model_trace['z']['value']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Designing a PPL
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.27
        """
    )
    return


@app.cell
def _(stats):
    x_6 = stats.norm.rvs(loc=1.0, scale=2.0, size=2, random_state=1234)
    _logp = stats.norm.logpdf(x_6, loc=1.0, scale=2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.28
        """
    )
    return


@app.cell
def _(stats):
    random_variable_x = stats.norm(loc=1.0, scale=2.0)
    x_7 = random_variable_x.rvs(size=2, random_state=1234)
    _logp = random_variable_x.logpdf(x_7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.29
        """
    )
    return


@app.cell
def _(stats):
    x_8 = stats.norm(loc=1.0, scale=2.0)
    y_4 = stats.norm(loc=x_8, scale=0.1)
    y_4.rvs()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.30
        """
    )
    return


@app.cell
def _(np, stats):
    class _RandomVariable:

        def __init__(self, distribution):
            self.distribution = distribution

        def __array__(self):
            return np.asarray(self.distribution.rvs())
    x_9 = _RandomVariable(stats.norm(loc=1.0, scale=2.0))
    z_1 = _RandomVariable(stats.halfnorm(loc=0.0, scale=1.0))
    y_5 = _RandomVariable(stats.norm(loc=x_9, scale=z_1))
    for _i in range(5):
        print(np.asarray(y_5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.31
        """
    )
    return


@app.cell
def _(np, stats):
    class _RandomVariable:

        def __init__(self, distribution, value=None):
            self.distribution = distribution
            self.set_value(value)

        def __repr__(self):
            return f'{self.__class__.__name__}(value={self.__array__()})'

        def __array__(self, dtype=None):
            if self.value is None:
                return np.asarray(self.distribution.rvs(), dtype=dtype)
            return self.value

        def set_value(self, value=None):
            self.value = value

        def log_prob(self, value=None):
            if value is not None:
                self.set_value(value)
            return self.distribution.logpdf(np.array(self))
    x_10 = _RandomVariable(stats.norm(loc=1.0, scale=2.0))
    z_2 = _RandomVariable(stats.halfnorm(loc=0.0, scale=1.0))
    y_6 = _RandomVariable(stats.norm(loc=x_10, scale=z_2))
    return x_10, y_6, z_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.32
        """
    )
    return


@app.cell
def _(np, x_10, y_6, z_2):
    for _i in range(3):
        print(y_6)
    print(f'  Set x=5 and z=0.1')
    x_10.set_value(np.asarray(5))
    z_2.set_value(np.asarray(0.05))
    for _i in range(3):
        print(y_6)
    print(f'  Reset z')
    z_2.set_value(None)
    for _i in range(3):
        print(y_6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.33
        """
    )
    return


@app.cell
def _(np, x_10, y_6, z_2):
    y_6.set_value(np.array(5.0))
    posterior_density = lambda xval, zval: x_10.log_prob(xval) + z_2.log_prob(zval) + y_6.log_prob()
    posterior_density(np.array(0.0), np.array(1.0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.34
        """
    )
    return


@app.cell
def _(stats):
    def log_prob(xval, zval, yval=5):
        x_dist = stats.norm(loc=1.0, scale=2.0)
        z_dist = stats.halfnorm(loc=0., scale=1.)
        y_dist = stats.norm(loc=xval, scale=zval)
        return x_dist.logpdf(xval) + z_dist.logpdf(zval) + y_dist.logpdf(yval)

    log_prob(0, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.35
        """
    )
    return


@app.cell
def _(stats):
    def _prior_sample():
        x = stats.norm(loc=1.0, scale=2.0).rvs()
        z = stats.halfnorm(loc=0.0, scale=1.0).rvs()
        y = stats.norm(loc=x, scale=z).rvs()
        return (x, z, y)
    _prior_sample()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Shape handling
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.36
        """
    )
    return


@app.cell
def _(stats):
    def _prior_sample(size):
        x = stats.norm(loc=1.0, scale=2.0).rvs(size=size)
        z = stats.halfnorm(loc=0.0, scale=1.0).rvs(size=size)
        y = stats.norm(loc=x, scale=z).rvs()
        return (x, z, y)
    print([x.shape for x in _prior_sample(size=2)])
    print([x.shape for x in _prior_sample(size=(2, 3, 5))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.37
        """
    )
    return


@app.cell
def _(np, stats):
    n_row, n_feature = 1000, 5
    X = np.random.randn(n_row, n_feature)

    def lm_prior_sample0():
        intercept = stats.norm(loc=0, scale=10.0).rvs()
        beta = stats.norm(loc=np.zeros(n_feature), scale=10.0).rvs()
        sigma = stats.halfnorm(loc=0., scale=1.).rvs()
        y_hat = X @ beta + intercept
        y = stats.norm(loc=y_hat, scale=sigma).rvs()
        return intercept, beta, sigma, y

    def lm_prior_sample(size=10):
        if isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)
        intercept = stats.norm(loc=0, scale=10.0).rvs(size=size)
        beta = stats.norm(loc=np.zeros(n_feature), scale=10.0).rvs(
            size=size + (n_feature,))
        sigma = stats.halfnorm(loc=0., scale=1.).rvs(size=size)
        y_hat = np.squeeze(X @ beta[..., None]) + intercept[..., None]
        y = stats.norm(loc=y_hat, scale=sigma[..., None]).rvs()
        return intercept, beta, sigma, y
    return X, lm_prior_sample, lm_prior_sample0, n_feature


@app.cell
def _(lm_prior_sample0):
    print([x.shape for x in lm_prior_sample0()])
    return


@app.cell
def _(lm_prior_sample):
    print([x.shape for x in lm_prior_sample(size=())])
    print([x.shape for x in lm_prior_sample(size=10)])
    print([x.shape for x in lm_prior_sample(size=[10, 3])])
    return


@app.cell
def _():
    # def lm_prior_sample2(size=10):
    #     intercept = stats.norm(loc=0, scale=10.0).rvs(size=size)
    #     beta = stats.multivariate_normal(
    #         mean=np.zeros(n_feature), cov=10.0).rvs(size=size)
    #     sigma = stats.halfnorm(loc=0., scale=1.).rvs(size=size)
    #     y_hat = np.einsum('ij,...j->...i', X, beta) + intercept[..., None]
    #     y = stats.norm(loc=y_hat, scale=sigma[..., None]).rvs()
    #     return intercept, beta, sigma, y

    # print([x.shape for x in lm_prior_sample2(size=())])
    # print([x.shape for x in lm_prior_sample2(size=10)])
    # print([x.shape for x in lm_prior_sample2(size=(10, 3))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 10.38
        """
    )
    return


@app.cell
def _(X, n_feature, tfp):
    import tensorflow as tf
    tfd_1 = tfp.distributions
    jd_1 = tfd_1.JointDistributionSequential([tfd_1.Normal(0, 10), tfd_1.Sample(tfd_1.Normal(0, 10), n_feature), tfd_1.HalfNormal(1), lambda sigma, beta, intercept: tfd_1.Independent(tfd_1.Normal(loc=tf.einsum('ij,...j->...i', X, beta) + intercept[..., None], scale=sigma[..., None]), reinterpreted_batch_ndims=1, name='y')])
    print(jd_1)
    n_sample = [3, 2]
    for log_prob_part in jd_1.log_prob_parts(jd_1.sample(n_sample)):
        assert log_prob_part.shape == n_sample
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
