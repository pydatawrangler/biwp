import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 5: Splines
        """
    )
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import warnings
    warnings.filterwarnings("ignore", message="hdi currently interprets 2d data as")

    import arviz as az
    import matplotlib.pyplot as plt
    from cycler import cycler
    import numpy as np
    import pandas as pd
    import pymc as pm
    from patsy import bs, dmatrix

    from scripts.splines import splines
    return az, bs, cycler, dmatrix, np, pd, plt, pm, splines, warnings


@app.cell
def _(az, np, plt):
    az.style.use('arviz-grayscale')
    plt.rcParams["figure.dpi"] = 300
    np.random.seed(435)
    viridish = [(0.2823529411764706, 0.11372549019607843, 0.43529411764705883, 1.0),
                (0.1843137254901961, 0.4196078431372549, 0.5568627450980392, 1.0),
                (0.1450980392156863, 0.6705882352941176, 0.5098039215686274, 1.0),
                (0.6901960784313725, 0.8666666666666667, 0.1843137254901961, 1.0),
                (0.2823529411764706, 0.11372549019607843, 0.43529411764705883, 0.5),
                (0.1843137254901961, 0.4196078431372549, 0.5568627450980392, 0.5),
                (0.1450980392156863, 0.6705882352941176, 0.5098039215686274, 0.5),
                (0.6901960784313725, 0.8666666666666667, 0.1843137254901961, 0.5),
                (0.2823529411764706, 0.11372549019607843, 0.43529411764705883, 0.3),
                (0.1843137254901961, 0.4196078431372549, 0.5568627450980392, 0.3),
                (0.1450980392156863, 0.6705882352941176, 0.5098039215686274, 0.3),
                (0.6901960784313725, 0.8666666666666667, 0.1843137254901961, 0.3)]
    return (viridish,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Polynomial Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.1
        """
    )
    return


@app.cell
def _(np):
    x = np.random.normal(0.5, 1, 50)
    y = np.random.normal(x**2, 1)
    return x, y


@app.cell
def _(np, plt, warnings, x, y):
    x_ = np.linspace(x.min(), x.max(), 500)
    warnings.filterwarnings('ignore', message='hdi currently interprets 2d data as')
    _, _axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    for _deg, _ax in zip([2, 10, 15], _axes):
        _ax.plot(x, y, 'C2.')
        _ax.set_title(f'Degree={_deg}')
        coefs = np.polyfit(x, y, deg=_deg)
        ffit = np.poly1d(coefs)
        _ax.plot(x_, ffit(x_), color='C0', lw=2)
        coefs = np.polyfit(x[:-1], y[:-1], deg=_deg)
        ffit = np.poly1d(coefs)
        _ax.plot(x_, ffit(x_), color='C0', lw=2, ls='--')
        _ax.plot(x[0], y[0], 'C0X', color='C4')
    _axes[1].set_xlabel('x', labelpad=10)
    _axes[0].set_ylabel('f(x)', rotation=0, labelpad=20)
    _ax.set_xticks([])
    _ax.set_yticks([])
    plt.savefig('img/chp05/polynomial_regression.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expanding the Feature Space
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.2
        """
    )
    return


@app.cell
def _(splines):
    splines([1.57, 4.71])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introducing Splines
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.3
        """
    )
    return


@app.cell
def _(bs, np, plt, viridish):
    x_1 = np.linspace(-0.0001, 1, 1000)
    knots = [0, 0.2, 0.4, 0.6, 0.8, 1]
    _, _axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True, sharey=True)
    for _deg, _ax in enumerate(_axes):
        b_splines = bs(x_1, degree=_deg, knots=knots, lower_bound=-0.01, upper_bound=1.01)
        for enu, b_s in enumerate(b_splines.T):
            _ax.plot(x_1, b_s, color=viridish[enu], lw=2, ls='--')
        _ax.plot(x_1, b_splines[:, _deg], lw=3)
        _ax.plot(knots, np.zeros_like(knots), 'ko', mec='w', ms=10)
        for _i in range(1, _deg + 1):
            _ax.plot([0, 1], np.array([0, 0]) - _i / 15, 'k.', clip_on=False)
        _ax.plot(knots[:_deg + 2], np.zeros_like(knots[:_deg + 2]), 'C4o', mec='w', ms=10)
    plt.ylim(0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('img/chp05/splines_basis.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Building the Design Matrix using Patsy
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.4 and Code 5.1
        """
    )
    return


@app.cell
def _(dmatrix, np):
    x_2 = np.linspace(0.0, 1.0, 500)
    knots_1 = [0.25, 0.5, 0.75]
    B0 = dmatrix('bs(x, knots=knots, degree=0, include_intercept=True) - 1', {'x': x_2, 'knots': knots_1})
    B1 = dmatrix('bs(x, knots=knots, degree=1, include_intercept=True) - 1', {'x': x_2, 'knots': knots_1})
    B3 = dmatrix('bs(x, knots=knots, degree=3,include_intercept=True) - 1', {'x': x_2, 'knots': knots_1})
    return B0, B1, B3, knots_1, x_2


@app.cell
def _(B0, B1, B3, knots_1, np, plt, viridish, x_2):
    np.random.seed(1563)
    _, _axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey='row')
    for _idx, (B, _title) in enumerate(zip((B0, B1, B3), ('Piecewise constant', 'Piecewise linear', 'Cubic spline'))):
        for _i in range(B.shape[1]):
            _axes[0, _idx].plot(x_2, B[:, _i], color=viridish[_i], lw=2, ls='--')
        _β = np.abs(np.random.normal(0, 1, size=B.shape[1]))
        for _i in range(B.shape[1]):
            _axes[1, _idx].plot(x_2, B[:, _i] * _β[_i], color=viridish[_i], lw=2, ls='--')
        _axes[1, _idx].plot(x_2, np.dot(B, _β), color='k', lw=3)
        _axes[0, _idx].plot(knots_1, np.zeros_like(knots_1), 'ko')
        _axes[1, _idx].plot(knots_1, np.zeros_like(knots_1), 'ko')
        _axes[0, _idx].set_title(_title)
    plt.savefig('img/chp05/splines_weighted.png')
    return (B,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.5
        """
    )
    return


@app.cell
def _(B, np, plt, x_2):
    _, _axes = plt.subplots(1, 1, figsize=(10, 4))
    for _i in range(4):
        _β = np.abs(np.random.normal(0, 1, size=B.shape[1]))
        _axes.plot(x_2, np.dot(B, _β), color=f'C{_i}', lw=3)
        _axes.set_title('4 realizations of cubic splines')
    plt.savefig('img/chp05/splines_realizations.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.6
        """
    )
    return


@app.cell
def _(dmatrix, np):
    x_3 = np.linspace(0.0, 1.0, 20)
    knots_2 = [0.25, 0.5, 0.75]
    B0_1 = dmatrix('bs(x, knots=knots, degree=0, include_intercept=True) - 1', {'x': x_3, 'knots': knots_2})
    B1_1 = dmatrix('bs(x, knots=knots, degree=1, include_intercept=True) - 1', {'x': x_3, 'knots': knots_2})
    B3_1 = dmatrix('bs(x, knots=knots, degree=3, include_intercept=True) - 1', {'x': x_3, 'knots': knots_2})
    return B0_1, B1_1, B3_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.2
        """
    )
    return


@app.cell
def _(B0_1, B1_1, B3_1, np, plt):
    _fig, _axes = plt.subplots(1, 3, sharey=True)
    for _idx, (B_1, _title, _ax) in enumerate(zip((B0_1, B1_1, B3_1), ('Piecewise constant', 'Piecewise linear', 'Cubic spline'), _axes)):
        cax = _ax.imshow(B_1, cmap='cet_gray_r', aspect='auto')
        _ax.set_xticks(np.arange(B_1.shape[1]))
        _ax.set_yticks(np.arange(B_1.shape[0]))
        _ax.spines['left'].set_visible(False)
        _ax.spines['bottom'].set_visible(False)
        _ax.set_title(_title)
    _axes[1].set_xlabel('b-splines')
    _axes[0].set_ylabel('x', rotation=0, labelpad=15)
    _fig.colorbar(cax, aspect=40, ticks=[0, 0.5, 1])
    plt.savefig('img/chp05/design_matrices.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Fitting Splines in PyMC3
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.3 and Figure 5.7
        """
    )
    return


@app.cell
def _(pd):
    data = pd.read_csv("../data/bikes_hour.csv")
    data.sort_values(by="hour", inplace=True)

    # We standardize the response variable
    data_cnt_om = data["count"].mean()
    data_cnt_os = data["count"].std()
    data["count_normalized"] = (data["count"] - data_cnt_om) / data_cnt_os
    # Remove data, you may later try to refit the model to the whole data
    data = data[::50]
    return data, data_cnt_om, data_cnt_os


@app.cell
def _(data, plt):
    _, _ax = plt.subplots(1, 1, figsize=(10, 4))
    _ax.plot(data.hour, data.count_normalized, 'o', alpha=0.3)
    _ax.set_xlabel('hour')
    _ax.set_ylabel('count_normalized')
    plt.savefig('img/chp05/bikes_data.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.4
        """
    )
    return


@app.cell
def _(np):
    num_knots = 6
    knot_list = np.linspace(0, 23, num_knots)[1:-1]
    return (knot_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.5
        """
    )
    return


@app.cell
def _(data, dmatrix, knot_list):
    B_2 = dmatrix('bs(cnt, knots=knots, degree=3, include_intercept=True) - 1', {'cnt': data.hour.values, 'knots': knot_list})
    return (B_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.6
        """
    )
    return


@app.cell
def _(B_2, data, np, pm):
    with pm.Model() as splines_1:
        _τ = pm.HalfCauchy('τ', 1)
        _β = pm.Normal('β', mu=0, sigma=_τ, shape=B_2.shape[1])
        _μ = pm.Deterministic('μ', pm.math.dot(np.asfortranarray(B_2), _β))
        _σ = pm.HalfNormal('σ', 1)
        _c = pm.Normal('c', _μ, _σ, observed=data['count_normalized'])
        idata_s = pm.sample(1000)
    return (idata_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.8
        """
    )
    return


@app.cell
def _(
    B_2,
    cycler,
    data,
    data_cnt_om,
    data_cnt_os,
    idata_s,
    knot_list,
    np,
    plt,
    viridish,
):
    _, _ax = plt.subplots(1, 1, figsize=(10, 4))
    _ax.set_prop_cycle(cycler('color', viridish))
    posterior = idata_s.posterior.stack(samples=['chain', 'draw'])
    _ax.plot(data.hour, B_2 * posterior['β'].mean('samples').values * data_cnt_os + data_cnt_om, lw=2, ls='--')
    _ax.plot(data.hour, posterior['μ'].mean('samples') * data_cnt_os + data_cnt_om, 'k', lw=3)
    _ax.set_xlabel('hour')
    _ax.set_ylabel('count')
    _ax.plot(knot_list, np.zeros_like(knot_list), 'ko')
    plt.savefig('img/chp05/bikes_spline_raw_data.png')
    return (posterior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.9
        """
    )
    return


@app.cell
def _(az, data, data_cnt_om, data_cnt_os, plt, posterior):
    _, _ax = plt.subplots(1, 1, figsize=(10, 4))
    _ax.plot(data.hour, data['count'], 'o', alpha=0.3, zorder=-1)
    _ax.plot(data.hour, posterior['μ'].mean('samples') * data_cnt_os + data_cnt_om, color='C4', lw=2)
    az.plot_hdi(data.hour, posterior['μ'].T * data_cnt_os + data_cnt_om, color='C0', smooth=False)
    _ax.set_xlabel('hour')
    _ax.set_ylabel('count')
    plt.savefig('img/chp05/bikes_spline_data.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Choosing Knots and Prior for Splines
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 5.1 and Figure 5.10
        """
    )
    return


@app.cell
def _(data, dmatrix, np):
    Bs = []
    num_knots_1 = (3, 6, 9, 12, 18)
    for nk in num_knots_1:
        knot_list_1 = np.linspace(0, 24, nk + 2)[1:-1]
        B_3 = dmatrix('bs(cnt, knots=knots, degree=3, include_intercept=True) - 1', {'cnt': data.hour.values, 'knots': knot_list_1})
        Bs.append(B_3)
    return Bs, num_knots_1


@app.cell
def _(Bs, data, np, pm):
    idatas = []
    for B_4 in Bs:
        with pm.Model() as splines_2:
            _τ = pm.HalfCauchy('τ', 1)
            _β = pm.Normal('β', mu=0, sigma=_τ, shape=B_4.shape[1])
            _μ = pm.Deterministic('μ', pm.math.dot(np.asfortranarray(B_4), _β))
            _σ = pm.HalfNormal('σ', 1)
            _c = pm.Normal('c', _μ, _σ, observed=data['count_normalized'].values)
            _idata = pm.sample(1000, idata_kwargs={'log_likelihood': True})
            idatas.append(_idata)
    return (idatas,)


@app.cell
def _(az, idatas, num_knots_1):
    dict_cmp = {f'm_{k}k': v for k, v in zip(num_knots_1, idatas)}
    cmp = az.compare(dict_cmp)
    cmp.round(2)
    return


@app.cell
def _(data, data_cnt_om, data_cnt_os, idatas, plt):
    _, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(data.hour, data['count'], 'o', alpha=0.1, zorder=-1)
    for _idx, (_idata, _i, ls, lw) in enumerate(zip(idatas, (0, 2, 2, 4, 2), ('-', '--', '--', '-', '--'), (3, 1.5, 1.5, 3, 1.5))):
        _mean_f = _idata.posterior['μ'].mean(dim=['chain', 'draw'])
        _ax.plot(data.hour, _mean_f * data_cnt_os + data_cnt_om, color=f'C{_i}', label=f'knots={(3, 6, 9, 12, 18)[_idx]}', ls=ls, lw=lw)
    plt.legend()
    _ax.set_xlabel('hour')
    _ax.set_ylabel('count')
    plt.savefig('img/chp05/bikes_spline_loo_knots.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 5.11
        """
    )
    return


@app.cell
def _(np):
    knot_list_2 = np.arange(1, 23)
    return (knot_list_2,)


@app.cell
def _(data, dmatrix, knot_list_2):
    B_5 = dmatrix('bs(cnt, knots=knots, degree=3, include_intercept=True) - 1', {'cnt': data.hour.values, 'knots': knot_list_2})
    return (B_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.7
        """
    )
    return


@app.cell
def _(B_5, data, np, pm):
    with pm.Model() as _splines_rw:
        _τ = pm.HalfCauchy('τ', 1)
        _β = pm.GaussianRandomWalk('β', mu=0, sigma=_τ, shape=B_5.shape[1])
        _μ = pm.Deterministic('μ', pm.math.dot(np.asfortranarray(B_5), _β))
        _σ = pm.HalfNormal('σ', 1)
        _c = pm.Normal('c', _μ, _σ, observed=data['count_normalized'])
        idata_splines_rw = pm.sample(1000)
        idata_splines_rw.extend(pm.sample_posterior_predictive(idata_splines_rw))
    return (idata_splines_rw,)


@app.cell
def _(B_5, data, np, pm):
    with pm.Model() as wiggly:
        _τ = pm.HalfCauchy('τ', 1)
        _β = pm.Normal('β', mu=0, sigma=_τ, shape=B_5.shape[1])
        _μ = pm.Deterministic('μ', pm.math.dot(np.asfortranarray(B_5), _β))
        _σ = pm.HalfNormal('σ', 1)
        _c = pm.Normal('c', _μ, _σ, observed=data['count_normalized'])
        idata_wiggly = pm.sample(1000)
        idata_wiggly.extend(pm.sample_posterior_predictive(idata_wiggly))
    return (idata_wiggly,)


@app.cell
def _(data, data_cnt_om, data_cnt_os, idata_splines_rw, idata_wiggly, plt):
    _, _ax = plt.subplots(1, 1, figsize=(10, 4))
    _ax.plot(data.hour, data['count'], 'o', alpha=0.1, zorder=-1)
    wiggly_posterior = idata_wiggly.posterior['μ'] * data_cnt_os + data_cnt_om
    _mean_f = wiggly_posterior.mean(dim=['chain', 'draw'])
    _ax.plot(data.hour, _mean_f, color='C0', lw=3)
    _splines_rw = idata_splines_rw.posterior['μ'] * data_cnt_os + data_cnt_om
    _mean_f = _splines_rw.mean(dim=['chain', 'draw'])
    _ax.plot(data.hour, _mean_f, color='C4', lw=3)
    _ax.set_xlabel('hour')
    _ax.set_ylabel('count')
    plt.savefig('img/chp05/bikes_spline_data_grw.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Modeling CO2 Uptake with Splines
        """
    )
    return


@app.cell
def _(np):
    np.random.seed(435)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.8
        """
    )
    return


@app.cell
def _(pd):
    plants_CO2 = pd.read_csv("../data/CO2_uptake.csv")
    plant_names = plants_CO2.Plant.unique()
    CO2_conc = plants_CO2.conc.values[:7]
    CO2_concs = plants_CO2.conc.values
    uptake = plants_CO2.uptake.values
    index = list(range(12))
    groups = len(index)
    return CO2_conc, CO2_concs, groups, index, plant_names, uptake


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.9
        """
    )
    return


@app.cell
def _(CO2_conc, CO2_concs, dmatrix, np):
    num_knots_2 = 2
    knot_list_3 = np.linspace(CO2_conc[0], CO2_conc[-1], num_knots_2 + 2)[1:-1]
    Bg = dmatrix('bs(conc, knots=knots, degree=3, include_intercept=True) - 1', {'conc': CO2_concs, 'knots': knot_list_3})
    return Bg, knot_list_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.10 and Figure 5.12
        """
    )
    return


@app.cell
def _(Bg, np, pm, uptake):
    with pm.Model() as sp_global:
        _τ = pm.HalfCauchy('τ', 1)
        _β = pm.Normal('β', mu=0, sigma=_τ, shape=Bg.shape[1])
        _μg = pm.Deterministic('μg', pm.math.dot(np.asfortranarray(Bg), _β))
        _σ = pm.HalfNormal('σ', 1)
        _up = pm.Normal('up', _μg, _σ, observed=uptake)
        idata_sp_global = pm.sample(3000, tune=2000, idata_kwargs={'log_likelihood': True})
    return (idata_sp_global,)


@app.cell
def _(CO2_conc, az, idata_sp_global, plant_names, plt, uptake):
    _fig, _axes = plt.subplots(4, 3, figsize=(10, 6), sharey=True, sharex=True)
    _μsg = idata_sp_global.posterior.stack(draws=('chain', 'draw'))['μg'].values.T
    _μsg_mean = _μsg.mean(0)
    for _count, (_idx, _ax) in enumerate(zip(range(0, 84, 7), _axes.ravel())):
        _ax.plot(CO2_conc, uptake[_idx:_idx + 7], '.', lw=1)
        _ax.plot(CO2_conc, _μsg_mean[_idx:_idx + 7], 'k', alpha=0.5)
        az.plot_hdi(CO2_conc, _μsg[:, _idx:_idx + 7], color='C2', smooth=False, ax=_ax)
        _ax.set_title(plant_names[_count])
    _fig.text(0.4, -0.05, 'CO2 concentration', size=18)
    _fig.text(-0.03, 0.4, 'CO2 uptake', size=18, rotation=90)
    plt.savefig('sp_global.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.11
        """
    )
    return


@app.cell
def _(CO2_conc, dmatrix, knot_list_3):
    Bi = dmatrix('bs(conc, knots=knots, degree=3, include_intercept=True) - 1', {'conc': CO2_conc, 'knots': knot_list_3})
    return (Bi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.12 and Figure 5.13
        """
    )
    return


@app.cell
def _(Bi, groups, index, np, pm, uptake):
    with pm.Model() as sp_individual:
        _τ = pm.HalfCauchy('τ', 1)
        _βi = pm.Normal('βi', mu=0, sigma=_τ, shape=(Bi.shape[1], groups))
        _μi = pm.Deterministic('μi', pm.math.dot(np.asfortranarray(Bi), _βi))
        _σ = pm.HalfNormal('σ', 1)
        _up = pm.Normal('up', _μi[:, index].T.ravel(), _σ, observed=uptake)
        idata_sp_individual = pm.sample(3000, idata_kwargs={'log_likelihood': True})
    return (idata_sp_individual,)


@app.cell
def _(CO2_conc, az, idata_sp_individual, index, plant_names, plt, uptake):
    _fig, _axes = plt.subplots(4, 3, figsize=(10, 6), sharey=True, sharex=True)
    _μsi = idata_sp_individual.posterior.stack(draws=('chain', 'draw'))['μi'].values.T
    _μsi_mean = _μsi.mean(0)
    for _count, (_idx, _ax) in enumerate(zip(range(0, 84, 7), _axes.ravel())):
        _ax.plot(CO2_conc, uptake[_idx:_idx + 7], '.', lw=1)
        _ax.plot(CO2_conc, _μsi_mean[index[_count]], 'k', alpha=0.5)
        az.plot_hdi(CO2_conc, _μsi[:, index[_count]], color='C2', smooth=False, ax=_ax)
        _ax.set_title(plant_names[_count])
    _fig.text(0.4, -0.075, 'CO2 concentration', size=18)
    _fig.text(-0.03, 0.4, 'CO2 uptake', size=18, rotation=90)
    plt.savefig('sp_individual.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.13 and Figure 5.14
        """
    )
    return


@app.cell
def _(Bg, Bi, groups, index, np, pm, uptake):
    with pm.Model() as sp_mix:
        _τ = pm.HalfCauchy('τ', 1)
        _β = pm.Normal('β', mu=0, sigma=_τ, shape=Bg.shape[1])
        _μg = pm.Deterministic('μg', pm.math.dot(np.asfortranarray(Bg), _β))
        _βi = pm.Normal('βi', mu=0, sigma=_τ, shape=(Bi.shape[1], groups))
        _μi = pm.Deterministic('μi', pm.math.dot(np.asfortranarray(Bi), _βi))
        _σ = pm.HalfNormal('σ', 1)
        _up = pm.Normal('up', _μg + _μi[:, index].T.ravel(), _σ, observed=uptake)
        idata_sp_mix = pm.sample(3000, idata_kwargs={'log_likelihood': True})
    return (idata_sp_mix,)


@app.cell
def _(CO2_conc, az, idata_sp_mix, index, plant_names, plt, uptake):
    _fig, _axes = plt.subplots(4, 3, figsize=(10, 6), sharey=True, sharex=True)
    _μsg = idata_sp_mix.posterior.stack(draws=('chain', 'draw'))['μg'].values.T
    _μsg_mean = _μsg.mean(0)
    _μsi = idata_sp_mix.posterior.stack(draws=('chain', 'draw'))['μi'].values.T
    _μsi_mean = _μsi.mean(0)
    for _count, (_idx, _ax) in enumerate(zip(range(0, 84, 7), _axes.ravel())):
        _ax.plot(CO2_conc, uptake[_idx:_idx + 7], '.', lw=1)
        _ax.plot(CO2_conc, _μsg_mean[_idx:_idx + 7] + _μsi_mean[index[_count]], 'C4', alpha=0.5)
        az.plot_hdi(CO2_conc, _μsg[:, _idx:_idx + 7] + _μsi[:, index[_count]], color='C4', smooth=False, ax=_ax)
        _ax.plot(CO2_conc, _μsg_mean[_idx:_idx + 7], 'k')
        az.plot_hdi(CO2_conc, _μsg[:, _idx:_idx + 7], color='k', smooth=False, ax=_ax)
        _ax.plot(CO2_conc, _μsi_mean[index[_count]], 'k', alpha=0.5)
        az.plot_hdi(CO2_conc, _μsi[:, index[_count]], color='C2', smooth=False, ax=_ax)
        _ax.set_title(plant_names[_count])
    _fig.text(0.4, -0.075, 'CO2 concentration', size=18)
    _fig.text(-0.03, 0.4, 'CO2 uptake', size=18, rotation=90)
    plt.savefig('sp_mix_decomposed.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 5.14 and Figure 5.15
        """
    )
    return


@app.cell
def _(az, idata_sp_global, idata_sp_individual, idata_sp_mix):
    cmp_1 = az.compare({'sp_global': idata_sp_global, 'sp_individual': idata_sp_individual, 'sp_mix': idata_sp_mix})
    cmp_1
    return (cmp_1,)


@app.cell
def _(az, cmp_1, plt):
    az.plot_compare(cmp_1, insample_dev=False, figsize=(8, 2))
    plt.savefig('sp_compare.png')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
