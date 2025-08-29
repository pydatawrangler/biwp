import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code 4: Extending Linear Models
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
    import numpy as np
    from scipy import stats
    return az, np, pd, plt, pm, stats, xr


@app.cell
def _(az, plt):
    az.style.use("arviz-grayscale")
    plt.rcParams['figure.dpi'] = 300
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Transforming Covariates
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.1
        """
    )
    return


@app.cell
def _(pd):
    babies = pd.read_csv('../data/babies.csv')

    # Add a constant term so we can use a the dot product approach
    babies["Intercept"] = 1

    babies.head()
    return (babies,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.1
        """
    )
    return


@app.cell
def _(babies, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(babies['Month'], babies['Length'], 'C0.', alpha=0.1)
    _ax.set_ylabel('Length')
    _ax.set_xlabel('Month')
    plt.savefig('img/chp04/baby_length_scatter.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.2
        """
    )
    return


@app.cell
def _(babies, pm):
    with pm.Model() as model_baby_linear:
        _β = pm.Normal('β', sigma=10, shape=2)
        _μ = pm.Deterministic('μ', pm.math.dot(babies[['Intercept', 'Month']], _β))
        ε = pm.HalfNormal('ϵ', sigma=10)
        _length = pm.Normal('length', mu=_μ, sigma=ε, observed=babies['Length'])
        idata_linear = pm.sample(draws=2000, tune=4000, idata_kwargs={'log_likelihood': True})
        idata_linear.extend(pm.sample_posterior_predictive(idata_linear))
    return (idata_linear,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.2
        """
    )
    return


@app.cell
def _(az, babies, idata_linear, plt):
    _fig, _ax = plt.subplots()
    _ax.set_ylabel('Length')
    _ax.set_xlabel('Month')
    _μ_m = idata_linear.posterior['μ'].mean(('chain', 'draw'))
    _ax.plot(babies['Month'], _μ_m, c='C4')
    az.plot_hdi(babies['Month'], idata_linear.posterior_predictive['length'], hdi_prob=0.5, ax=_ax)
    az.plot_hdi(babies['Month'], idata_linear.posterior_predictive['length'], hdi_prob=0.94, ax=_ax)
    _ax.plot(babies['Month'], babies['Length'], 'C0.', alpha=0.1)
    plt.savefig('img/chp04/baby_length_linear_fit.png', dpi=300)
    return


@app.cell
def _(az, idata_linear):
    az.loo(idata_linear)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.3
        """
    )
    return


@app.cell
def _(babies, np, pm):
    with pm.Model() as model_baby_sqrt:
        _β = pm.Normal('β', sigma=10, shape=2)
        _μ = pm.Deterministic('μ', _β[0] + _β[1] * np.sqrt(babies['Month']))
        _σ = pm.HalfNormal('σ', sigma=10)
        _length = pm.Normal('length', mu=_μ, sigma=_σ, observed=babies['Length'])
        idata_sqrt = pm.sample(draws=2000, tune=4000, idata_kwargs={'log_likelihood': True})
        idata_sqrt.extend(pm.sample_posterior_predictive(idata_sqrt))
    return (idata_sqrt,)


@app.cell
def _(az, babies, idata_sqrt, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(babies['Month'], babies['Length'], 'C0.', alpha=0.1)
    _ax.set_ylabel('Length')
    _ax.set_xlabel('Month')
    _μ_m = idata_sqrt.posterior['μ'].mean(('chain', 'draw'))
    az.plot_hdi(babies['Month'], idata_sqrt.posterior_predictive['length'], hdi_prob=0.5, ax=_ax)
    az.plot_hdi(babies['Month'], idata_sqrt.posterior_predictive['length'], hdi_prob=0.94, ax=_ax)
    _ax.plot(babies['Month'], _μ_m, c='C4')
    plt.savefig('img/chp04/baby_length_sqrt_fit.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.3
        """
    )
    return


@app.cell
def _(az, babies, idata_sqrt, np, plt):
    _fig, _axes = plt.subplots(1, 2)
    _axes[0].plot(babies['Month'], babies['Length'], 'C0.', alpha=0.1)
    _μ_m = idata_sqrt.posterior['μ'].mean(('chain', 'draw'))
    _axes[0].plot(babies['Month'], _μ_m, c='C4')
    az.plot_hdi(babies['Month'], idata_sqrt.posterior_predictive['length'], hdi_prob=0.5, ax=_axes[0])
    az.plot_hdi(babies['Month'], idata_sqrt.posterior_predictive['length'], hdi_prob=0.94, ax=_axes[0])
    _axes[0].set_ylabel('Length')
    _axes[0].set_xlabel('Month')
    _axes[1].plot(np.sqrt(babies['Month']), babies['Length'], 'C0.', alpha=0.1)
    _axes[1].set_xlabel('Square Root of Month')
    az.plot_hdi(np.sqrt(babies['Month']), idata_sqrt.posterior_predictive['length'], hdi_prob=0.5, ax=_axes[1])
    az.plot_hdi(np.sqrt(babies['Month']), idata_sqrt.posterior_predictive['length'], hdi_prob=0.94, ax=_axes[1])
    _axes[1].plot(np.sqrt(babies['Month']), _μ_m, c='C4')
    _axes[1].set_yticks([])
    _axes[1]
    plt.savefig('img/chp04/baby_length_sqrt_fit.png', dpi=300)
    return


@app.cell
def _(az, idata_linear, idata_sqrt):
    az.compare({"Linear Model":idata_linear,
                "Non Linear Model":idata_sqrt})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Varying Uncertainty
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.4
        """
    )
    return


@app.cell
def _(babies, np, pm):
    with pm.Model() as model_baby_vv:
        _β = pm.Normal('β', sigma=10, shape=2)
        δ = pm.HalfNormal('δ', sigma=10, shape=2)
        _μ = pm.Deterministic('μ', _β[0] + _β[1] * np.sqrt(babies['Month']))
        _σ = pm.Deterministic('σ', δ[0] + δ[1] * babies['Month'])
        _length = pm.Normal('length', mu=_μ, sigma=_σ, observed=babies['Length'])
        idata_baby_vv = pm.sample(2000, target_accept=0.95)
        idata_baby_vv.extend(pm.sample_posterior_predictive(idata_baby_vv))
    return (idata_baby_vv,)


@app.cell
def _(az, idata_baby_vv):
    az.summary(idata_baby_vv, var_names=["δ"])
    return


@app.cell
def _(az, babies, idata_baby_vv, plt):
    _fig, _ax = plt.subplots()
    _ax.set_ylabel('Length')
    _ax.set_xlabel('Month')
    _ax.plot(babies['Month'], babies['Length'], 'C0.', alpha=0.1)
    _μ_m = idata_baby_vv.posterior['μ'].mean(('chain', 'draw'))
    _ax.plot(babies['Month'], _μ_m, c='C4')
    az.plot_hdi(babies['Month'], idata_baby_vv.posterior_predictive['length'], hdi_prob=0.5, ax=_ax)
    az.plot_hdi(babies['Month'], idata_baby_vv.posterior_predictive['length'], hdi_prob=0.94, ax=_ax)
    plt.savefig('img/chp04/baby_length_sqrt_vv_fit.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.4
        """
    )
    return


@app.cell
def _(az, babies, idata_baby_vv, plt):
    _fig, _axes = plt.subplots(2, 1)
    _axes[0].plot(babies['Month'], babies['Length'], 'C0.', alpha=0.1)
    _μ_m = idata_baby_vv.posterior['μ'].mean(('chain', 'draw'))
    _axes[0].plot(babies['Month'], _μ_m, c='C4')
    az.plot_hdi(babies['Month'], idata_baby_vv.posterior_predictive['length'], hdi_prob=0.5, ax=_axes[0])
    az.plot_hdi(babies['Month'], idata_baby_vv.posterior_predictive['length'], hdi_prob=0.94, ax=_axes[0])
    _axes[0].set_ylabel('Length')
    _σ_m = idata_baby_vv.posterior['σ'].mean(('chain', 'draw'))
    _axes[1].plot(babies['Month'], _σ_m, c='C1')
    _axes[1].set_ylabel('σ')
    _axes[1].set_xlabel('Month')
    _axes[0].set_xlim(0, 24)
    _axes[1].set_xlim(0, 24)
    plt.savefig('img/chp04/baby_length_sqrt_vv_fit_include_error.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interaction effects
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.5
        """
    )
    return


@app.cell
def _(pd):
    tips_df = pd.read_csv('../data/tips.csv')
    tips_df.head()
    return (tips_df,)


@app.cell
def _(pd, pm, tips_df):
    tips = tips_df['tip']
    total_bill_c = tips_df['total_bill'] - tips_df['total_bill'].mean()
    smoker = pd.Categorical(tips_df['smoker']).codes
    with pm.Model() as model_no_interaction:
        _β = pm.Normal('β', mu=0, sigma=1, shape=3)
        _σ = pm.HalfNormal('σ', 1)
        _μ = _β[0] + _β[1] * total_bill_c + _β[2] * smoker
        _obs = pm.Normal('obs', _μ, _σ, observed=tips)
        idata_no_interaction = pm.sample(1000, tune=1000)
    return idata_no_interaction, smoker, tips, total_bill_c


@app.cell
def _(idata_no_interaction):
    idata_no_interaction.posterior
    return


@app.cell
def _(az, idata_no_interaction, plt, smoker, tips, total_bill_c, xr):
    _, _ax = plt.subplots(figsize=(8, 4.5))
    total_bill_c_da = xr.DataArray(total_bill_c)
    _posterior_no_interaction = az.extract(idata_no_interaction, var_names=['β'])
    _β0_nonint = _posterior_no_interaction.sel(β_dim_0=0)
    _β1_nonint = _posterior_no_interaction.sel(β_dim_0=1)
    _β2_nonint = _posterior_no_interaction.sel(β_dim_0=2)
    _pred_y_non_smokers = _β0_nonint + _β1_nonint * total_bill_c_da
    _pred_y_smokers = _β0_nonint + _β1_nonint * total_bill_c_da + _β2_nonint
    _ax.scatter(total_bill_c[smoker == 0], tips[smoker == 0], label='non-smokers', marker='.')
    _ax.scatter(total_bill_c[smoker == 1], tips[smoker == 1], label='smokers', marker='.', c='C4')
    _ax.set_xlabel('Total Bill')
    _ax.set_ylabel('Tip')
    _ax.legend()
    _ax.plot(total_bill_c, _pred_y_non_smokers.mean('sample'), lw=2)
    _ax.plot(total_bill_c, _pred_y_smokers.mean('sample'), lw=2, c='C4')
    return (total_bill_c_da,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.6
        """
    )
    return


@app.cell
def _(pm, smoker, tips, total_bill_c):
    with pm.Model() as model_interaction:
        _β = pm.Normal('β', mu=0, sigma=1, shape=4)
        _σ = pm.HalfNormal('σ', 1)
        _μ = _β[0] + _β[1] * total_bill_c + _β[2] * smoker + _β[3] * smoker * total_bill_c
        _obs = pm.Normal('obs', _μ, _σ, observed=tips)
        idata_interaction = pm.sample(1000, tune=1000)
    return (idata_interaction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.5
        """
    )
    return


@app.cell
def _(
    az,
    idata_interaction,
    idata_no_interaction,
    plt,
    smoker,
    tips,
    total_bill_c,
    total_bill_c_da,
):
    _, _ax = plt.subplots(1, 2, figsize=(8, 4.5))
    _posterior_no_interaction = az.extract(idata_no_interaction, var_names=['β'])
    _β0_nonint = _posterior_no_interaction.sel(β_dim_0=0)
    _β1_nonint = _posterior_no_interaction.sel(β_dim_0=1)
    _β2_nonint = _posterior_no_interaction.sel(β_dim_0=2)
    _pred_y_non_smokers = _β0_nonint + _β1_nonint * total_bill_c_da
    _pred_y_smokers = _β0_nonint + _β1_nonint * total_bill_c_da + _β2_nonint
    _ax[0].scatter(total_bill_c[smoker == 0], tips[smoker == 0], label='non-smokers', marker='.')
    _ax[0].scatter(total_bill_c[smoker == 1], tips[smoker == 1], label='smokers', marker='.', c='C4')
    _ax[0].set_xlabel('Total Bill (Centered)')
    _ax[0].set_ylabel('Tip')
    _ax[0].legend(frameon=True)
    _ax[0].plot(total_bill_c, _pred_y_non_smokers.mean('sample'), lw=2)
    _ax[0].plot(total_bill_c, _pred_y_smokers.mean('sample'), lw=2, c='C4')
    _ax[0].set_title('No Interaction')
    az.plot_hdi(total_bill_c, _pred_y_non_smokers, color='C0', ax=_ax[0])
    az.plot_hdi(total_bill_c, _pred_y_smokers, ax=_ax[0], color='C4')
    posterior_interaction = az.extract(idata_interaction, var_names=['β'])
    β0_int = posterior_interaction.sel(β_dim_0=0)
    β1_int = posterior_interaction.sel(β_dim_0=1)
    β2_int = posterior_interaction.sel(β_dim_0=2)
    β3_int = posterior_interaction.sel(β_dim_0=3)
    _pred_y_non_smokers = β0_int + β1_int * total_bill_c_da
    _pred_y_smokers = β0_int + β1_int * total_bill_c_da + β2_int + β3_int * total_bill_c_da
    _ax[1].scatter(total_bill_c[smoker == 0], tips[smoker == 0], label='non-smokers', marker='.')
    _ax[1].scatter(total_bill_c[smoker == 1], tips[smoker == 1], label='smokers', marker='.', c='C4')
    _ax[1].set_xlabel('Total Bill (Centered)')
    _ax[1].set_yticks([])
    _ax[1].set_title('Interaction')
    _ax[1].plot(total_bill_c, _pred_y_non_smokers.mean('sample'), lw=2)
    _ax[1].plot(total_bill_c, _pred_y_smokers.mean('sample'), lw=2)
    az.plot_hdi(total_bill_c, _pred_y_non_smokers, color='C0', ax=_ax[1])
    az.plot_hdi(total_bill_c, _pred_y_smokers, ax=_ax[1], color='C4')
    plt.savefig('img/chp04/smoker_tip_interaction.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Robust Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.6
        """
    )
    return


@app.cell
def _(np, plt, stats):
    mean = 5
    _sigma = 2
    x = np.linspace(-5, 15, 1000)
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(x, stats.norm(5, 2).pdf(x), label=f'Normal μ={mean}, σ={_sigma}', color='C4')
    for _i, nu in enumerate([1, 2, 20], 1):
        _ax.plot(x, stats.t(loc=5, scale=2, df=nu).pdf(x), label=f'Student T μ={mean}, σ={_sigma}, ν={nu}', color=f'C{_i}')
    _ax.set_xlim(-5, 18)
    _ax.legend(loc='upper right', frameon=False)
    _ax.set_yticks([])
    plt.savefig('img/chp04/studentt_normal_comparison.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.7
        """
    )
    return


@app.cell
def _(np, pd, stats):
    def generate_sales(*, days, mean, std, label):
        np.random.seed(0)
        df = pd.DataFrame(index=range(1, days+1), columns=["customers", "sales"])
        for day in range(1, days+1):
            num_customers = stats.randint(30, 100).rvs()+1
        
            # This is correct as there is an independent draw for each customers orders
            dollar_sales = stats.norm(mean, std).rvs(num_customers).sum()
        
            df.loc[day, "customers"] = num_customers
            df.loc[day, "sales"] = dollar_sales
        
        # Fix the types as not to cause Theano errors
        df = df.astype({'customers': 'int32', 'sales': 'float32'})
    
        # Sorting will make plotting the posterior predictive easier later
        df["Food_Category"] = label
        df = df.sort_values("customers")
        return df
    return (generate_sales,)


@app.cell
def _(generate_sales, plt):
    _fig, _ax = plt.subplots()
    empanadas = generate_sales(days=200, mean=180, std=30, label='Empanada')
    empanadas.iloc[0] = [50, 92000, 'Empanada']
    empanadas.iloc[1] = [60, 90000, 'Empanada']
    empanadas.iloc[2] = [70, 96000, 'Empanada']
    empanadas.iloc[3] = [80, 91000, 'Empanada']
    empanadas.iloc[4] = [90, 99000, 'Empanada']
    empanadas = empanadas.sort_values('customers')
    empanadas.sort_values('sales')[:-5].plot(x='customers', y='sales', kind='scatter', ax=_ax)
    empanadas.sort_values('sales')[-5:].plot(x='customers', y='sales', kind='scatter', c='C4', ax=_ax)
    _ax.set_ylabel('Argentine Peso')
    _ax.set_xlabel('Customer Count')
    _ax.set_title('Empanada Sales')
    plt.savefig('img/chp04/empanada_scatter_plot.png', dpi=300)
    return (empanadas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.7
        """
    )
    return


@app.cell
def _(empanadas, pm):
    with pm.Model() as model_non_robust:
        _σ = pm.HalfNormal('σ', 50)
        _β = pm.Normal('β', mu=150, sigma=20)
        _μ = pm.Deterministic('μ', _β * empanadas['customers'])
        _sales = pm.Normal('sales', mu=_μ, sigma=_σ, observed=empanadas['sales'])
        idata_non_robust = pm.sample(random_seed=1, idata_kwargs={'log_likelihood': True})
        idata_non_robust.extend(pm.sample_posterior_predictive(idata_non_robust))
    return (idata_non_robust,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.8
        """
    )
    return


@app.cell
def _(az, empanadas, idata_non_robust, plt):
    _fig, _axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    _μ_m = idata_non_robust.posterior['μ'].mean(('chain', 'draw'))
    for _i in range(2):
        empanadas.sort_values('sales')[:-5].plot(x='customers', y='sales', kind='scatter', ax=_axes[_i])
        empanadas.sort_values('sales')[-5:].plot(x='customers', y='sales', kind='scatter', c='C4', ax=_axes[_i])
        _axes[_i].plot(empanadas.customers, _μ_m, c='C4')
        az.plot_hdi(empanadas.customers, idata_non_robust.posterior_predictive['sales'], hdi_prob=0.95, ax=_axes[_i])
        _axes[1].set_ylabel('Argentine Peso')
    _axes[0].set_ylabel('')
    _axes[1].set_xlabel('Customer Count')
    _axes[1].set_ylim(400, 25000)
    plt.savefig('img/chp04/empanada_scatter_non_robust.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 4.1
        """
    )
    return


@app.cell
def _(az, idata_non_robust):
    az.summary(idata_non_robust, kind="stats", var_names=["β", "σ"]).round(1)
    return


@app.cell
def _(empanadas, pm):
    with pm.Model() as model_robust:
        _σ = pm.HalfNormal('σ', 50)
        _β = pm.Normal('β', mu=150, sigma=20)
        ν = pm.HalfNormal('ν', 20)
        _μ = pm.Deterministic('μ', _β * empanadas['customers'])
        _sales = pm.StudentT('sales', mu=_μ, sigma=_σ, nu=ν, observed=empanadas['sales'])
        idata_robust = pm.sample(random_seed=0, idata_kwargs={'log_likelihood': True})
        idata_robust.extend(pm.sample_posterior_predictive(idata_robust))
    return (idata_robust,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 4.2
        """
    )
    return


@app.cell
def _(az, idata_robust):
    az.summary(idata_robust, var_names=["β", "σ", "ν"], kind="stats").round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.9
        """
    )
    return


@app.cell
def _(az, empanadas, idata_robust, plt):
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _μ_m = idata_robust.posterior['μ'].mean(('chain', 'draw'))
    _ax.plot(empanadas.customers, _μ_m, c='C4')
    az.plot_hdi(empanadas.customers, idata_robust.posterior_predictive['sales'], hdi_prob=0.95, ax=_ax)
    empanadas.plot(x='customers', y='sales', kind='scatter', ax=_ax)
    _ax.set_ylim(4000, 20000)
    _ax.set_ylabel('Argentine Peso')
    _ax.set_xlabel('Customer Count')
    _ax.set_title('Empanada Sales with Robust Regression Fit')
    plt.savefig('img/chp04/empanada_scatter_robust.png', dpi=300)
    return


@app.cell
def _(az, idata_non_robust, idata_robust):
    az.compare({"Non robust": idata_non_robust,
                "Robust":idata_robust})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Pooling, Multilevel Models, and Mixed Effects
        """
    )
    return


@app.cell
def _(np, pd, stats):
    def generate_sales_1(*, days, mean, std, label):
        np.random.seed(0)
        df = pd.DataFrame(index=range(1, days + 1), columns=['customers', 'sales'])
        for day in range(1, days + 1):
            num_customers = stats.randint(30, 100).rvs() + 1
            dollar_sales = stats.norm(mean, std).rvs(num_customers).sum()
            df.loc[day, 'customers'] = num_customers
            df.loc[day, 'sales'] = dollar_sales
        df = df.astype({'customers': 'int32', 'sales': 'float32'})
        df['Food_Category'] = label
        df = df.sort_values('customers')
        return df
    return (generate_sales_1,)


@app.cell
def _(generate_sales_1):
    pizza_df = generate_sales_1(days=365, mean=13, std=5, label='Pizza')
    sandwich_df = generate_sales_1(days=100, mean=6, std=5, label='Sandwich')
    salad_days = 3
    salad_df = generate_sales_1(days=salad_days, mean=8, std=3, label='Salad')
    salad_df.plot(x='customers', y='sales', kind='scatter')
    return pizza_df, salad_df, sandwich_df


@app.cell
def _(pd, pizza_df, salad_df, sandwich_df):
    sales_df = pd.concat([pizza_df, sandwich_df, salad_df]).reset_index(drop=True)
    sales_df["Food_Category"] = pd.Categorical(sales_df["Food_Category"])
    sales_df
    return (sales_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.10
        """
    )
    return


@app.cell
def _(pizza_df, plt, salad_df, sandwich_df):
    _fig, _ax = plt.subplots()
    pizza_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, c='C1', label='Pizza', marker='^', s=60)
    sandwich_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, label='Sandwich', marker='s')
    salad_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, label='Salad', c='C4')
    _ax.set_xlabel('Number of Customers')
    _ax.set_ylabel('Daily Sales Dollars')
    _ax.set_title('Aggregated Sales Dollars')
    _ax.legend()
    plt.savefig('img/chp04/restaurant_order_scatter.png', dpi=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Unpooled Model
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.9
        """
    )
    return


@app.cell
def _(pd, pm, sales_df):
    customers = sales_df.loc[:, 'customers'].values
    sales_observed = sales_df.loc[:, 'sales'].values
    food_category = pd.Categorical(sales_df['Food_Category'])
    coords = {'meals': food_category.categories}
    with pm.Model(coords=coords) as model_sales_unpooled:
        _σ = pm.HalfNormal('σ', 20, dims='meals')
        _β = pm.Normal('β', mu=10, sigma=10, dims='meals')
        _μ = pm.Deterministic('μ', _β[food_category.codes] * customers)
        _sales = pm.Normal('sales', mu=_μ, sigma=_σ[food_category.codes], observed=sales_observed)
        idata_sales_unpooled = pm.sample(target_accept=0.9)
    return (
        customers,
        food_category,
        idata_sales_unpooled,
        model_sales_unpooled,
        sales_observed,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.12
        """
    )
    return


@app.cell
def _(model_sales_unpooled, pm):
    sales_unpooled_diagram = pm.model_to_graphviz(model_sales_unpooled)
    sales_unpooled_diagram.render("img/chp04/salad_sales_basic_regression_model_unpooled", format="png", cleanup=True)
    sales_unpooled_diagram
    return


@app.cell
def _(idata_sales_unpooled):
    idata_salads_sales_unpooled = idata_sales_unpooled.posterior.sel(meals="Salad", μ_dim_0=slice(465, 467))
    return


@app.cell
def _(az, idata_sales_unpooled):
    az.summary(idata_sales_unpooled, var_names=["β", "σ"])
    return


@app.cell
def _(az, idata_sales_unpooled):
    az.plot_trace(idata_sales_unpooled, var_names=["β", "σ"], compact=False);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.13
        """
    )
    return


@app.cell
def _(az, idata_sales_unpooled, plt):
    _axes = az.plot_forest([idata_sales_unpooled], model_names=['Unpooled'], var_names=['β'], combined=True, figsize=(7, 1.8))
    _axes[0].set_title('β parameter estimates 94% HDI')
    plt.savefig('img/chp04/salad_sales_basic_regression_forestplot_beta.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.14
        """
    )
    return


@app.cell
def _(az, idata_sales_unpooled, plt):
    _axes = az.plot_forest([idata_sales_unpooled], model_names=['Unpooled'], var_names=['σ'], combined=True, figsize=(7, 1.8))
    _axes[0].set_title('σ parameter estimates 94% HDI')
    plt.savefig('img/chp04/salad_sales_basic_regression_forestplot_sigma.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Pooled Model
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.10
        """
    )
    return


@app.cell
def _(customers, pm, sales_observed):
    with pm.Model() as model_sales_pooled:
        _σ = pm.HalfNormal('σ', 20)
        _β = pm.Normal('β', mu=10, sigma=10)
        _μ = pm.Deterministic('μ', _β * customers)
        _sales = pm.Normal('sales', mu=_μ, sigma=_σ, observed=sales_observed)
        idata_sales_pooled = pm.sample()
    return idata_sales_pooled, model_sales_pooled


@app.cell
def _(idata_sales_pooled, model_sales_pooled, pm):
    with model_sales_pooled:
        idata_sales_pooled.extend(pm.sample_posterior_predictive(idata_sales_pooled))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.16
        """
    )
    return


@app.cell
def _(model_sales_pooled, pm):
    pooled_sales_diagram = pm.model_to_graphviz(model_sales_pooled)
    pooled_sales_diagram.render("img/chp04/salad_sales_basic_regression_model_pooled", format="png", cleanup=True)
    pooled_sales_diagram
    return


@app.cell
def _(az, idata_sales_pooled):
    az.plot_trace(idata_sales_pooled, var_names=["β", "σ"], compact=False);
    return


@app.cell
def _(az, idata_sales_pooled):
    az.summary(idata_sales_pooled, var_names=["β", "σ"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.17
        """
    )
    return


@app.cell
def _(az, idata_sales_pooled, idata_sales_unpooled, plt):
    _axes = az.plot_forest([idata_sales_pooled, idata_sales_unpooled], model_names=['Pooled', 'Unpooled'], var_names=['σ'], combined=True, figsize=(10, 3))
    _axes[0].set_title('Comparison of pooled and unpooled models \n 94% HDI')
    plt.savefig('img/chp04/salad_sales_basic_regression_forestplot_sigma_comparison.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.18
        """
    )
    return


@app.cell
def _(az, customers, idata_sales_pooled, pizza_df, plt, salad_df, sandwich_df):
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _μ_m = idata_sales_pooled.posterior['μ'].mean(('chain', 'draw'))
    _ax.plot(customers, _μ_m, c='C4')
    az.plot_hdi(customers, idata_sales_pooled.posterior_predictive['sales'], hdi_prob=0.5, ax=_ax)
    az.plot_hdi(customers, idata_sales_pooled.posterior_predictive['sales'], hdi_prob=0.94, ax=_ax)
    pizza_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, c='C1', label='Pizza', marker='^', s=60)
    sandwich_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, label='Sandwich', marker='s')
    salad_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, label='Salad', c='C4')
    _ax.set_xlabel('Number of Customers')
    _ax.set_ylabel('Daily Sales Dollars')
    _ax.set_title('Pooled Regression')
    plt.savefig('img/chp04/salad_sales_basic_regression_scatter_pooled.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.11
        """
    )
    return


@app.cell
def _(customers, food_category, pm, sales_observed):
    coords_1 = {'meals': food_category.categories, 'meals_idx': food_category}
    with pm.Model(coords=coords_1) as model_pooled_sigma_sales:
        _σ = pm.HalfNormal('σ', 20)
        _β = pm.Normal('β', mu=10, sigma=20, dims='meals')
        _μ = pm.Deterministic('μ', _β[food_category.codes] * customers, dims='meals_idx')
        _sales = pm.Normal('sales', mu=_μ, sigma=_σ, observed=sales_observed, dims='meals_idx')
        idata_pooled_sigma_sales = pm.sample()
        idata_pooled_sigma_sales.extend(pm.sample_posterior_predictive(idata_pooled_sigma_sales))
    return coords_1, idata_pooled_sigma_sales, model_pooled_sigma_sales


@app.cell
def _(model_pooled_sigma_sales, pm):
    multilevel_sales_diagram = pm.model_to_graphviz(model_pooled_sigma_sales)
    multilevel_sales_diagram.render("img/chp04/salad_sales_basic_regression_model_multilevel", format="png", cleanup=True)
    multilevel_sales_diagram
    return


@app.cell
def _(az, idata_pooled_sigma_sales):
    az.summary(idata_pooled_sigma_sales, var_names=["β", "σ"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.20
        """
    )
    return


@app.cell
def _(
    az,
    food_category,
    idata_pooled_sigma_sales,
    idata_sales_pooled,
    pizza_df,
    plt,
    salad_df,
    sales_df,
    sandwich_df,
):
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _σ_m = idata_sales_pooled.posterior['σ'].mean().values
    for meal in food_category.categories:
        category_mask = food_category == meal
        μ_m_meals = idata_pooled_sigma_sales.posterior['μ'].sel({'meals_idx': meal})
        _ax.plot(sales_df.customers[category_mask], μ_m_meals.mean(('chain', 'draw')), c='C4')
        az.plot_hdi(sales_df.customers[category_mask], idata_pooled_sigma_sales.posterior_predictive['sales'].sel({'meals_idx': meal}), hdi_prob=0.5, ax=_ax, fill_kwargs={'alpha': 0.5})
    pizza_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, c='C1', label='Pizza', marker='^', s=60)
    sandwich_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, label='Sandwich', marker='s')
    salad_df.plot(x='customers', y='sales', kind='scatter', ax=_ax, label='Salad', c='C4')
    _ax.set_xlabel('Number of Customers')
    _ax.set_ylabel('Daily Sales Dollars')
    _ax.set_title('Unpooled Slope Pooled Sigma Regression')
    plt.savefig('img/chp04/salad_sales_basic_regression_scatter_sigma_pooled_slope_unpooled.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.21
        """
    )
    return


@app.cell
def _(az, idata_pooled_sigma_sales, idata_sales_unpooled, plt):
    _axes = az.plot_forest([idata_sales_unpooled, idata_pooled_sigma_sales], model_names=['Unpooled', 'Multilevel '], var_names=['σ'], combined=True, figsize=(7, 1.8))
    _axes[0].set_title('Comparison of σ parameters 94% HDI')
    plt.savefig('img/chp04/salad_sales_forestplot_sigma_unpooled_multilevel_comparison.png')
    return


@app.cell
def _(az, idata_pooled_sigma_sales, idata_sales_unpooled):
    _axes = az.plot_forest([idata_sales_unpooled, idata_pooled_sigma_sales], model_names=['Unpooled', 'Multilevel'], var_names=['β'], combined=True, figsize=(7, 2.8))
    _axes[0].set_title('Comparison of β parameters 94% HDI')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Hierarchical
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.12
        """
    )
    return


@app.cell
def _(coords_1, customers, food_category, pm, sales_observed):
    with pm.Model(coords=coords_1) as model_hierarchical_sales:
        σ_hyperprior = pm.HalfNormal('σ_hyperprior', 20)
        _σ = pm.HalfNormal('σ', σ_hyperprior, dims='meals')
        _β = pm.Normal('β', mu=10, sigma=20, dims='meals')
        _μ = pm.Deterministic('μ', _β[food_category.codes] * customers)
        _sales = pm.Normal('sales', mu=_μ, sigma=_σ[food_category.codes], observed=sales_observed)
        idata_hierarchical_sales = pm.sample(target_accept=0.9)
    return idata_hierarchical_sales, model_hierarchical_sales


@app.cell
def _(az, idata_hierarchical_sales):
    az.plot_trace(idata_hierarchical_sales, compact=False, var_names=["β", "σ", "σ_hyperprior"]);
    return


@app.cell
def _(az, idata_hierarchical_sales):
    az.plot_parallel(idata_hierarchical_sales, var_names=["σ", "σ_hyperprior"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.23
        """
    )
    return


@app.cell
def _(model_hierarchical_sales, pm):
    hierarchial_sales_diagram = pm.model_to_graphviz(model_hierarchical_sales)
    hierarchial_sales_diagram.render("img/chp04/salad_sales_hierarchial_regression_model", format="png", cleanup=True)
    hierarchial_sales_diagram
    return


@app.cell
def _(az, idata_hierarchical_sales):
    az.summary(idata_hierarchical_sales, var_names=["β", "σ"])
    return


@app.cell
def _(az, idata_hierarchical_sales):
    _axes = az.plot_forest(idata_hierarchical_sales, var_names=['β'], combined=True, figsize=(7, 1.5))
    _axes[0].set_title('Hierarchical β estimates 94% HDI')
    return


@app.cell
def _(az, idata_hierarchical_sales, plt):
    _axes = az.plot_forest(idata_hierarchical_sales, var_names=['σ', 'σ_hyperprior'], combined=True, figsize=(7, 1.8))
    _axes[0].set_title('Hierarchical σ estimates 94% HDI')
    plt.savefig('img/chp04/salad_sales_forestplot_sigma_hierarchical.png')
    return


@app.cell
def _(food_category):
    print(food_category.categories)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 4.3
        """
    )
    return


@app.cell
def _(az, idata_sales_unpooled):
    az.summary(idata_sales_unpooled.posterior["σ"], kind="stats").round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Table 4.4
        """
    )
    return


@app.cell
def _(az, idata_hierarchical_sales):
    az.summary(idata_hierarchical_sales, var_names=["σ", "σ_hyperprior"], kind="stats").round(1)
    return


@app.cell
def _(az, idata_hierarchical_sales, idata_sales_unpooled, plt):
    _axes = az.plot_forest([idata_sales_unpooled.posterior['σ'].sel({'meals': 'Salad'}), idata_hierarchical_sales], model_names=['sales_unpooled', 'sales_hierarchical'], combined=True, figsize=(10, 4), var_names=['σ', 'σ_hyperprior'])
    _axes[0].set_title('Comparison of σ parameters from unpooled \n and hierarchical models \n 94% HDI')
    plt.savefig('img/chp04/salad_sales_forestolot_sigma_unpooled_multilevel_comparison.png')
    return


@app.cell
def _(az, idata_hierarchical_sales, idata_sales_unpooled, plt):
    _fig, _ax = plt.subplots()
    az.plot_kde(idata_sales_unpooled.posterior['σ'].sel({'meals': 'Salad'}).values, label='Unpooled Salad Sigma', ax=_ax)
    az.plot_kde(idata_hierarchical_sales.posterior['σ'].sel({'meals': 'Salad'}).values, label='Hierarchical Salad Sigma', plot_kwargs={'color': 'C4'}, ax=_ax)
    _ax.set_title('Comparison of Hierarchical versus Unpooled Variance')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.25
        """
    )
    return


@app.cell
def _(np, plt, stats):
    nsample = 10000
    nd = 1
    yr = stats.norm.rvs(loc=2.0, scale=3.0, size=nsample)
    xnr = stats.norm.rvs(loc=0.0, scale=np.exp(yr / 4), size=(nd, nsample))
    _fig, _ax = plt.subplots()
    _ax.scatter(xnr[0], yr, marker='.', alpha=0.05, color='C4')
    _ax.set_xlim(-20, 20)
    _ax.set_ylim(-9, 9)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    return


@app.cell
def _(np, pd, stats):
    def salad_generator(hyperprior_beta_mean=5, hyperprior_beta_sigma=0.2, sigma=50, days_per_location=[6, 4, 15, 10, 3, 5], sigma_per_location=[50, 10, 20, 80, 30, 20]):
        """Generate noisy salad data"""
        beta_hyperprior = stats.norm(hyperprior_beta_mean, hyperprior_beta_sigma)
        df = pd.DataFrame()
        for _i, days in enumerate(days_per_location):
            np.random.seed(0)
            num_customers = stats.randint(30, 100).rvs(days)
            sales_location = beta_hyperprior.rvs() * num_customers + stats.norm(0, sigma_per_location[_i]).rvs(num_customers.shape)
            location_df = pd.DataFrame({'customers': num_customers, 'sales': sales_location})
            location_df['location'] = _i
            location_df.sort_values(by='customers', ascending=True)
            df = pd.concat([df, location_df])
        df.reset_index(inplace=True, drop=True)
        return df
    hierarchical_salad_df = salad_generator()
    return (hierarchical_salad_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.26
        """
    )
    return


@app.cell
def _(hierarchical_salad_df, plt):
    _fig, _axes = plt.subplots(2, 3, sharex=True, sharey=True)
    for _i, _ax in enumerate(_axes.ravel()):
        location_filter = hierarchical_salad_df['location'] == _i
        hierarchical_salad_df[location_filter].plot(kind='scatter', x='customers', y='sales', ax=_ax)
        _ax.set_xlabel('')
        _ax.set_ylabel('')
    _axes[1, 0].set_xlabel('Number of Customers')
    _axes[1, 0].set_ylabel('Sales')
    plt.savefig('img/chp04/multiple_salad_sales_scatter.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.13
        """
    )
    return


@app.cell
def _():
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    root = tfd.JointDistributionCoroutine.Root
    return root, tf, tfd, tfp


@app.cell
def _(tf, tfp):
    run_mcmc = tf.function(
        tfp.experimental.mcmc.windowed_adaptive_nuts,
        autograph=False, jit_compile=True)
    return (run_mcmc,)


@app.cell
def _(hierarchical_salad_df, root, tf, tfd):
    def gen_hierarchical_salad_sales(input_df, beta_prior_fn, dtype=tf.float32):
        customers = tf.constant(hierarchical_salad_df['customers'].values, dtype=dtype)
        location_category = hierarchical_salad_df['location'].values
        _sales = tf.constant(hierarchical_salad_df['sales'].values, dtype=dtype)

        @tfd.JointDistributionCoroutine
        def model_hierarchical_salad_sales():
            β_μ_hyperprior = (yield root(tfd.Normal(0, 10, name='beta_mu')))
            β_σ_hyperprior = (yield root(tfd.HalfNormal(0.1, name='beta_sigma')))
            _β = (yield from beta_prior_fn(β_μ_hyperprior, β_σ_hyperprior))
            σ_hyperprior = (yield root(tfd.HalfNormal(30, name='sigma_prior')))
            _σ = (yield tfd.Sample(tfd.HalfNormal(σ_hyperprior), 6, name='sigma'))
            loc = tf.gather(_β, location_category, axis=-1) * customers
            scale = tf.gather(_σ, location_category, axis=-1)
            _sales = (yield tfd.Independent(tfd.Normal(loc, scale), reinterpreted_batch_ndims=1, name='sales'))
        return (model_hierarchical_salad_sales, _sales)
    return (gen_hierarchical_salad_sales,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.14 and 4.15
        """
    )
    return


@app.cell
def _(gen_hierarchical_salad_sales, hierarchical_salad_df, np, run_mcmc, tfd):
    def centered_beta_prior_fn(hyper_mu, hyper_sigma):
        _β = (yield tfd.Sample(tfd.Normal(hyper_mu, hyper_sigma), 6, name='beta'))
        return _β
    centered_model, observed = gen_hierarchical_salad_sales(hierarchical_salad_df, centered_beta_prior_fn)
    mcmc_samples_centered, sampler_stats_centered = run_mcmc(1000, centered_model, n_chains=4, num_adaptation_steps=1000, sales=observed)
    _divergent_per_chain = np.sum(sampler_stats_centered['diverging'], axis=0)
    print(f'There were {_divergent_per_chain} divergences after tuning per chain.')
    return mcmc_samples_centered, sampler_stats_centered


@app.cell
def _(az, mcmc_samples_centered, np, sampler_stats_centered):
    idata_centered_model = az.from_dict(
        posterior={
            k:np.swapaxes(v, 1, 0)
            for k, v in mcmc_samples_centered._asdict().items()},
        sample_stats={
            k:np.swapaxes(sampler_stats_centered[k], 1, 0)
            for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]}
    )

    az.plot_trace(idata_centered_model, compact=True);
    return (idata_centered_model,)


@app.cell
def _(az, idata_centered_model):
    az.summary(idata_centered_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.27
        """
    )
    return


@app.cell
def _(az, mcmc_samples_centered, plt, sampler_stats_centered):
    _slope = mcmc_samples_centered.beta[..., 4].numpy().flatten()
    _sigma = mcmc_samples_centered.beta_sigma.numpy().flatten()
    _divergences = sampler_stats_centered['diverging'].numpy().flatten()
    _axes = az.plot_pair({'β[4]': _slope, 'β_σ_hyperprior': _sigma}, figsize=(10, 4))
    _axes.scatter(_slope[_divergences], _sigma[_divergences], c='C4', alpha=0.3, label='divergent sample')
    _axes.legend(frameon=True)
    _axes.set_ylim(0, 0.3)
    _axes.set_xlim(4.5, 5.5)
    plt.savefig('img/chp04/Neals_Funnel_Salad_Centered.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.16
        """
    )
    return


@app.cell
def _(
    gen_hierarchical_salad_sales,
    hierarchical_salad_df,
    np,
    root,
    run_mcmc,
    tfd,
):
    def non_centered_beta_prior_fn(hyper_mu, hyper_sigma):
        β_offset = (yield root(tfd.Sample(tfd.Normal(0, 1), 6, name='beta_offset')))
        return β_offset * hyper_sigma[..., None] + hyper_mu[..., None]
    non_centered_model, observed_1 = gen_hierarchical_salad_sales(hierarchical_salad_df, non_centered_beta_prior_fn)
    mcmc_samples_noncentered, sampler_stats_noncentered = run_mcmc(1000, non_centered_model, n_chains=4, num_adaptation_steps=1000, sales=observed_1)
    _divergent_per_chain = np.sum(sampler_stats_noncentered['diverging'], axis=0)
    print(f'There were {_divergent_per_chain} divergences after tuning per chain.')
    return (
        mcmc_samples_noncentered,
        non_centered_model,
        observed_1,
        sampler_stats_noncentered,
    )


@app.cell
def _(az, mcmc_samples_noncentered, np, sampler_stats_noncentered):
    idata_non_centered_model = az.from_dict(
        posterior={
            k:np.swapaxes(v, 1, 0)
            for k, v in mcmc_samples_noncentered._asdict().items()},
        sample_stats={
            k:np.swapaxes(sampler_stats_noncentered[k], 1, 0)
            for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]}
    )

    az.plot_trace(idata_non_centered_model, compact=True);
    return (idata_non_centered_model,)


@app.cell
def _(az, idata_non_centered_model):
    az.summary(idata_non_centered_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure  4.28
        """
    )
    return


@app.cell
def _(az, mcmc_samples_noncentered, plt, sampler_stats_noncentered):
    noncentered_beta = mcmc_samples_noncentered.beta_mu[..., None] + mcmc_samples_noncentered.beta_offset * mcmc_samples_noncentered.beta_sigma[..., None]
    _slope = noncentered_beta[..., 4].numpy().flatten()
    _sigma = mcmc_samples_noncentered.beta_sigma.numpy().flatten()
    _divergences = sampler_stats_noncentered['diverging'].numpy().flatten()
    _axes = az.plot_pair({'β[4]': _slope, 'β_σ_hyperprior': _sigma}, figsize=(10, 4))
    _axes.scatter(_slope[_divergences], _sigma[_divergences], c='C4', alpha=0.3, label='divergent sample')
    _axes.legend(frameon=True)
    _axes.set_ylim(0, 0.3)
    _axes.set_xlim(4.5, 5.5)
    plt.savefig('img/chp04/Neals_Funnel_Salad_NonCentered.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Figure 4.29
        """
    )
    return


@app.cell
def _(mcmc_samples_centered, mcmc_samples_noncentered):
    centered_β_sigma = mcmc_samples_centered.beta_sigma.numpy()
    noncentered_β_sigma = mcmc_samples_noncentered.beta_sigma.numpy()
    return centered_β_sigma, noncentered_β_sigma


@app.cell
def _(az, centered_β_sigma, noncentered_β_sigma, plt):
    _fig, _ax = plt.subplots()
    az.plot_kde(centered_β_sigma, label='Centered β_σ_hyperprior', ax=_ax)
    az.plot_kde(noncentered_β_sigma, label='Noncentered β_σ_hyperprior', plot_kwargs={'color': 'C4'}, ax=_ax)
    _ax.set_title('Comparison of Centered vs Non Centered Estimates')
    plt.savefig('img/chp04/Salad_Sales_Hierarchical_Comparison.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.17
        """
    )
    return


@app.cell
def _(mcmc_samples_noncentered, non_centered_model, observed_1, root, tf, tfd):
    out_of_sample_customers = 50.0

    @tfd.JointDistributionCoroutine
    def out_of_sample_prediction_model():
        model = (yield root(non_centered_model))
        _β = model.beta_offset * model.beta_sigma[..., None] + model.beta_mu[..., None]
        β_group = (yield tfd.Normal(model.beta_mu, model.beta_sigma, name='group_beta_prediction'))
        group_level_prediction = (yield tfd.Normal(β_group * out_of_sample_customers, model.sigma_prior, name='group_level_prediction'))
        for l in [2, 4]:
            yield tfd.Normal(tf.gather(_β, l, axis=-1) * out_of_sample_customers, tf.gather(model.sigma, l, axis=-1), name=f'location_{l}_prediction')
    amended_posterior = tf.nest.pack_sequence_as(non_centered_model.sample(), list(mcmc_samples_noncentered) + [observed_1])
    ppc = out_of_sample_prediction_model.sample(var0=amended_posterior)
    return amended_posterior, ppc


@app.cell
def _(az, plt, ppc):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    az.plot_kde(ppc.group_level_prediction.numpy(), plot_kwargs={'color': 'C0'}, ax=_ax, label='All locations')
    az.plot_kde(ppc.location_2_prediction.numpy(), plot_kwargs={'color': 'C2'}, ax=_ax, label='Location 2')
    az.plot_kde(ppc.location_4_prediction.numpy(), plot_kwargs={'color': 'C4'}, ax=_ax, label='Location 4')
    _ax.set_xlabel('Predicted revenue with 50 customers')
    _ax.set_xlim([0, 600])
    _ax.set_yticks([])
    plt.savefig('img/chp04/Salad_Sales_Hierarchical_Predictions.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Code 4.18
        """
    )
    return


@app.cell
def _(amended_posterior, non_centered_model, np, root, tfd):
    out_of_sample_customers2 = np.arange(50, 90)

    @tfd.JointDistributionCoroutine
    def out_of_sample_prediction_model2():
        model = (yield root(non_centered_model))
        β_new_loc = (yield tfd.Normal(model.beta_mu, model.beta_sigma, name='beta_new_loc'))
        σ_new_loc = (yield tfd.HalfNormal(model.sigma_prior, name='sigma_new_loc'))
        group_level_prediction = (yield tfd.Normal(β_new_loc[..., None] * out_of_sample_customers2, σ_new_loc[..., None], name='new_location_prediction'))
    ppc_1 = out_of_sample_prediction_model2.sample(var0=amended_posterior)
    return out_of_sample_customers2, ppc_1


@app.cell
def _(az, out_of_sample_customers2, ppc_1):
    az.plot_hdi(out_of_sample_customers2, ppc_1.new_location_prediction, hdi_prob=0.95, figsize=(10, 2))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
