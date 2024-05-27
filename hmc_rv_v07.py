# Updated RV fitting model using exoplanet, pymc3, and theano/aesara
# Following examples by Dan Foreman-Mackey: https://docs.exoplanet.codes/en/stable/tutorials/intro-to-pymc3/
# Author: Cole Johnston

import yaml
import corner
import arviz as az
import numpy as np
import pymc3 as pm
import pandas as pd
import exoplanet as xo
import pymc3_ext as pmx
import matplotlib.pyplot as plt
import aesara_theano_fallback.tensor as tt

from sys import argv,exit
from matplotlib import rc

plt.rcParams.update({
    "text.usetex": True,
        "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage{amsmath}",
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
    ]),
})

plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18

def get_dist(variable, module, dist, func_name, **kwargs):
    dist_ = getattr(module,dist)
    return dist_(func_name,**kwargs)


def get_prior(variable, module, dist, func_name, **kwargs):
    dist_ = getattr(module,'Normal')
    return dist_(func_name,**kwargs)


def wrap_attr(variable, argument, module, cfg, func_name='None'):
    if func_name == 'None':
        return cfg['varpars'][variable]['kwargs'][argument]
    else:
        func = getattr(module, func_name)
        return func(cfg['varpars'][variable]['kwargs'][argument])


def set_variables(variable, cfg):

    if cfg['varpars'][variable]['transform'] == 'None':
        print('{} is sampled linearly'.format(variable))
        linear_var = get_dist(variable, pm, cfg['varpars'][variable]['prior'],
                              variable+'_',
                              **{argument: wrap_attr(variable, argument, np, cfg,
                                           cfg['varpars'][variable]['transform'])
                              for argument in cfg['varpars'][variable]['kwargs']}
                              )
        log_var = pm.Deterministic('log'+variable, tt.log(linear_var))

    else:
        log_var = get_dist( variable, pm, cfg['varpars'][variable]['prior'],
                            'log'+variable,
                            **{argument: wrap_attr(variable, argument, np, cfg,
                               cfg['varpars'][variable]['transform'])
                               for argument in cfg['varpars'][variable]['kwargs']}
                            )
        linear_var = pm.Deterministic(variable+'_', tt.exp(log_var))

    return log_var, linear_var


def get_obs(cfg, component='primary'):

    if cfg['general']['type'] == 'rv':
        loc = cfg['general']['path'] + cfg['general']['{}_obs'.format(component)]
    elif cfg['general']['type'] == 'td':
        loc = cfg['general']['path'] + cfg['general']['{}_obs'.format(component)]
    elif cfg['general']['type'] == 'lc':
        loc = cfg['general']['path'] + cfg['general']['lc_obs']
    else:
        raise TypeError

    try:
        x,y,z = np.loadtxt(loc).T
    except:
        x,y = np.loadtxt(loc).T
        z = None

    return x,y,z



def get_RV_model(cfg):

    # Get data
    t_primary, obs_primary, obs_err_primary = get_obs(cfg,'primary')
    t_mean = np.mean(t_primary)
    t_primary -= t_mean

    # Make model
    with pm.Model() as model:

        # Setup sampled and deterministic relations for parameters

        # Orbital semi-amplitude of the primary
        logK1, K1 = set_variables('K1', cfg)

        # Orbital period
        logP, P = set_variables('Period', cfg)


        # Velocity offset
        gamma = get_dist( 'Gamma', pm, cfg['varpars']['Gamma']['prior'], 'Gamma_',
                           **{argument: wrap_attr('Gamma', argument, np, cfg,
                                        cfg['varpars']['Gamma']['transform'])
                           for argument in cfg['varpars']['Gamma']['kwargs']}
                          )


        # Set deterministic relationship for background
        rvtrend = pm.Normal('rvtrend', mu=0.0, sd=1.0, testval=0.0)
        bkg = pm.Deterministic('bkg', gamma+rvtrend*t_primary/365.25)


        # In order to make our sampling more efficient, and to sample
        # more physically meaningful parameters, we will reparameterize
        # how we handle the eccentricity, argument of periastron, and
        # time of periastron passage

        # If we are dealing with a low eccentricity, we will sample
        # (omega +/- phi) / 2
        if cfg['general']['ecc_type'] == 'small':
            print('Using small eccentricity formulation')
            print('Drawing eccentricity from a bounded Half-Caucht distribution')
            # ecc = pm.Uniform("ecc", lower=0, upper=1, testval=0.001)
            ecc_dist = pm.Bound(pm.HalfCauchy, lower=0, upper=1.)
            ecc = ecc_dist('ecc', beta=0.3, testval=0.005)
            plus = pmx.Angle("plus")
            minus = pmx.Angle("minus")
            phi = pm.Deterministic("phi", plus + minus)
            omega = pm.Deterministic("omega", plus - minus)

        # If we are dealing with a larger eccentricity, we want to
        # sample on the unit disk
        elif cfg['general']['ecc_type'] == 'disc':
            print('Sampling h & k from a unit disc')
            print('and transforming to ecc & omega')
            # Sample hk
            #  h = sqrt(e) * sin(w)
            #  k = sqrt(e) * cos(w)
            hk = pmx.UnitDisk("hk", testval=np.array([0.01, 0.01]))
            # Set deterministic relationship between sampled hk and ecc, omega
            ecc = pm.Deterministic("ecc", hk[0] ** 2 + hk[1] ** 2)
            omega = pm.Deterministic("omega", tt.arctan2(hk[1], hk[0]))
            phi = pmx.Angle('phi')

        # Otherwise, we can just use the good ol' sample both independently,
        # even though this is sometimes a pain for the sampler.
        elif cfg['general']['ecc_type'] == 'normal':
            print('Sampling ecc & omega from uniform distributions')
            ecc = pm.Uniform("ecc", lower=0, upper=1)
            omega = pmx.Angle("omega")
            phi = pmx.Angle("phi")

        # This setup fixes the orbit to circular
        # and only sample phi
        elif cfg['general']['ecc_type'] == 'circular':
            print('Assuming a circular orbit')
            ecc = 0.
            omega = -0.5 * np.pi
            phi = pmx.Angle("phi")

        else:
            print('Current eccentricity setup not supported')
            print('Please select from: "small", "disc", "normal", or "circular"')
            exit()


        # Calculate the mean angular motion, i.e. the average rate of
        # completion of 2pi radians over one orbital period P
        #
        # --> n = 2 pi / P = 2 pi * exp(-logP)
        n = 2 * np.pi * tt.exp(-logP)

        # Here, we want to calculate what the actual time of periastron
        # passage is
        # phi = pm.Deterministic('Phi', n*t0 - omega)
        # t0 = pm.Deterministic("t0", (phi + omega)/n + t_mean)
        t0 = pm.Deterministic("t0", tt.exp(logP) * phi / (2.*np.pi) + t_mean)

        # Calculate the mean anomaly
        M = n * t_primary - (phi + omega)
         # t0 = (phi + w) / n
        cosw = tt.cos(omega)
        sinw = tt.sin(omega)


        f = xo.orbits.get_true_anomaly(M, ecc + tt.zeros_like(M))
        rvmodel = pm.Deterministic("rvmodel", bkg + K1 * (cosw * (tt.cos(f) + ecc)
                                    - sinw * tt.sin(f))
                                  )


        # Now we add a jitter term for the uncertainties
        log_jitter = pm.Normal("log_jitter_1", mu=np.log(0.05), sd=5.0)
        err_primary = tt.sqrt(obs_err_primary**2 + tt.exp( 2. * log_jitter) )
        # Condition on the observations
        pm.Normal("obs", mu=rvmodel, sd=err_primary, observed=obs_primary)


        # Compute the phased RV signal
        phase = np.linspace(0, 1, 500)
        M_pred = 2 * np.pi * phase - (phi + omega)
        f_pred = xo.orbits.get_true_anomaly(M_pred, ecc + tt.zeros_like(M_pred))
        rvphase = pm.Deterministic(
            "rvphase", K1 * (cosw * (tt.cos(f_pred) + ecc) - sinw * tt.sin(f_pred))
        )


        # Keep track of some derived parameters
        fm = pm.Deterministic('fm', 1.0361e-7 * (1-ecc**2)**(1.5) * K1**3 * P)
        a1sini = pm.Deterministic('a1sini', 0.019758*tt.sqrt(1-ecc**2) * K1 * P )

        # Check if this is an SB1 or SB2
        if cfg['general']['secondary_obs'] != 'None':

            t_secondary, obs_secondary, obs_err_secondary = get_obs(cfg,'secondary')
            t_secondary -= t_mean

            logK2 = get_dist( 'K2', pm, cfg['varpars']['K2']['prior'], 'logK2',
                               **{argument: wrap_attr('K2', argument, np, cfg,
                                            cfg['varpars']['K2']['transform'])
                               for argument in cfg['varpars']['K2']['kwargs']}
                              )

            K2 = pm.Deterministic("K2_", tt.exp(logK2))
            q = pm.Deterministic("q", K1/K2)


            rvmodel_secondary = pm.Deterministic("rvmodel_secondary", bkg - K2 * (cosw * (tt.cos(f) + ecc)
                                        - sinw * tt.sin(f))
                                      )
            log_jitter_2 = pm.Normal("log_jitter_2", mu=np.log(0.05), sd=5.0)
            err_secondary = tt.sqrt(obs_err_secondary**2 + tt.exp( 2. * log_jitter_2) )
            pm.Normal("obs_secondary", mu=rvmodel_secondary, sd=err_secondary, observed=obs_secondary)

            rvphase_secondary = pm.Deterministic(
                "rvphase_secondary", -K2 * (cosw * (tt.cos(f_pred) + ecc) - sinw * tt.sin(f_pred))
            )

            a2sini = pm.Deterministic('a2sini', 0.019758*tt.sqrt(1-ecc**2) * K2 * P )


        # TODO: Implement additional priors
        # if cfg['priors']['set'] == 'True':
        #     for var in cfg['priors']['pars']:
        #         exec('{}_prior={:f}')
        #         pm.Normal('{}_prior'.format(var), mu=cfg['priors'][var]['mean'],
        #                   sd=cfg['priors'][var]['std'], observed=val)


    return model


def plot_map_model(ax,cfg, soln):


    x_primary, obs_primary, err_primary = get_obs(cfg,'primary')
    x_mean = np.mean(x_primary)
    x_primary -= x_mean

    phase = np.linspace(0, 1, 500)
    period = soln["Period_"]

    ax.errorbar(x_primary % period, obs_primary - soln["bkg"], yerr=err_primary, fmt=".k")
    ax.plot(phase * period, soln["rvphase"], color="C0", lw=1)
    # ax.set_ylim(-110, 110)
    ax.set_ylabel(r"${\rm Radial~Velocity~[km\,s^{-1}]}$", fontsize=14)
    ax.set_xlabel(r"${\rm Time~[days]}$", fontsize=14)

    if cfg["general"]["secondary_obs"] != 'None':
        x_secondary, obs_secondary, err_secondary = get_obs(cfg,'secondary')
        x_secondary -= x_mean
        ax.errorbar(x_secondary % period, obs_secondary - soln["bkg"], yerr=err_secondary, fmt="xr")
        ax.plot(phase * period, soln["rvphase_secondary"], color="C1", lw=1)

    plt.tight_layout()

    return ax


def run_setup(cfg):

    # Instantiate the model first
    model = get_RV_model(cfg)

    # Decide if running sb1 or sb2 model

    # SB2
    if cfg['general']['secondary_obs'] != 'None':
        sampled_pars = [ "logK1", "logK2", "logPeriod", "phi", "Gamma_",
                         "log_jitter_1", "log_jitter_2", "rvtrend"]

        derived_pars = [ "K1_", "K2_", "Period_", "t0", "q", "a1sini",
                         "a2sini", "fm" ]

        summary_pars=[ "logK1", "logK2", "logPeriod", "phi", "Gamma_",
                       "log_jitter_1", "log_jitter_2", "rvtrend", "K1_", "K2_",
                       "Period_", "t0", "q", "a1sini", "a2sini", "fm"]
    # SB1
    else:
        sampled_pars = [ "logK1", "logPeriod", "phi", "Gamma_",
                         "log_jitter_1", "rvtrend"]
        derived_pars = [ "K1_", "Period_", "t0", "a1sini", "fm" ]
        summary_pars = [ "logK1", "logPeriod", "phi", "Gamma_",
                         "log_jitter_1", "rvtrend", "a1sini",
                         "K1_", "Period_", "t0", "a1sini", "fm" ]

    if cfg['general']['ecc_type'] != 'circular':

        sampled_pars.append('ecc')
        sampled_pars.append('omega')
        summary_pars.append('ecc')
        summary_pars.append('omega')


    return model, summary_pars, sampled_pars, derived_pars


def run_map(cfg, model):

    # MAP optimise
    with model:

        map_soln = pmx.optimize()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        axes = plot_map_model(ax, cfg, map_soln)
        plt.show()

    return model, map_soln


def run_NUTS( cfg, model, map_soln, summary_pars=["logK1", "logPeriod", "phi", "ecc",
              "omega", "Gamma_", "log_jitter", "a1sini", "fm", "Period_", "K1_", "t0",]):

    with model:

        trace = pmx.sample( draws=cfg['general']['draws'],
                            tune=cfg['general']['tune'],
                            start=map_soln,
                            chains=cfg['general']['chains'],
                            cores=cfg['general']['cores'],
                            target_accept=cfg['general']['target'],
                            return_inferencedata = True
                          )

        circ_var_names = ['phi']
        if cfg['general']['ecc_type'] != 'circular':
            circ_var_names.append('omega')

        summary = az.summary( trace, var_names=summary_pars,
                              circ_var_names=circ_var_names)

        print(summary)
        summary_save_path = '{}solns/{}.{}-{}.NUTS.soln'.format(cfg['general']['path'],
                                                        cfg['general']['prefix'],
                                                        cfg['general']['type'],
                                                        cfg['general']['iter'])
        summary.to_csv(summary_save_path, index=False)

    return trace


def plot_posteriors( trace, sampled_pars = [ "logK1", "logPeriod", "phi",
                    "Gamma_", "ecc", "omega", "log_jitter", "rvtrend" ],
                     derived_pars=["K1_", "Period_", "a1sini", "fm"]):


    sampled_save_path = '{}solns/{}.{}-{}.sampled_posteriors.png'.format(
                                                    cfg['general']['path'],
                                                    cfg['general']['prefix'],
                                                    cfg['general']['type'],
                                                    cfg['general']['iter'])
    a_ = corner.corner(trace, var_names=sampled_pars)
    plt.savefig(sampled_save_path)
    plt.show()

    derived_save_path = '{}solns/{}.{}-{}.derived_posteriors.png'.format(
                                                    cfg['general']['path'],
                                                    cfg['general']['prefix'],
                                                    cfg['general']['type'],
                                                    cfg['general']['iter'])
    b_ = corner.corner(trace, var_names=derived_pars)
    plt.savefig(derived_save_path)
    plt.show()


def get_stats(ary, var, prob=0.68, circular=False):
    median = ary.median()
    out = az.hdi(ary, prob, circular=circular)[var].values
    return out[0], median, out[1]


def plot_sampled_model(ax, cfg, trace):

    x_primary, obs_primary, err_primary = get_obs(cfg,'primary')
    x_mean = np.mean(x_primary)
    x_primary -= x_mean

    phase = np.linspace(0, 1, 500)

    bkg_mean = trace.posterior.bkg.median().values
    period_mean = trace.posterior.Period_.median().values
    t0_mean = trace.posterior.t0.median().values

    # print(period_mean, t0_mean)

    # _, bkg_mean, _ = np.percentile( trace.posterior.bkg, [16, 50, 84],
    #                                 axis=(0,1) )
    # _, period_mean, _ = np.percentile( trace.posterior.Period_, [16, 50, 84],
    #                                    axis=(0,1) )
    #
    # _, t0_mean, _ = np.percentile( trace.posterior.t0, [16, 50, 84],
    #                                axis=(0,1) )

    rvphase_low, rvphase_mean, rvphase_hi = np.percentile( trace.posterior.rvphase,
                                    [16, 50, 84], axis=(0,1))


    ph_primary = (x_primary - t0_mean + 0.5 * period_mean) % period_mean - 0.5 * period_mean



    ax.errorbar(x_primary % period_mean, obs_primary - bkg_mean, yerr=err_primary, fmt=".k")
    # ax.errorbar(ph_primary, obs_primary - bkg_mean, yerr=err_primary, fmt=".k")
    ax.fill_between(phase * period_mean, rvphase_low, rvphase_hi, color='C0', alpha=0.3)
    ax.plot(phase * period_mean, rvphase_mean, color="C0", lw=1)
    # ax.set_ylim(-110, 110)
    ax.set_ylabel(r"${\rm Radial~Velocity~[km\,s^{-1}]}$", fontsize=14)
    ax.set_xlabel(r"${\rm Time~[days]}$", fontsize=14)

    if cfg["general"]["secondary_obs"] != 'None':
        x_secondary, obs_secondary, err_secondary = get_obs(cfg,'secondary')
        x_secondary -= x_mean
        ph_primary = (x_primary - t0_mean + 0.5 * period_mean) % period_mean - 0.5 * period_mean


        rvph_secondary_low, rvph_secondary_mean, \
        rvph_secondary_hi = np.percentile( trace.posterior.rvphase_secondary,
                                           [16, 50, 84], axis=(0,1))

        ax.errorbar(x_secondary % period_mean, obs_secondary - bkg_mean, yerr=err_secondary, fmt="xr")
        ax.fill_between(phase * period_mean, rvph_secondary_low, rvph_secondary_hi, color='C1', alpha=0.3)
        ax.plot(phase * period_mean, rvph_secondary_mean, color="C1", lw=1)


    plt.tight_layout()
    plt.savefig('{}solns/{}.{}-{}.NUTS_model.png'.format(cfg['general']['path'],
                                                    cfg['general']['prefix'],
                                                    cfg['general']['type'],
                                                    cfg['general']['iter']))
    plt.show()


if __name__ == "__main__":


    with open(argv[1], 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    f.close()

    print('Config file loaded')
    print('Building {} model'.format(cfg['general']['type']))
    # model, summary_pars, sampled_pars, derived_pars = run_setup(cfg)
    model, summary_pars, sampled_pars, derived_pars = run_setup(cfg)
    print('Model built successfully')
    print('Optimizing MAP Solution')
    model, map_soln = run_map(cfg,model)
    print('Sampling')
    trace = run_NUTS(cfg, model, map_soln, summary_pars)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_sampled_model(ax, cfg, trace)
    plt.show()

    print(sampled_pars)
    plot_posteriors(trace, sampled_pars, derived_pars)
