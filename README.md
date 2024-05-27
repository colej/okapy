# okapy
Routines for drawing inference on orbital parameters from radial velocity measurements.
** Overview **

First, you need to create a conda environment with the hmc_rv_modelling.yml file provided.
  --> You need to change two things in this:
      --> First, you CAN change the name at the top of the file
      --> Second, you MUST change the path at the bottom of the file to lead to where you
          store the rest of your conda environments

  --> To install the environment, simply run:
      --> conda env create --file hmc_rv_modelling.yml
  --> To enter environment, run:
      --> source activate NAME


To actually run the code, you'll need to place it in its own directory, i.e. rv_hmc_modelling
The code assumes a particular sub-directory structure to organize the data, config files,
solution files, etc. The structure is as follows:

  - rv_hmc_modelling/
    -> hmc_rv_v07.py
    -> configs/  # This is where you put the configuration files
    -> data/     # This is where you put the data
    -> solns/    # This is where the code dumps the output and plots


** Config files **

Great! Now let's take a look at an example Configuration file. These are stored as YAML
files. The file will have a few required inputs, and I'm working on adding some features in
the meantime. But, for now, there are three sections: 1) general, 2) varpars, and 3) priors.
  1) general - contains general information about paths and values used to format the save
     strings. It also contains information used to configure the NUTS sampler, such as the
     number of chains, draws, and tuning samples. In general, these numbers should work
     fairly well, however, if you want to increase these values, feel free! That being said,
     the computation time scales quite quickly with these values.
     - The `target` parameter should never be set below 0.9.
     - There are numerous settings to handle the eccentricity.
     For small eccentricities, `ecc_type: small` is a good choice because of how the
     code samples from a half-cauchy distribution for the value of ecc. Alternatively,
     you can sample sqrt(e)cos(omega) and sqrt(e)sin(omega) from a uniform disc using
     `ecc_type: disc` or you can sample ecc and omega directly from uniform distributions
     using `ecc_type: normal`. Finally, you can force a circular orbit using:
     `ecc_type: circular`.
     - This will assume SB1 as long as `secondary_obs: None` is set. To use an SB2 setup,
       just put a path to the data file. ** You will also need to include a section for
       `k2` under the varpars section. **

  2) varpars - these are the parameters that will be fit during the modelling.
     Try to put the values per variable reasonably close to what you think
     they should be, otherwise the MAP algorithm could struggle. The prior argument
     should be the name of a distribution within pymc. The parameter values are
     sampled from this distribution. There is also the option to transform the
     scale of the data. If `transform: None` is set, then the distribution is sampled
     linearly. If `transform: log` and `prior: Uniform`, then the variable will be sampled
     from a uniform distribution in log-space. Similarly, you can set `prior: Normal`
     and `transform: None` to apply a strict Gaussian prior. The values of `kwargs`
     should be able to be passed to the distribution. For instance, Uniform priors will
     always have lower and upper arguments, whereas Normal priors require mu and sigma
     as arguments. A sensible value for `testval` needs to be set for all variables.

  3) priors - effectively a place-holder for the time being, so don't worry about these!

  general:
    path: ./
    prefix: v772Cas
    iter: i01
    primary_obs: data/v772Cas_primary_RVs.dat
    secondary_obs: None
    chains: 2
    cores: 4
    draws: 1000
    tune: 1000
    target: 0.97
    type: rv
    ecc_type: small
  varpars:
    Period:
      prior: Uniform
      transform: log
      kwargs:
        testval: 5.01370
        lower: 4.8
        upper: 5.2
    Gamma:
      prior: Uniform
      transform: None
      kwargs:
        testval: 0.
        lower: -100.
        upper: 100.
    K1:
      prior: Uniform
      transform: log
      kwargs:
        testval: 60.5
        lower: 1.
        upper: 400
  priors:
    ecc:
      mean: 0.
      std: 0.4
    omega:
      mean: 3.14
      std: 0.1


** Running the code **

Running the code is very simple. Just issue:
  --> python3 hmc_rv_v0X.py configs/hd165246.yaml

It will tell you what its doing along the way.
First, it has to instantiate the model and all of
its symbolic links in pymc3. Then, it will perform
an initial Maximum a-posteriori (MAP) optimisation
and return the best model. A plot will appear for
the MAP solution.

Next, it will use the MAP solution as the starting
point to initialise the NUTS Hamiltonian Monte Carlo
sampling algorithm. After it runs, it will throw
away the tuning samples and marginalize over the
remaining samples to produce our posteriors. A plot
of the binned posteriors will pop up and save,
followed by a plot of the best model from the sampling
and its 16 & 84 percentile models. The best parameters
and plots are saved in the soln folder.
