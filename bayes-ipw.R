library(tidyverse)
library(broom)
library(broom.mixed)
library(brms)
library(cmdstanr)
library(rstan)
library(tidybayes)
library(patchwork)
library(here)

# Code for data comes from https://github.com/andrewheiss/evalsp21.classes.andrewheiss.com/blob/master/content/assignment/03-problem-set.Rmarkdown#L148

# `id`: A unique ID number for each household
# `water_bill`: The family's average monthly water bill, in dollars
# `barrel`: An indicator variable showing if the family participated in the program
# `barrel_num`: A 0/1 numeric version of `barrel`
# `yard_size`: The size of the family's yard, in square feet
# `home_garden`: An indicator variable showing if the family has a home garden
# `home_garden_num`: A 0/1 numeric version of `home_garden`
# `attitude_env`: The family's self-reported attitude toward the environment, on a scale of 1-10 (10 meaning highest regard for the environment)
# `temperature`: The average outside temperature (these get wildly unrealistic for the Atlanta area; just go with it)
barrels_obs <- read_csv(here("data", "barrels_observational.csv")) %>% 
  mutate(barrel = fct_relevel(barrel, "No barrel"))

# Confounders = yard_size + attitude_env + home_garden_num + temperature


# Frequentist matching ----------------------------------------------------

library(MatchIt)
matched <- matchit(barrel_num ~ yard_size + attitude_env + home_garden_num + temperature,
                   data = barrels_obs, method = "nearest", distance = "mahalanobis",
                   replace = TRUE)

barrels_matched <- match.data(matched)


# Use the weights from `matchit()` to fix the imbalance in weighting that
# happens from reusing matched observations
model_matched <- lm(water_bill ~ barrel, data = barrels_matched, weights = weights)
tidy(model_matched, conf.int = TRUE)
tidy(model_matched) %>% filter(term == "barrelBarrel") %>% pull(estimate)

plot_freq_matching <- tidy(model_matched, conf.int = TRUE) %>% 
  filter(term == "barrelBarrel") %>% 
  ggplot(aes(x = estimate, y = term)) +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
  geom_vline(xintercept = -40) +
  labs(title = "ATE from frequentist matching", y = NULL) +
  coord_cartesian(xlim = c(-44, -23))
plot_freq_matching


# Frequentist IPW ---------------------------------------------------------

wants_barrel_model <- glm(barrel ~ yard_size + attitude_env + home_garden + temperature,
                          data = barrels_obs, family = binomial(link = "logit"))

barrel_propensities <- augment(wants_barrel_model, barrels_obs, 
                               type.predict = "response") %>% 
  rename(p_barrel = .fitted)

barrels_ipw <- barrel_propensities %>% 
  mutate(ipw = (barrel_num / p_barrel) + ((1 - barrel_num) / (1 - p_barrel)))

model_ipw <- lm(water_bill ~ barrel, 
                data = barrels_ipw, weights = ipw)
tidy(model_ipw, conf.int = TRUE)
tidy(model_ipw) %>% filter(term == "barrelBarrel") %>% pull(estimate)


plot_freq_ipw <- tidy(model_ipw, conf.int = TRUE) %>% 
  filter(term == "barrelBarrel") %>% 
  ggplot(aes(x = estimate, y = term)) +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
  geom_vline(xintercept = -40) +
  labs(title = "ATE from frequentist IPW", y = NULL) +
  coord_cartesian(xlim = c(-44, -23))
plot_freq_ipw


# Bayesian IPW ------------------------------------------------------------

## Treatment model --------------------------------------------------------

treatment_bf <- bf(
  barrel ~ yard_size + attitude_env + home_garden + temperature,
  family = brmsfamily("bernoulli", "logit"),
  decomp = "QR"
)

treatment_priors <- 
  prior("student_t(3, 0, 2.5)", class = "Intercept") +
  prior("normal(0, 2)", class = "b")

treatment_model <- brm(
  formula = treatment_bf,
  prior = treatment_priors,
  data = barrels_obs,
  chains = 8, 
  cores = 8, 
  iter = 2000, # 1000 warmup/1000 sampling per chain
  seed = 1234, 
  backend = "cmdstanr"
)

pred_probs_chains <- posterior_epred(
  treatment_model,
  cores = 8L,
  seed = 1234
)

ipw_matrix <- t(pred_probs_chains) %>% 
  as_tibble() %>% 
  mutate(barrel_num = barrels_obs$barrel_num, .before = 1) %>% 
  transmute(across(
    starts_with("V"),
    ~ (barrel_num / .x) + ((1 - barrel_num) / (1 - .x))
  ))


# Weights-per-iteration outcome model -------------------------------------

outcome_model <- stan_model(
  stanc_ret = stanc(here("stan", "modified_stan_code.stan"),
                    allow_undefined = TRUE), 
  includes = paste0('\n#include "', here("stan", "iterfuns.hpp"), '"\n')
)

outcome_covariates <- model.matrix(~ barrel_num, data = barrels_obs)

outcome_data <- list(
  N = nrow(barrels_obs),
  Y = barrels_obs$water_bill,
  K = ncol(outcome_covariates),
  X = outcome_covariates,
  L = ncol(ipw_matrix),
  IPW = as.matrix(ipw_matrix),
  prior_only = 0
)

outcome_samples <- sampling(
  outcome_model, 
  data = outcome_data, 
  chains = 8, 
  iter = 2000,
  cores = 8,
  seed = 1234
)

tidy(outcome_samples, conf.int = TRUE)
tidy(outcome_samples) %>% filter(term == "b[1]") %>% pull(estimate)

plot_lots_of_weights_ipw <- outcome_samples %>% 
  tidy_draws() %>% 
  ggplot(aes(x = `b[1]`)) +
  stat_halfeye() +
  geom_vline(xintercept = -40) +
  labs(title = "ATE from different-weights-per-iteration outcome model", y = NULL) +
  coord_cartesian(xlim = c(-44, -23))
plot_lots_of_weights_ipw


## Latent IPW outcome model -----------------------------------------------

cmdstan_ipw <- cmdstan_model(
  stan_file = here("stan", "latent_ipw.stan"),
  dir = here("stan", "compiled"),
  force_recompile = T
)

outcome_covariates <- model.matrix(~ barrel_num, data = barrels_obs)

stan_data_ls <- list(
  N = nrow(barrels_obs),
  Y = barrels_obs$water_bill,
  K = 2,
  X = model.matrix(~ barrel_num, data = barrels_obs),
  IPW_N = 8000,
  IPW = t(ipw_matrix)
)

cmdstan_test_fit <- cmdstan_ipw$sample(
  data = stan_data_ls, 
  seed = 1234, 
  output_dir = here("stan", "compiled"),
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 3000, 
  iter_sampling = 3000, 
  refresh = 50
)

cmdstan_samples <- cmdstan_test_fit$draws(variables = c("Intercept", "beta", "mu_treated", "sigma"))

posterior::summarise_draws(cmdstan_samples)

plot_latent_ipw <- cmdstan_samples %>% 
  tidy_draws() %>% 
  ggplot(aes(x = mu_treated)) +
  stat_halfeye() +
  geom_vline(xintercept = -40) +
  labs(title = "ATE from latent IPW outcome model", y = NULL) +
  coord_cartesian(xlim = c(-44, -23))
plot_latent_ipw


(plot_freq_matching / plot_freq_ipw / plot_lots_of_weights_ipw / plot_latent_ipw) +
  plot_annotation(caption = "Ostensible true ATE is -40")
