data {
  // Data for the outcome model
  int<lower = 0> N; // Total Number of Observations
  vector[N] Y; // Response Vector
  int<lower = 1> K; // Number of Population-Level Effects
  matrix[N, K] X; // Design Matrix for the Population-Level Effects
  // Data from the Design Stage Model
  int<lower = 1> IPW_N; // Number of Rows in the Weights Matrix
  matrix[IPW_N, N] IPW; // Matrix of IP Weights from the Design Stage Model
}

transformed data {
  // Data for the latent weights
  vector[N] gamma_w; // Mean of the IP Weights
  vector[N] delta_w; // SD of the IP Weights
  // Calculate the location and scale for each observation weight
  for (i in 1:N) {
    gamma_w[i] = mean(IPW[, i]);
    delta_w[i] = sd(IPW[, i]);
  }
  // Centering the Predictor Matrix
  int Kc = K - 1;
  matrix[N, Kc] Xc;  // Centered Version of X without an Intercept
  vector[Kc] means_X;  // Column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}

parameters {
  real alpha; // Population-Level Intercept
  vector[Kc] beta; // Population-Level Effects
  real<lower=0> sigma;  // Dispersion Parameter
  vector<lower=0>[N] weights_z; // Standardized IPW Weights
}

transformed parameters {
  vector[N] weights_tilde; // Latent IPW Weights
  weights_tilde = gamma_w + delta_w .* weights_z;
}

model {
  // Likelihood
  vector[N] mu = alpha + Xc * beta;
  for (n in 1:N) {
    target += (normal_lpdf(Y[n] | mu[n], sigma)) * weights_tilde[n];
  }
  // Sampling the Weights
  target += exponential_lpdf(weights_z | 1);
  // Priors for the model parameters
  target += student_t_lpdf(alpha | 3, 0, 2.5);
  target += normal_lpdf(beta | -40, 6);
  target += student_t_lpdf(sigma | 3, 0, 10) - 1 * student_t_lccdf(0 | 3, 0, 10);
}

generated quantities {
  // Actual Population-Level Intercept
  real Intercept = alpha - dot_product(means_X, beta);
  // Average Treatment Effect
  real mu_treated = (Intercept + beta[1]) - Intercept;
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = (normal_lpdf(Y[n] | alpha + Xc[n] * beta, sigma)) * weights_tilde[n];
  }
}
