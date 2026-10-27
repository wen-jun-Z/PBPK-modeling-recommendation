"""Official sbi SNPE-C adapted to the frozen four-axis human PBPK inverse problem.

No molecular predictor is trained here. All simulated observations are generated
by the supplied PBPK function at the visible development times. SNPE-C's proposal
correction is supplied by sbi, not reimplemented. Target observations are not an
argument. Dependencies are isolated under this project's runtime/sbi-python.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEPS = ROOT / 'runtime/sbi-python'
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

import torch
from torch.distributions import Distribution, constraints

VERSION = 'official-sbi-snpe-c-pbpk-adaptation-1'
DEFAULT_CONFIG = dict(rounds=3, simulations_per_round=4096, posterior_draws=2048,
                      max_epochs=150, stop_after_epochs=20,
                      training_batch_size=256, hidden_features=64,
                      num_transforms=5, sigma_log=.3, device='cuda')


class EvidencePrior(Distribution):
    """Uniform log-CL; bounded zero-centered normal priors on three other axes."""
    arg_constraints = {}
    has_rsample = False

    def __init__(self, lower, upper, sd, device='cpu'):
        self.lower = torch.as_tensor(lower, dtype=torch.float32, device=device)
        self.upper = torch.as_tensor(upper, dtype=torch.float32, device=device)
        self.sd = torch.as_tensor(sd, dtype=torch.float32, device=device)
        if self.lower.shape != (4,) or self.upper.shape != (4,) or self.sd.shape != (3,):
            raise ValueError('Exactly four log-parameter axes and three normal SDs required')
        if not torch.all(self.lower < self.upper) or not torch.all(self.sd > 0):
            raise ValueError('Invalid prior bounds or standard deviations')
        self.a, self.b = self.lower[1:] / self.sd, self.upper[1:] / self.sd
        self.cdf_a = torch.special.ndtr(self.a)
        self.cdf_b = torch.special.ndtr(self.b)
        self.mass = self.cdf_b-self.cdf_a
        if not torch.all(self.mass > 1e-7):
            raise ValueError('Numerically empty truncated normal prior')
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([4]), validate_args=False)

    @property
    def support(self):
        return constraints.independent(constraints.interval(self.lower, self.upper), 1)

    @property
    def mean(self):
        phi = lambda z: torch.exp(-.5*z*z)/math.sqrt(2*math.pi)
        normal_mean = self.sd*(phi(self.a)-phi(self.b))/self.mass
        return torch.cat([((self.lower[0]+self.upper[0])/2).reshape(1), normal_mean])

    @property
    def variance(self):
        phi = lambda z: torch.exp(-.5*z*z)/math.sqrt(2*math.pi)
        delta = (phi(self.a)-phi(self.b))/self.mass
        normal_var = self.sd**2*(1+(self.a*phi(self.a)-self.b*phi(self.b))/self.mass-delta**2)
        return torch.cat([((self.upper[0]-self.lower[0])**2/12).reshape(1),normal_var])

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape)
        u = torch.rand(shape+(4,), device=self.lower.device)
        cl = self.lower[0]+u[..., :1]*(self.upper[0]-self.lower[0])
        p = self.cdf_a+u[..., 1:]*self.mass
        p = p.clamp(torch.finfo(p.dtype).eps, 1-torch.finfo(p.dtype).eps)
        other = self.sd*torch.special.ndtri(p)
        return torch.cat([cl, other], dim=-1)

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.lower.device)
        p = -torch.log(self.upper[0]-self.lower[0])
        p = p+(-.5*(value[..., 1:]/self.sd)**2-torch.log(self.sd*self.mass)
               -.5*math.log(2*math.pi)).sum(-1)
        valid = torch.isfinite(value).all(-1)&(value >= self.lower).all(-1)&(value <= self.upper).all(-1)
        return torch.where(valid, p, torch.full_like(p, -torch.inf))


class _NoFileTracker:
    """Retain the returned fit diagnostics, without one TensorBoard file per fit."""
    log_dir = str(ROOT/'runtime/cache/published-inference')
    def log_metric(self, *args, **kwargs): pass
    def log_metrics(self, *args, **kwargs): pass
    def log_params(self, *args, **kwargs): pass
    def add_figure(self, *args, **kwargs): pass
    def flush(self): pass


def fit(predict, observed_log, initial, lower, upper, *, method='snpe_c', seed=20261011,
        rounds=3, simulations_per_round=4096, posterior_draws=2048,
        max_epochs=150, stop_after_epochs=20, training_batch_size=256,
        hidden_features=64, num_transforms=5, sigma_log=.3, device='cuda', **unused):
    """Return log-parameter posterior medians and samples from official SNPE-C."""
    from sbi.inference import SNPE_C
    from sbi.neural_nets import posterior_nn
    import sbi

    if method != 'snpe_c':
        raise ValueError('This entry point only implements SNPE-C')
    if unused:
        raise ValueError('Unknown SNPE configuration: '+', '.join(unused))
    obs = np.asarray(observed_log, float)
    lo, hi, center = map(lambda x: np.asarray(x, float), (lower, upper, initial))
    if obs.ndim != 1 or len(obs) < 4 or not np.isfinite(obs).all():
        raise ValueError('At least four finite visible log-concentrations required')
    if not callable(predict) or lo.shape != (4,) or hi.shape != (4,) or center.shape != (4,):
        raise ValueError('Four-parameter forward simulator required')
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo >= hi):
        raise ValueError('Finite increasing bounds required')
    if not math.isfinite(sigma_log) or sigma_log <= 0:
        raise ValueError('Positive observation-noise SD required')
    for x in (rounds, simulations_per_round, posterior_draws, max_epochs,
              stop_after_epochs, training_batch_size, hidden_features, num_transforms):
        if isinstance(x, bool) or not isinstance(x, int) or x < 1:
            raise ValueError('Positive integer inference budgets required')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Protocol requested CUDA, but CUDA is unavailable')

    begin = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(seed)
    sd = sigma_log*np.array([math.log(10), math.log(100), math.log(10)])/(math.sqrt(len(obs))*.1)
    prior = EvidencePrior(lo, hi, sd, device=device)
    inference = SNPE_C(prior=prior,
        density_estimator=posterior_nn(model='maf', hidden_features=hidden_features,
                                       num_transforms=num_transforms),
        device=device, tracker=_NoFileTracker(), show_progress_bars=False)
    x_o = torch.as_tensor(obs, dtype=torch.float32, device=device)
    proposal = prior
    traces = []
    calls = failures = 0
    for r in range(rounds):
        round_start = time.monotonic()
        if r == 0:
            theta = proposal.sample((simulations_per_round,))
        else:
            theta = proposal.sample((simulations_per_round,), show_progress_bars=False,
                                    max_sampling_batch_size=simulations_per_round)
        rows = []
        for index, values in enumerate(theta.detach().cpu().numpy()):
            calls += 1
            try:
                simulated = np.asarray(predict(values.astype(float)), float)
                if simulated.shape != obs.shape or not np.isfinite(simulated).all():
                    raise ValueError('Nonfinite or wrong-shape simulated log-concentrations')
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                raise ValueError('Invalid SNPE simulation; filtering would bias sequential inference') from exc
            rows.append(simulated+rng.normal(0, sigma_log, len(obs)))
        # Every proposed simulation must be retained: filtering even one failed
        # trajectory changes the sequential proposal and can bias SNPE-C.
        x = torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=device)
        inference.append_simulations(theta, x, proposal=proposal)
        density = inference.train(training_batch_size=training_batch_size,
            max_num_epochs=max_epochs, stop_after_epochs=stop_after_epochs,
            learning_rate=5e-4, validation_fraction=.1, show_train_summary=False,
            force_first_round_loss=False, retrain_from_scratch=False)
        proposal = inference.build_posterior(density).set_default_x(x_o)
        summary = inference.summary
        traces.append(dict(round=r+1, simulations=simulations_per_round,
            valid_simulations=len(rows), epochs=int(summary['epochs_trained'][-1]),
            best_validation_loss=float(summary['best_validation_loss'][-1]),
            wall_seconds=time.monotonic()-round_start,
            proposal='prior' if r == 0 else 'previous_round_corrected_posterior'))
    samples = proposal.sample((posterior_draws,), show_progress_bars=False,
                               max_sampling_batch_size=posterior_draws).detach().cpu().numpy().astype(float)
    if samples.shape != (posterior_draws, 4) or not np.isfinite(samples).all():
        raise ArithmeticError('Invalid neural posterior draws')
    if np.any(samples < lo-1e-6) or np.any(samples > hi+1e-6):
        raise ArithmeticError('Neural posterior left registered parameter bounds')
    point = np.median(samples, axis=0)
    simulation_diagnostics = []
    for values in samples[np.linspace(0, len(samples)-1, min(64, len(samples)), dtype=int)]:
        calls += 1
        pred = np.asarray(predict(values), float)
        simulation_diagnostics.append(float(np.mean((pred-obs)**2)))
    result = dict(version=VERSION, method=method, point_parameters=point.tolist(),
        posterior_samples=samples.tolist(), selected_indices=[0,1,2,3],
        simulator_calls=calls, failures=failures, wall_seconds=time.monotonic()-begin,
        diagnostics=dict(backend='sbi.SNPE_C', backend_version=sbi.__version__,
            device=device, parameter_space='natural-log multipliers', rounds=traces,
            posterior_draws=posterior_draws, posterior_summary='componentwise median of log multipliers',
            convergence_assessed=False, convergence_note='Flow validation is not MCMC convergence or calibrated posterior coverage.',
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.startswith('cuda') else 0,
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith('cuda') else 0,
            simulated_validation_only=True, target_observations_used=False,
            sigma_log=sigma_log, prior_sd=sd.tolist(),
            prior='bounded uniform log CL; zero-centered truncated normals on log K/exchange/RBC',
            sampled_development_log_mse_quantiles=np.quantile(simulation_diagnostics,[.05,.5,.95]).tolist(),
            proposal_correction='official SNPE-C atomic loss after first round',
            reference='Greenberg et al. ICML2019 APT; official sbi0.27.0 PBPK adaptation'))
    del inference, proposal, density, theta, x
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return result
