"""Small numerical checks of utility functions; no data loading or model fitting."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from censml import interval_nll, tobit_grad_hess
from ensembles import conformal_upper, coverage_bounds


def main():
    sigma = .7
    mu = np.array([.2, -.1, .3])
    lo, hi = np.array([0., -np.inf, 1.]), np.array([0., 0., np.inf])
    values = interval_nll(mu, lo, hi, sigma)
    assert np.all(np.isfinite(values))
    expected = .5 * (.2 / sigma)**2 + np.log(sigma) + .5 * np.log(2 * np.pi) + np.log(np.log(10.))
    np.testing.assert_allclose(values[0], expected, atol=1e-12)
    step = 1e-6
    finite = (interval_nll(mu + step, lo, hi, sigma) - interval_nll(mu - step, lo, hi, sigma)) / (2 * step)
    gradient, hessian = tobit_grad_hess(mu, lo, hi, sigma)
    np.testing.assert_allclose(gradient, finite, rtol=1e-5, atol=1e-6)
    assert np.all(hessian > 0)
    lower, upper = coverage_bounds(np.array([0., .5, .3]), np.array([-np.inf, -.2, .4]), np.array([.1, .2, 1.]))
    np.testing.assert_allclose([lower, upper], [1 / 3, 2 / 3])
    bound, quantile, _ = conformal_upper(np.zeros(9), np.ones(9), np.arange(9.), alpha=.1)
    assert quantile == 8
    np.testing.assert_allclose(bound(np.zeros(1), np.array([2.])), [16.])
    # Evaluate the actual public s05 expressions in memory, rather than importing
    # the data-processing module and triggering its top-level file operations.
    tree = ast.parse(Path(__file__).with_name('s05_compliance_blindspot.py').read_text(encoding='utf-8'))
    expressions = {node.targets[0].id: node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['und', 'exc']}
    for threshold in [.08, .07, 1.2, .00065, 4.]:
        tau = np.log10(threshold)
        frame = pd.DataFrame({'cens': [1, 1, 0, 0],
                              'y_hi': [tau, np.nextafter(tau, np.inf), tau, np.nextafter(tau, np.inf)],
                              'y_lo': [-np.inf, -np.inf, tau, np.nextafter(tau, np.inf)]})
        scope = {'np': np, 'd': frame, 'thr': threshold, 'lt': frame.cens.eq(1)}
        for key in ['und', 'exc']:
            actual = eval(compile(ast.Expression(expressions[key]), '<s05 boundary expression>', 'eval'), scope)
            expected = [False, True, False, False] if key == 'und' else [False, False, False, True]
            assert actual.tolist() == expected, (key, threshold, actual.tolist())
    print('PASS: likelihood, finite-difference gradient, interval coverage bounds, conformal rank and strict log-threshold equality.')


if __name__ == '__main__':
    main()
