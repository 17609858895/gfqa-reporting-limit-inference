"""Base learners for the censored-concentration benchmark.

Every learner is called as fit(Xtr, lotr, hitr, Xva, lova, hiva) and returns a
predictor f(X) -> mu on log10 scale. Substitution learners regress on DL/2 for
non-detects; censoring-aware learners consume the interval labels directly.
"""
from __future__ import annotations

import os

# a shared, often saturated workstation: avoid OpenMP busy-waiting and oversubscription
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import numpy as np
import pandas as pd

from censml import LN10, interval_nll, tobit_grad_hess, tobit_linear, linear_matrix, CATS

N_THREADS = 6
SUB_SMALL = 30_000   # training subsample for O(n^2)-ish or slow learners
HALF = np.log10(2.0)


def subst(lo, hi, shift=HALF):
    return np.where(np.isfinite(lo), lo, hi - shift)


def _sub(X, *arrs, n=SUB_SMALL, seed=0):
    if len(X) <= n:
        return (X, *arrs)
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return (X.iloc[idx] if hasattr(X, "iloc") else X[idx], *[a[idx] for a in arrs])


# ------------------------------------------------------------------ linear family
class LinearWrap:
    """Fits a train-only design (impute, indicators, scale, one-hot) then a sklearn estimator."""

    def __init__(self, est, feats, small=False):
        self.est, self.feats, self.small = est, feats, small

    def __call__(self, Xtr, lotr, hitr, Xva, lova, hiva):
        y = subst(lotr, hitr)
        if self.small:
            Xtr, y = _sub(Xtr, y)
        self.ref = Xtr
        A, _ = linear_matrix(Xtr, Xtr.iloc[:2], self.feats)
        self.est.fit(A, y)
        return lambda X: self.est.predict(linear_matrix(self.ref, X, self.feats)[1])


def tobit_lin(feats):
    def fit(Xtr, lotr, hitr, Xva, lova, hiva):
        A, _ = linear_matrix(Xtr, Xtr.iloc[:2], feats)
        beta, s = tobit_linear(A, lotr, hitr, l2=1e-3)
        return lambda X: np.c_[np.ones(len(X)), linear_matrix(Xtr, X, feats)[1]] @ beta
    return fit


# ------------------------------------------------------------------- tree family
def tree_l2(kind, feats):
    """Native-NaN tree learners on a numeric matrix with ordinal-coded categoricals."""
    def enc(X, cats_ref):
        Z = X.copy()
        for c in CATS:
            Z[c] = pd.Categorical(Z[c].astype(str), categories=cats_ref[c]).codes.astype(float)
        return Z.to_numpy(np.float32)

    def fit(Xtr, lotr, hitr, Xva, lova, hiva):
        from sklearn import ensemble, tree
        y = subst(lotr, hitr)
        cats_ref = {c: sorted(Xtr[c].astype(str).unique()) for c in CATS}
        A = enc(Xtr, cats_ref)
        if kind in ("rf", "et", "dt", "ada", "knn"):
            med = np.nan_to_num(np.nanmedian(np.where(np.all(np.isnan(A), 0), 0.0, A), axis=0))
            A = np.where(np.isnan(A), med, A)
        if kind == "hgb":
            m = ensemble.HistGradientBoostingRegressor(max_iter=300, learning_rate=0.15, early_stopping=True,
                                                       validation_fraction=0.1, random_state=0)
        elif kind == "rf":
            m = ensemble.RandomForestRegressor(200, min_samples_leaf=5, max_features=0.5, n_jobs=N_THREADS, random_state=0,
                                               max_samples=min(1.0, 40_000 / len(A)))
        elif kind == "et":
            m = ensemble.ExtraTreesRegressor(200, min_samples_leaf=5, max_features=0.7, n_jobs=N_THREADS, random_state=0,
                                             bootstrap=True, max_samples=min(1.0, 40_000 / len(A)))
        elif kind == "dt":
            m = tree.DecisionTreeRegressor(max_depth=14, min_samples_leaf=20, random_state=0)
        elif kind == "ada":
            A, y = _sub(A, y)
            m = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=6), n_estimators=150, learning_rate=0.1, random_state=0)
        elif kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.preprocessing import StandardScaler
            A, y = _sub(A, y)
            sc = StandardScaler().fit(A)
            m = KNeighborsRegressor(25, weights="distance", n_jobs=N_THREADS)
            m.fit(sc.transform(A), y)
            return lambda X: m.predict(sc.transform(np.where(np.isnan(enc(X, cats_ref)), med, enc(X, cats_ref))))
        m.fit(A, y)
        if kind in ("rf", "et", "dt", "ada"):
            return lambda X: m.predict(np.where(np.isnan(enc(X, cats_ref)), med, enc(X, cats_ref)))
        return lambda X: m.predict(enc(X, cats_ref))
    return fit


LGB_P = dict(learning_rate=0.08, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=N_THREADS, max_cat_to_onehot=8)


def lgb_l2(Xtr, lotr, hitr, Xva, lova, hiva):
    import lightgbm as lgb
    m = lgb.train({**LGB_P, "objective": "l2"}, lgb.Dataset(Xtr, subst(lotr, hitr)), 2000,
                  valid_sets=[lgb.Dataset(Xva, subst(lova, hiva))], callbacks=[lgb.early_stopping(60, verbose=False)])
    return lambda X: m.predict(X, num_iteration=m.best_iteration)


def lgb_tobit(Xtr, lotr, hitr, Xva, lova, hiva):
    """LightGBM with an interval-censored Gaussian (Tobit) objective; sigma profiled on validation."""
    import lightgbm as lgb
    from censml import fit_sigma
    mu0 = float(np.median(subst(lotr, hitr)))
    sigma = float(np.std(subst(lotr, hitr)))
    for _ in range(2):
        def fobj(preds, data, s=sigma):
            return tobit_grad_hess(preds + mu0, lotr, hitr, s)

        def feval(preds, data, s=sigma):
            return "cnll", float(interval_nll(preds + mu0, lova, hiva, s).mean()), False
        dtr = lgb.Dataset(Xtr, np.zeros(len(Xtr)), free_raw_data=False)
        m = lgb.train({**LGB_P, "objective": fobj, "metric": "None"}, dtr, 2000,
                      valid_sets=[lgb.Dataset(Xva, np.zeros(len(Xva)), reference=dtr)], feval=feval,
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        sigma = fit_sigma(m.predict(Xva, num_iteration=m.best_iteration) + mu0, lova, hiva)
    return lambda X: m.predict(X, num_iteration=m.best_iteration) + mu0


def xgb_l2(Xtr, lotr, hitr, Xva, lova, hiva):
    import xgboost as xgb
    p = dict(objective="reg:squarederror", tree_method="hist", eta=0.08, max_depth=8, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.8, nthread=N_THREADS, max_cat_to_onehot=1)
    dtr = xgb.DMatrix(Xtr, subst(lotr, hitr), enable_categorical=True)
    dva = xgb.DMatrix(Xva, subst(lova, hiva), enable_categorical=True)
    m = xgb.train(p, dtr, 2000, evals=[(dva, "va")], early_stopping_rounds=60, verbose_eval=False)
    return lambda X: m.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, m.best_iteration + 1))


def xgb_aft(dist):
    def fit(Xtr, lotr, hitr, Xva, lova, hiva):
        import xgboost as xgb

        def dm(X, lo, hi):
            d = xgb.DMatrix(X, enable_categorical=True)
            d.set_float_info("label_lower_bound", np.where(np.isfinite(lo), 10.0 ** lo, 0.0))
            d.set_float_info("label_upper_bound", np.where(np.isfinite(hi), 10.0 ** hi, np.inf))
            return d
        p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution=dist,
                 aft_loss_distribution_scale=1.0, tree_method="hist", eta=0.08, max_depth=8, min_child_weight=5,
                 subsample=0.8, colsample_bytree=0.8, nthread=N_THREADS, max_cat_to_onehot=1)
        m = xgb.train(p, dm(Xtr, lotr, hitr), 2000, evals=[(dm(Xva, lova, hiva), "va")],
                      early_stopping_rounds=60, verbose_eval=False)
        return lambda X: m.predict(xgb.DMatrix(X, enable_categorical=True), output_margin=True,
                                   iteration_range=(0, m.best_iteration + 1)) / LN10
    return fit


def _cb_pool(X, label=None):
    import catboost as cb
    Z = X.copy()
    for c in CATS:
        Z[c] = Z[c].astype(str)
    return cb.Pool(Z, label=label, cat_features=CATS)


def cat_l2(Xtr, lotr, hitr, Xva, lova, hiva):
    import catboost as cb
    m = cb.CatBoostRegressor(iterations=600, learning_rate=0.15, depth=6, loss_function="RMSE", verbose=0, one_hot_max_size=64, border_count=64,
                             thread_count=N_THREADS, od_type="Iter", od_wait=60)
    m.fit(_cb_pool(Xtr, subst(lotr, hitr)), eval_set=_cb_pool(Xva, subst(lova, hiva)), use_best_model=True)
    return lambda X: m.predict(_cb_pool(X))


def cat_aft(Xtr, lotr, hitr, Xva, lova, hiva):
    import catboost as cb

    def lab(lo, hi):
        return np.c_[np.where(np.isfinite(lo), 10.0 ** lo, 0.0), np.where(np.isfinite(hi), 10.0 ** hi, -1.0)]
    m = cb.CatBoostRegressor(iterations=600, learning_rate=0.15, depth=6, loss_function="SurvivalAft:dist=Normal;scale=1.0", one_hot_max_size=64, border_count=64,
                             verbose=0, thread_count=N_THREADS, od_type="Iter", od_wait=60)
    m.fit(_cb_pool(Xtr, lab(lotr, hitr)), eval_set=_cb_pool(Xva, lab(lova, hiva)), use_best_model=True)
    return lambda X: m.predict(_cb_pool(X), prediction_type="RawFormulaVal") / LN10


# ------------------------------------------------------------------- neural family
def mlp_fit(feats, censored):
    """Torch MLP (3x128) trained with L2 on DL/2 labels or with the Tobit interval likelihood."""
    def fit(Xtr, lotr, hitr, Xva, lova, hiva):
        import torch
        torch.set_num_threads(N_THREADS)
        torch.manual_seed(0)
        A, B = linear_matrix(Xtr, Xva, feats)
        At, Bt = torch.tensor(A), torch.tensor(B)
        net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        log_s = torch.nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32))
        opt = torch.optim.AdamW(list(net.parameters()) + [log_s], lr=2e-3, weight_decay=1e-4)
        y_sub = torch.tensor(subst(lotr, hitr), dtype=torch.float32)
        hi_t = torch.tensor(np.where(np.isfinite(hitr), hitr, 0), dtype=torch.float32)
        cen_t = torch.tensor(~np.isfinite(lotr))
        normal = torch.distributions.Normal(0.0, 1.0)

        def loss_fn(mu, idx):
            if not censored:
                return ((mu - y_sub[idx]) ** 2).mean()
            s = log_s.exp()
            c = cen_t[idx]
            z_det = (y_sub[idx] - mu) / s
            nll_det = 0.5 * z_det ** 2 + log_s
            z_c = (hi_t[idx] - mu) / s
            nll_c = -torch.special.log_ndtr(z_c)
            return torch.where(c, nll_c, nll_det).mean()

        best, best_state, wait = np.inf, None, 0
        n = len(A)
        yva_sub = subst(lova, hiva)
        for ep in range(60):
            net.train()
            perm = torch.randperm(n)
            for b in range(0, n, 1024):
                idx = perm[b:b + 1024]
                opt.zero_grad()
                loss = loss_fn(net(At[idx]).squeeze(-1), idx)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Bt).squeeze(-1).numpy()
            v = interval_nll(pv, lova, hiva, float(log_s.exp())).mean() if censored else np.mean((pv - yva_sub) ** 2)
            if v < best - 1e-4:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= 6:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        def pred(X):
            with torch.no_grad():
                return net(torch.tensor(linear_matrix(Xtr, X, feats)[1])).squeeze(-1).numpy()
        return pred
    return fit


# ----------------------------------------------------------------- survival family
def rsf_fit(feats):
    """Random survival forest on reflected concentrations (left -> right censoring)."""
    def fit(Xtr, lotr, hitr, Xva, lova, hiva):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        y_obs = subst(lotr, hitr, shift=0.0)          # DL for non-detects
        Xs, ys, ev = _sub(Xtr, y_obs, np.isfinite(lotr), n=20_000)
        shift = ys.max() + 1.0
        t = shift - ys                                  # reflected time, >0
        A, _ = linear_matrix(Xs, Xs.iloc[:2], feats)
        m = RandomSurvivalForest(n_estimators=150, min_samples_leaf=15, max_features="sqrt", n_jobs=N_THREADS,
                                 random_state=0, low_memory=False)
        m.fit(A, Surv.from_arrays(ev, t))
        times = m.unique_times_

        def pred(X):
            B = linear_matrix(Xs, X, feats)[1]
            out = np.empty(len(B))
            for i in range(0, len(B), 2000):
                S = m.predict_survival_function(B[i:i + 2000], return_array=True)
                # median reflected time -> median concentration
                k = (S <= 0.5).argmax(axis=1)
                k = np.where((S <= 0.5).any(axis=1), k, len(times) - 1)
                out[i:i + 2000] = shift - times[k]
            return out
        return pred
    return fit


def make_learners(feats):
    from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
    from sklearn.svm import LinearSVR
    return {
        # substitution (DL/2) family
        "Ridge": ("Substitution", LinearWrap(Ridge(1.0), feats)),
        "Lasso": ("Substitution", LinearWrap(Lasso(1e-3, max_iter=3000), feats)),
        "ElasticNet": ("Substitution", LinearWrap(ElasticNet(1e-3, l1_ratio=0.5, max_iter=3000), feats)),
        "Huber": ("Substitution", LinearWrap(HuberRegressor(max_iter=300), feats, small=True)),
        "LinearSVR": ("Substitution", LinearWrap(LinearSVR(epsilon=0.0, C=0.1, max_iter=5000, dual="auto"), feats, small=True)),
        "KNN": ("Substitution", tree_l2("knn", feats)),
        "DecisionTree": ("Substitution", tree_l2("dt", feats)),
        "AdaBoost": ("Substitution", tree_l2("ada", feats)),
        "RandomForest": ("Substitution", tree_l2("rf", feats)),
        "ExtraTrees": ("Substitution", tree_l2("et", feats)),
        "HistGB": ("Substitution", tree_l2("hgb", feats)),
        "XGBoost": ("Substitution", xgb_l2),
        "LightGBM": ("Substitution", lgb_l2),
        "CatBoost": ("Substitution", cat_l2),
        "MLP": ("Substitution", mlp_fit(feats, censored=False)),
        # censoring-aware family
        "Tobit": ("Censored", tobit_lin(feats)),
        "XGB-AFT-N": ("Censored", xgb_aft("normal")),
        "XGB-AFT-L": ("Censored", xgb_aft("logistic")),
        "CatBoost-AFT": ("Censored", cat_aft),
        "LGB-Tobit": ("Censored", lgb_tobit),
        "DeepTobit": ("Censored", mlp_fit(feats, censored=True)),
        "RSF": ("Censored", rsf_fit(feats)),
    }
