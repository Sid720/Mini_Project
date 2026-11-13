"""Reproducible, point-in-time retail repeat-purchase experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import urllib.request
import zipfile
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, PrecisionRecallDisplay

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'
FEATURES = ['recency', 'orders', 'spend', 'units', 'products', 'basket_value', 'active_days', 'orders_30', 'spend_30']
SEED = 42


def download():
    folder = ROOT / 'data/raw'
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'source.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist() if n.endswith('.xlsx'))
        target = folder / 'online_retail_II.xlsx'
        if not target.exists():
            target.write_bytes(z.read(member))
    return target


def clean(raw):
    """Positive sales only; exclusions are sequential and auditable."""
    df = raw.rename(columns={'Customer ID': 'customer', 'Invoice': 'invoice', 'InvoiceDate': 'date', 'Quantity': 'quantity', 'Price': 'price', 'StockCode': 'product'}).copy()
    required = ['customer', 'invoice', 'date', 'quantity', 'price', 'product']
    if not set(required).issubset(df.columns):
        raise ValueError('Input must use the original UCI Online Retail II column names.')
    audit = {'raw_rows': len(df)}
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['quantity', 'price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    before = len(df)
    df = df.dropna(subset=required)
    audit['missing_required_removed'] = before - len(df)
    df['customer'] = df.customer.astype(str)
    df['invoice'] = df.invoice.astype(str)
    df['product'] = df['product'].astype(str)
    before = len(df)
    df = df[~df.invoice.str.upper().str.startswith('C') & (df.quantity > 0) & (df.price > 0)].copy()
    audit['cancellations_nonpositive_removed'] = before - len(df)
    before = len(df)
    df = df.drop_duplicates()
    audit['exact_duplicates_removed'] = before - len(df)
    df['value'] = df.quantity * df.price
    if not np.isfinite(df[['quantity','price','value']].to_numpy()).all():
        raise ValueError('Nonfinite monetary or quantity values.')
    audit['clean_rows'] = len(df)
    audit['customers'] = df.customer.nunique()
    return df.sort_values('date'), audit


def features(df, cutoff, lookback=180):
    """Only transactions strictly BEFORE cutoff; eligible = bought in lookback."""
    cutoff = pd.Timestamp(cutoff)
    h = df[(df.date >= cutoff - pd.Timedelta(days=lookback)) & (df.date < cutoff)]
    g = h.groupby('customer')
    f = g.agg(last=('date','max'), orders=('invoice','nunique'), spend=('value','sum'), units=('quantity','sum'), products=('product','nunique'))
    f['recency'] = (cutoff - f.pop('last')).dt.total_seconds() / 86400
    f['basket_value'] = f.spend / f.orders
    f['active_days'] = h.assign(day=h.date.dt.normalize()).groupby('customer').day.nunique()
    recent = h[h.date >= cutoff - pd.Timedelta(days=30)].groupby('customer').agg(orders_30=('invoice','nunique'), spend_30=('value','sum'))
    return f.join(recent).fillna(0)[FEATURES].sort_index()


def snapshot(df, cutoff, observation_end, horizon=30):
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon)
    if end > pd.Timestamp(observation_end):
        raise ValueError('Incomplete future label window.')
    f = features(df, cutoff)
    buyers = set(df.loc[(df.date >= cutoff) & (df.date < end), 'customer'])
    f['target'] = f.index.isin(buyers).astype(int)
    f['cutoff'] = cutoff
    return f.reset_index()


def rank_metrics(y, scores, fraction):
    if not 0 < fraction <= 1:
        raise ValueError('Budget fraction must be in (0,1].')
    y, scores = np.asarray(y), np.asarray(scores)
    k = max(1, math.ceil(len(y) * fraction))
    selected = np.argsort(-scores, kind='stable')[:k]
    hits = int(y[selected].sum())
    prevalence = float(y.mean())
    return {'selected': k, 'buyers_found': hits, 'precision': hits/k,
            'recall': hits/y.sum() if y.sum() else 0.,
            'lift': (hits/k)/prevalence if prevalence else 0.}


def models():
    return {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1., random_state=SEED)),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=15, max_features='sqrt', n_jobs=-1, random_state=SEED)
    }


def run(input_path=None):
    out = ROOT/'outputs'
    out.mkdir(exist_ok=True)
    path = Path(input_path) if input_path else download()
    print('Reading original workbook...', flush=True)
    sheets = pd.read_excel(path, sheet_name=None)
    raw = pd.concat(sheets.values(), ignore_index=True)
    # Exclude the final partial calendar day conservatively.
    observation_end = pd.to_datetime(raw.InvoiceDate).max().normalize()
    df, audit = clean(raw)
    (out/'data_audit.json').write_text(json.dumps(audit, indent=2))
    cutoffs = pd.date_range('2010-06-01', '2011-11-01', freq='MS')
    frames = [snapshot(df, c, observation_end) for c in cutoffs]
    all_data = pd.concat(frames, ignore_index=True)
    train = all_data[all_data.cutoff < '2011-07-01']
    valid = all_data[(all_data.cutoff >= '2011-07-01') & (all_data.cutoff < '2011-09-01')]
    test = all_data[all_data.cutoff >= '2011-09-01']
    for split in [train, valid, test]:
        if split.empty or split.target.nunique() != 2:
            raise ValueError('Each split must contain both classes.')
    assert train.cutoff.max() + pd.Timedelta(days=30) <= valid.cutoff.min()
    assert valid.cutoff.max() + pd.Timedelta(days=30) <= test.cutoff.min()
    split_info = pd.DataFrame([{'split': name, 'first_cutoff': part.cutoff.min(), 'last_cutoff': part.cutoff.max(), 'rows': len(part), 'unique_customers': part.customer.nunique(), 'positive_rate': part.target.mean()} for name,part in [('train',train),('validation',valid),('test',test)]])
    split_info.to_csv(out/'split_summary.csv', index=False)
    candidates = models()
    validation_rows = []
    for name, model in candidates.items():
        print('Fitting', name, flush=True)
        model.fit(train[FEATURES], train.target)
        for date, month in valid.groupby('cutoff'):
            m = rank_metrics(month.target, model.predict_proba(month[FEATURES])[:,1], .2)
            validation_rows.append({'model': name, 'cutoff': date, **m})
    val = pd.DataFrame(validation_rows)
    val.to_csv(out/'validation_metrics.csv', index=False)
    winner = val.groupby('model').precision.mean().idxmax()
    fitting = pd.concat([train, valid])
    predictions = test[['customer','cutoff','target']].copy()
    for name, model in candidates.items():
        model.fit(fitting[FEATURES], fitting.target)
        predictions[name] = model.predict_proba(test[FEATURES])[:,1]
    predictions['recency_baseline'] = -test.recency.to_numpy()
    predictions['spend_baseline'] = test.spend.to_numpy()
    predictions['frequency_baseline'] = test.orders.to_numpy()
    predictions['random_baseline'] = np.random.default_rng(SEED).random(len(test))
    predictions.to_csv(out/'test_predictions.csv', index=False)
    rows = []
    for date, month in predictions.groupby('cutoff'):
        for name in list(candidates) + ['recency_baseline','spend_baseline','frequency_baseline','random_baseline']:
            for budget in [.05,.1,.2,.3,.5]:
                rows.append({'model':name, 'cutoff':date, 'budget_fraction':budget, **rank_metrics(month.target, month[name], budget)})
        for budget in [.05,.1,.2,.3,.5]:
            k = math.ceil(len(month)*budget)
            rows.append({'model':'random_expected', 'cutoff':date, 'budget_fraction':budget, 'selected':k, 'buyers_found':k*month.target.mean(), 'precision':month.target.mean(), 'recall':k/len(month), 'lift':1.})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'monthly_budget_metrics.csv', index=False)
    summary = metrics.groupby(['model','budget_fraction'])[['precision','recall','lift']].mean().reset_index()
    summary.to_csv(out/'budget_summary.csv', index=False)
    classification = pd.DataFrame([{'model':name, 'average_precision':average_precision_score(test.target,predictions[name]), 'roc_auc':roc_auc_score(test.target,predictions[name]), 'brier_score':brier_score_loss(test.target,predictions[name])} for name in candidates])
    classification.to_csv(out/'classification_metrics.csv', index=False)
    joblib.dump({'model':candidates[winner], 'features':FEATURES, 'selected_model':winner, 'horizon_days':30, 'lookback_days':180, 'last_training_label_end':str(valid.cutoff.max()+pd.Timedelta(days=30))}, out/'model.joblib')
    fig, ax = plt.subplots(figsize=(9,5))
    for name, group in summary.groupby('model'):
        if name != 'random_baseline':
            ax.plot(group.budget_fraction*100, group.precision, marker='o', label=name.replace('_',' '))
    ax.set(xlabel='Customers selected (%)', ylabel='Precision: fraction who buy within 30 days', title='Held-out targeting performance (mean across 3 months)')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'budget_precision.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5))
    for name in candidates:
        PrecisionRecallDisplay.from_predictions(test.target, predictions[name], name=name, ax=ax)
    fig.tight_layout(); fig.savefig(out/'precision_recall.png',dpi=180); plt.close(fig)
    importance = pd.Series(candidates['random_forest'].feature_importances_, index=FEATURES).sort_values()
    importance.rename('impurity_importance').to_csv(out/'feature_importance.csv')
    fig, ax = plt.subplots(figsize=(7,5)); importance.plot.barh(ax=ax)
    ax.set(title='Random forest feature importance (not causal)', xlabel='Impurity importance')
    fig.tight_layout(); fig.savefig(out/'feature_importance.png',dpi=180); plt.close(fig)
    meta = {'selected_model':winner, 'selection_rule':'Mean validation precision at 20% contact capacity', 'source_url':URL, 'workbook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'seed':SEED, 'python':platform.python_version(), 'pandas':pd.__version__, 'sklearn':sklearn.__version__, 'observation_end_exclusive':str(observation_end), 'feature_window_days':180, 'label_horizon_days':30}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    chosen = summary[(summary.model==winner)&(summary.budget_fraction==.2)].iloc[0]
    (out/'RESULTS.md').write_text(f'''# Retail Repeat-Purchase Prediction — measured results

Selected model: **{winner}**, chosen using validation months only.

Raw transaction lines: {audit['raw_rows']:,}; cleaned positive-sale lines: {audit['clean_rows']:,}.

At 20% monthly contact capacity, held-out mean precision is **{chosen.precision:.2%}**, recall **{chosen.recall:.2%}**, and lift over random selection **{chosen.lift:.2f}×**.

These are historical buyer-identification results, not demonstrated campaign-driven sales or profit. See budget_summary.csv for comparisons against simple baselines; an ML win is not assumed. Always compare against frequency as well as random selection; a small observed difference is not established statistical superiority.

Training cutoffs: June 2010–June 2011. Validation: July–August 2011. Test: September–November 2011. Each label spans 30 days. Final models are refit on train + validation; test is never used for model selection. Customers may recur across dates because the task is predicting future behaviour of existing customers. Monthly scores are macro-averaged; no independent-customer confidence intervals are claimed.

Eligibility: a known customer with a positive purchase in the preceding 180 days. Excludes cancellations, nonpositive prices/quantities, missing required fields and exact duplicate lines. Returns are not netted against sales; spend is positive-sale value, not profit. A purchase is any positive qualifying invoice within the next 30 days. Dormant customers outside the lookback and anonymous customers are outside scope.
''')
    print((out/'RESULTS.md').read_text(), flush=True)


def score(input_path, cutoff, fraction, destination):
    bundle = joblib.load(ROOT/'outputs/model.joblib')
    sheets = pd.read_excel(input_path, sheet_name=None)
    df, _ = clean(pd.concat(sheets.values(), ignore_index=True))
    f = features(df, cutoff)
    if f.empty:
        raise ValueError('No eligible customer history before cutoff.')
    p = bundle['model'].predict_proba(f[bundle['features']])[:,1]
    result = pd.DataFrame({'customer':f.index, 'repeat_purchase_score':p})
    result = result.sort_values(['repeat_purchase_score','customer'],ascending=[False,True])
    if not 0 < fraction <= 1:
        raise ValueError('Budget fraction must be in (0,1].')
    result['selected'] = np.arange(len(result)) < math.ceil(len(result)*fraction)
    result.to_csv(destination,index=False)
    print(f'Saved {len(result)} customer scores to {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    r = sub.add_parser('run'); r.add_argument('--input',help='Optional original-format workbook')
    s = sub.add_parser('score'); s.add_argument('--input',required=True); s.add_argument('--cutoff',required=True); s.add_argument('--budget',type=float,default=.2); s.add_argument('--output',default='customer_scores.csv')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.input)
    else:
        score(args.input,args.cutoff,args.budget,args.output)
