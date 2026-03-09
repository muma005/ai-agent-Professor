from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
import json

state = initial_state('test-eda', 'data/spaceship_titanic/train.csv')
state = run_data_engineer(state)
state = run_eda_agent(state)

report = state['eda_report']
required_keys = ['target_distribution','feature_correlations','outlier_profile',
                 'duplicate_analysis','temporal_profile','leakage_fingerprint',
                 'drop_candidates','summary']
for k in required_keys:
    assert k in report, f'Missing key: {k}'
print(json.dumps(report['target_distribution'], indent=2))
print(f"Drop candidates: {report['drop_candidates']}")
print(f"Leakage flags: {[f for f in report['leakage_fingerprint'] if f['verdict']=='FLAG']}")
print('[PASS] EDA report complete')
