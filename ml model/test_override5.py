import pandas as pd
cf = pd.read_parquet('/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_features_v1.parquet')
p = cf[(cf['athlete_id']==27623) & (cf['split_id']=='ALL__ALL')]
print(len(p))
print(p[['teamId', 'shots_total', 'minutes_total']].to_dict('records'))
