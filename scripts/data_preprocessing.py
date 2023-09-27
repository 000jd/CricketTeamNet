import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

output_dir = 'data/preprocessed'

data = pd.read_csv('data/match.csv')

Selected_Featers = [
                    'inning',
                    'player',
                    'bat_inning',
                    'ball_inning',
                    'ball_faced',
                    'run_scored',
                    '4s',
                    '6s',
                    '50s',
                    '100s',
                    'ball_delivered',
                    'run_given',
                    'wicket',
                    'catch',
                    'stump',
                    'run_out',
                    'dismissed_by',
                    'dismissal_type',
                    'dream11_score'
                    ]
X = data[Selected_Featers].values

data_numeric = data.drop(['player', 'dismissed_by', 'dismissal_type'], axis=1)


scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(data_numeric)

X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=data_numeric.columns)

X_processed = pd.concat([data['player'], X_numeric_scaled_df], axis=1)

input_features = X_processed.drop('player', axis=1).values
input_features

y = X_processed['player'].values
X_train, X_test, y_train, y_test = train_test_split(input_features, y, test_size=0.2, random_state=42)

train_df = pd.DataFrame(X_train, columns=X_processed.columns[1:])
train_df['player'] = y_train
train_df.to_csv(os.path.join(output_dir, 'preprocessed_train.csv'), index=False)

test_df = pd.DataFrame(X_test, columns=X_processed.columns[1:])
test_df['player'] = y_test
test_df.to_csv(os.path.join(output_dir, 'preprocessed_test.csv'), index=False)