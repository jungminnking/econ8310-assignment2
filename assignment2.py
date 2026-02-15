# Drawing Data
train = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/refs/heads/master/AssignmentData/assignment3.csv")
test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/refs/heads/master/AssignmentData/assignment3test.csv")

# Just Exploring...
px.line(train, x='DateTime', y='meal')

# Preprocessing
train['DateTime'] = pd.to_datetime(train['DateTime'])
train['year'] = train['DateTime'].dt.year
train['month'] = train['DateTime'].dt.month
train['day'] = train['DateTime'].dt.weekday
train['hour'] = train['DateTime'].dt.hour
train.head()

test['DateTime'] = pd.to_datetime(test['DateTime'])
test['year'] = test['DateTime'].dt.year
test['month'] = test['DateTime'].dt.month
test['day'] = test['DateTime'].dt.weekday
test['hour'] = test['DateTime'].dt.hour
test.head()

# Defining Factors
trY = train['meal']
trX = train.drop(['meal', 'id', 'DateTime'], axis=1)
teY = test['meal']
teX = test.drop(['meal', 'id', 'DateTime'], axis=1)

# Extreme Boosting; Boosted Trees
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.5, objective='binary:logistic')
modelFit = model.fit(trX, trY)
pred = modelFit.predict(teX)
