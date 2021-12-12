import pandas as pd
import sys
import h2o
from h2o.estimators import H2OKMeansEstimator

# some colors for textual representation
class color:
   BLUE = '\033[94m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   END = '\033[0m'

# read input dataset for clustering 
Data = pd.read_csv(sys.argv[1])

# remove text column
Data.pop('Dataset')

# convert to numpy and show data
sample_np=Data.values
print(Data)

# create h2o data frame
h2o.init()
sample = h2o.H2OFrame(sample_np)

# model inicializaton
model=H2OKMeansEstimator(  k=10,
                                estimate_k=True,
                                standardize=True,
                                seed=1234,
                                max_iterations = 100)

# create model for kmeans algorithm
model.train(training_frame=sample)

# show some model metrics
print(model.model_performance())

# apply kmeans to our dataset
result=model.predict(sample)

############################################################# show report on clustered dataset #############################################################

# number of created clusters
number_of_clusters=len(model.size())

# clustered data from h2o frame to pandas data frame
data_as_df = result.as_data_frame(use_pandas=True, header=True)

# init dictionary for individual clusters
clusters={}
for x in range(0,number_of_clusters):
    clusters['cluster{0}'.format(x)]=[]

# insert clustered data to predicted cluster
for i,data in enumerate(sample_np):
    clusters['cluster{0}'.format(data_as_df.values[i][0])].append(data)

for cluster in clusters:
    # number of cluster
    print(color.BOLD + color.RED+ cluster+ color.END)

    # dataframe for effective metrics computation
    df=pd.DataFrame(clusters[cluster],columns=Data.columns)

    # variance of features
    print(color.BOLD +color.BLUE+'\nVariance of features:'+ color.END)
    print(df.var().sort_values().to_string())

    # average of features
    print(color.BOLD +color.BLUE+'\nAverage of features:'+ color.END)
    mean=df.mean()
    original_mean=Data.mean()
    print(mean.sort_values().to_string())
    
    # average deviation of features
    diff=original_mean-mean
    lower=original_mean-Data.min()
    print(color.BOLD +color.BLUE+'\nAverage deviation of features (%):'+ color.END)
    print((-1*(diff/lower)*100).sort_values().to_string())
    
    # correlation of features
    print(color.BOLD +color.BLUE+'\nCorrelation of features:'+ color.END)
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    print(df.corr().unstack().drop(labels=pairs_to_drop).sort_values().to_string())

    # size of cluster
    print(color.BOLD + color.BLUE+'\nSize of cluster:'+ color.END)
    print(len(clusters[cluster]),'\n')
    
