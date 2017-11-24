import numpy as np
from numpy import random
import pickle
from scipy.stats import sem

LABEL = "Curb" #Oracle, Baseline, Flagratio
DATASET = "twitter" #weibo

print("Running %s, for %s dataset" % (LABEL, DATASET))

if LABEL == 'Curb' or LABEL == 'Oracle':
    import curb as run
if LABEL == 'Baseline':
    import baseline as run
if LABEL == 'Flagratio':
    import flagratio as run

file_hist = pickle.load(open('./../%s/results/file_hist.pkl' % (DATASET), 'rb'))

a = 1e-4; w = 1e-5

target_files = pickle.load(open('./../%s/results/target_files.pkl' %(DATASET) , 'rb'))

print("")
print("There are %d files to process", len(target_files))
print("")

n_d = len([v for v in target_files if v[0] == 'R'])
p_d = n_d / len(target_files)

p_fd = 0.3; p_fd_bar = 0.01
p_f = p_fd * p_d + p_fd_bar * (1 - p_d)
p_df = p_fd * p_d / p_f
p_df_bar = (1 - p_fd) * p_d / (1 - p_f)
alpha = 100.
iter = 100

if DATASET == 'twitter': ext = -3 
else: ext = -4

file_data = dict()
for file in target_files:
    file_pkl = file[:ext] + 'pkl'
    data = pickle.loads(open('./../%s/exposure_data/' % (DATASET)  + file_pkl  , 'rb').read())
    file_data[file_pkl] = data

q_file = dict()
p_f_expected = p_f
params = [pow(10, vv) for vv in np.arange(0,10) ]
print("Running with q = ["+", ".join([str(x) for x in params])+"]")
print("-"*10)

for q in params:
    if q not in q_file:
        q_file[q] = dict()
    for cnt, file in enumerate(target_files):
        print("parameter q=%f, file=%s"%(q,file))
        file_pkl = file[:ext] + 'pkl'
        data = file_data[file_pkl]
        p_f = p_fd if file[0] == 'R' else p_fd_bar
        user_activity = run.SimulationUserFile(a=a, w=w, data=data, p_f=p_f)
        if LABEL == "Curb":
            policy = run.OptimalPolicy(T = user_activity.T, q=q, p_df=p_df, p_df_bar=p_df_bar, alpha=alpha, beta=alpha * (1/p_f_expected - 1), activity_model=user_activity)
        elif LABEL == 'Oracle':
            policy = run.OraclePolicy(T = user_activity.T, q=q, p_df=p_df, p_df_bar=p_df_bar, p_f=p_f, activity_model=user_activity)
        elif LABEL == 'Baseline':
            policy = run.ExposurePolicy(T = user_activity.T, q=q, activity_model=user_activity)            
        else:
            policy = run.FlagratioPolicy(T = user_activity.T, q=q, p_df=p_df, p_df_bar=p_df_bar, alpha=alpha, beta=alpha * (1/p_f_expected - 1), activity_model=user_activity)            

        for j in range(iter):
            simulation = run.Simulation(policy, user_activity)

            tau = simulation.run()[1]

            if file not in q_file[q]:
                q_file[q][file] = list()
            q_file[q][file].append(tau)

pickle.dump(q_file, open('./../%s/results/q_file_%s_alpha_%d.pkl' % (DATASET, LABEL, int(alpha)), 'wb'))