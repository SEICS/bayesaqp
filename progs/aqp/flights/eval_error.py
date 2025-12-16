import numpy as np
import pickle
def eval_gb(pred, true):
    rel_err = None
    q_err = None# evaluate groupby
    rel_err = []
    q_err = []
    for t in true:
        for p in pred:
            # if all([tt==pp for (tt, pp) in zip(t[:-1],p[:-1])]):
            print(f"t: {t}")
            print(f"p: {p}")
            new_t = float(t[-1])
            rel_err.append(abs(new_t-p[-1])/new_t)
            if p[-1] < 1:
                q_err.append(round(max(1/new_t, new_t), 3))
            else:
                q_err.append(round(max(p[-1]/new_t, new_t/p[-1]), 3))
    rel_err = np.mean(rel_err) * 100
    q_err = np.mean(q_err)
    
    return rel_err, q_err

def eval(pred, true):
    true = float(true[0][0])
    # print(f"true: {true}. pred: {pred}")
    rel_err = abs(pred-true)/true*100
    if pred < 1:
        pred = 1
    q_err = max(pred/true, true/pred)
    return rel_err, q_err

def map_gb():
    

if __name__ == "__main__":
    with open("~/Desktop/bayes-aqp/Benchmarks/flights/ground_truth_1B.pkl", "rb+") as f:
        gt = pickle.load(f)
        
    # flights: only q1, q2 without gb. 
    pred = [1817962.15692789, 1314984.55072701, 1156236.93825767, 991898.58162347, 867967.45412991, 627500.57576843, 572033.97322935, 537623.51826619, 465119.33597434, 459359.37028122, 275270.19433063, 271422.52812502, 215118.83662095, 199015.49202967, 176416.80738784, 148848.31011762, 137431.31237717, 134536.89557318, 106454.63110493, 82150.39986686, 69446.37585798, 55042.45527001, 24513.42097939, 20610.2213786, 16617.1967488, 12418.46704586]
    # err = eval(pred, gt[1])
    err = eval_gb(pred, gt[2])
    print(f"err: {err}")
    