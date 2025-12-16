import numpy as np

bayesaqp_opt1_numpy = [18.968466809019446, 23.845758382230997, 23.9416582044214,
                        71.16558339912444, 61.41624159645289, 73.23214178904891] # Inlining
bayesaqp_opt2_numpy = [18.086441792547703, 23.740633204579353, 23.78370021469891,
                        70.76727517414838, 18.867750209756196, 14.359733206219971] # Algebra Simplification
bayesaqp_opt1_torch = [4.615349974483252, 4.864433384500444, 5.365425208583474,
                        36.09415839891881, 32.93633321300149, 37.03601663000882] # Inlining
bayesaqp_opt2_torch = [5.23929160553962, 4.9301247810944915, 5.401033372618258,
                        35.049833240918815, 14.688949822448194, 12.73534162901342] # Algebra Simplification

bayesaqp_opt1_numpy = np.array(bayesaqp_opt1_numpy)
bayesaqp_opt2_numpy = np.array(bayesaqp_opt2_numpy)
bayesaqp_opt1_torch = np.array(bayesaqp_opt1_torch)
bayesaqp_opt2_torch = np.array(bayesaqp_opt2_torch)

sp1 = bayesaqp_opt1_numpy / bayesaqp_opt1_torch
sp2 = bayesaqp_opt2_numpy / bayesaqp_opt2_torch
print("average o1 gpu speedup: ", np.mean(sp1))
print("average o2 gpu speedup: ", np.mean(sp2))
print("\n")
print("max o1 gpu speedup: ", max(sp1))
print("max o2 gpu speedup: ", max(sp2))
print("\n")
print("average speedup: ", np.mean(np.concatenate([sp1, sp2])))
print("max speedup: ", max(np.concatenate([sp1, sp2])))
print("\n")