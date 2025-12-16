import sys
sys.path.append('~/Desktop/bayes-aqp/')
import pgmpy
from Pgmpy.factors.discrete import TabularCPD
from Models.Bayescard_BN import Bayescard_BN
from collections import defaultdict
from Pgmpy.models.BayesianModel import BayesianModel
from Pgmpy.base import DAG
from time import perf_counter
survey_struct = [('A', 'E'), ('S', 'E'), ('E', 'R'), ('E', 'O'), ('O', 'T'), ('R', 'T')]
survey_model = BayesianModel(survey_struct)
cpd_A = TabularCPD(variable='A', variable_card=3, values=[[0.3], [0.5], [0.2]])
cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])
cpd_E = TabularCPD(variable='E', variable_card=2, values=[[0.75, 0.64, 0.72, 0.7, 0.88, 0.9], [0.25, 0.36, 0.28, 0.3, 0.12, 0.1]], evidence=['A', 'S'], evidence_card=[3, 2],)
cpd_R = TabularCPD(variable='R', variable_card=2, values=[[0.25, 0.2], [0.75, 0.8]], evidence=['E'], evidence_card=[2],)
cpd_O = TabularCPD(variable='O', variable_card=2, values=[[0.96, 0.92], [0.04, 0.08]], evidence=['E'], evidence_card=[2],)
cpd_T = TabularCPD(variable='T', variable_card=3, values=[[0.48, 0.58, 0.56, 0.7], [0.42, 0.24, 0.36, 0.21], [0.1, 0.18, 0.08, 0.09]], evidence=['O', 'R'], evidence_card=[2, 2],)
survey_model.add_cpds(cpd_A,cpd_S,cpd_E,cpd_R,cpd_O,cpd_T)
survey_model.check_model()
BN = Bayescard_BN('survey')
BN.model = survey_model
BN.cpds = survey_model.cpds
BN.infer_algo = 'exact-jit'
BN.root = BN.model.get_roots()
BN.init_inference_method()
start = perf_counter()
probs, writefiletime = BN.infer_machine.optimised_multi_stage_inference(variables=['A','S','E','R','O','T'], lib='numpy') # lib='opt_einsum'
print(f"probs: {probs}\ninference time taken: {perf_counter()-start-writefiletime}ms")