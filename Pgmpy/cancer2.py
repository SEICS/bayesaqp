import sys
sys.path.append('~/Desktop/bayes-aqp/')
import pgmpy
from Pgmpy.factors.discrete import TabularCPD
from Models.Bayescard_BN import Bayescard_BN
from collections import defaultdict
from Pgmpy.models.BayesianModel import BayesianModel
from Pgmpy.base import DAG
cancer_struct = [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Cancer', 'Dyspnoea'), ('Cancer', 'Xray')]
cancer_model = BayesianModel(cancer_struct)
cpd_Pollution = TabularCPD(variable='Pollution', variable_card=3, values=[[0.5], [0.4], [0.1]])
cpd_Smoker = TabularCPD(variable='Smoker', variable_card=2, values=[[0.3], [0.7]])
cpd_Cancer = TabularCPD(variable='Cancer', variable_card=2, values=[[0.03, 0.001, 0.03, 0.001, 0.05, 0.02], [0.97, 0.999, 0.97, 0.999, 0.95, 0.98]], evidence=['Pollution', 'Smoker'], evidence_card=[3, 2],)
cpd_Dyspnoea = TabularCPD(variable='Dyspnoea', variable_card=2, values=[[0.65, 0.3], [0.35, 0.7]], evidence=['Cancer'], evidence_card=[2],)
cpd_Xray = TabularCPD(variable='Xray', variable_card=2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=['Cancer'], evidence_card=[2],)
cancer_model.add_cpds(cpd_Pollution,cpd_Smoker,cpd_Cancer,cpd_Dyspnoea,cpd_Xray)
cancer_model.check_model()
BN = Bayescard_BN('cancer')
BN.model = cancer_model
BN.cpds = cancer_model.cpds
BN.infer_algo = 'exact-jit'
BN.root = BN.model.get_roots()
BN.init_inference_method()
# print(BN.infer_machine.query(variables=['Pollution','Smoker','Cancer','Dyspnoea','Xray']).values)
print(BN.infer_machine.query({'Pollution':[0,1,2],'Smoker':[0,1],'Cancer':[0,1],'Dyspnoea':[0,1],'Xray':[0,1]}).values)