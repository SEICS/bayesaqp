import os, sys, clr
import faulthandler; faulthandler.enable()
# sys.path.append("~/Desktop/pythonnet")
sys.path.append(os.path.join(os.path.dirname(__file__), r'~/Desktop/infer/test/Tests/bin/Release/net461'))
clr.AddReference("Microsoft.ML.Probabilistic.Compiler")
clr.AddReference("Microsoft.ML.Probabilistic")
clr.AddReference("System")
#clr.AddReference("System.Collections.Generic")
from System.Collections.Generic import List
from Microsoft.ML.Probabilistic.Distributions import VectorGaussian, Beta, Bernoulli, Discrete, DiscreteChar, Dirichlet
from Microsoft.ML.Probabilistic.Math import Vector
from Microsoft.ML.Probabilistic.Models import Variable, Range, InferenceEngine, VariableArray
from Microsoft.ML.Probabilistic.Algorithms import VariationalMessagePassing
from System import Convert
# SELECT COUNT(*) FROM climate WHERE iMay75880 = 0
# SELECT COUNT(*) FROM climate WHERE iMay75880 = 0 AND iSex = 0||1168118
# SELECT COUNT(*) FROM climate WHERE dAncstry1 >= 0 AND dAncstry1 <= 1 AND iClass = 1 AND dIncome5 = 0 AND iPerscare >= 1 AND iPerscare <= 2 AND dPwgt1 = 1 AND iRagechld >= 0 AND iRagechld <= 4 AND iSept80 = 0 AND iSubfam2 = 0||311948
class BN_examples():
    def __init__(self,df):   
        self.df = df
        numOfExamples = len(df.index) # NumberOfExamples 
        N = Range(numOfExamples).Named("N")
        M = Range(2).Named("M")
        S = Range(2).Named("S")
        
        self.ProbiMayPrior = Variable.New[Dirichlet]().Named("ProbiMayPrior")
        self.ProbiMay = Variable[Vector].Random(self.ProbiMayPrior).Named("ProbiMay")
        self.ProbiMay.SetValueRange(M)
        
        self.CPTiSexPrior = Variable.Array[Dirichlet](M).Named("CPTiSexPrior")
        self.CPTiSex = Variable.Array[Vector](M).Named("CPTiSex")
        cpt = Variable[Vector].Random(self.CPTiSexPrior.get_Item(M)).ForEach(M)
        self.CPTiSex.set_Item(M,cpt)
        self.CPTiSex.SetValueRange(S)
        
        self.iMay = Variable.Array[int](N).Named("iMay")
        prob = Variable.Discrete(self.ProbiMay).ForEach(N)
        self.iMay.set_Item(N,prob)
        self.iSex = self.addChildFromOneParent(self.iMay, self.CPTiSex).Named("iSex")
        
        self.engine = InferenceEngine()
        # imay_ = self.df["iMay75880"].tolist()
        # self.iMay = [1 if d>0 else 0 for d in imay_]
        
        # self.ProbiMay.ObservedValue = self.getProb(self.df["iMay75880"])
        # self.CPTiSex.ObservedValue = self.getCPT(self.df["iMay75880"],self.df["iSex"])
        
    # def probiSex(self,imay):
    #     if imay:
    #         self.iMay = imay

    #     CPTiSexPosterior = self.engine.Infer(self.iSex)
    #     print("\nCPTiSexPosterior:",CPTiSexPosterior)
        
    #     return CPTiSexPosterior
    
    # def getProb(self,data):
    #     prob_ = len([d for d in data if d>0])/len(data)
    #     prob = [(1-prob_),prob_] # not P, P
    #     print("\niMay:",prob)
    #     prob = Vector.FromArray((1-prob_),prob_)
    #     return prob
    
    # def getCPT(self,df_p,df_c): # df_p for parent, df_c for children
    #     # P(children|Parent)
    #     cpt_ = len([dc for dp,dc in zip(df_p,df_c) if dp>0 and dc>0])/len(df_c)
    #     cpt = [(1-cpt_),cpt_] # not C, C
    #     # P(children|NOT Parent)
    #     cpt2_ = (len(df_c)-cpt_)/len(df_c)
    #     cpt2 = [(1-cpt2_),cpt2_] # not C, C
    #     print("\niSex",[cpt,cpt2])
        
    #     prob = List[Vector](Range(2))
    #     prob.Add(Vector.FromArray((1-cpt_),cpt_))
    #     prob.Add(Vector.FromArray((1-cpt2_),cpt2_))
    #     return prob
    
    def addChildFromOneParent(self,parent,cpt):
        n = parent.Range
        child = Variable.Array[int](n)
        b1 = Variable.ForEach(n)
        b2 = Variable.Switch()
        c = Variable.Discrete(cpt.get_Item(parent.get_Item(n)))
        child.set_Item(n,c) 
        b2.Dispose()
        b1.Dispose()
        return child
     