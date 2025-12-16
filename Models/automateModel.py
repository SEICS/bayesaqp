import os
import pandas as pd
from collections import defaultdict
from anytree import Node, PreOrderIter
from anytree.exporter import DotExporter
import logging
logger = logging.getLogger(__name__)

def tree_structure(relation):
    nodes = []
    if relation:
        itsParents = defaultdict() # dictionary keys are children, values store a list of parents for each child node
        itsChildren = defaultdict() # dictionary keys are parents, values store a list of children for each parent node
        
        for parent, child in relation:
            if parent not in itsChildren.keys():
                itsChildren[parent] = [child]
            else:
                itsChildren[parent].append(child)
                
            if child not in itsParents.keys():
                itsParents[child] = [parent]
            else:
                itsParents[child].append(parent)

            nodes.append(parent)
            nodes.append(child)
    else:
        print("Empty list for Csharp automation")
        
    nodes = set(nodes)
    p_ = nodes - set(itsParents.keys())
    for p in p_:
        itsParents[p] = []
    
    c_ = nodes - set(itsChildren.keys())
    for c in c_:
        itsChildren[c] = []
 
    return itsParents, itsChildren, list(nodes)

def construct_tree(graph_path, itsParents, itsChildren, nodes, bn_count=0):
    nodes_list = {}
    notVisited = nodes.copy()
    while notVisited:
        for n in notVisited:
            if itsParents[n] == []:
                root = Node(n,parent=None) # root
                nodes_list[n] = root
                notVisited.remove(n)
            else:
                p = itsParents[n][0] # this node's parent
                if p in nodes_list.keys():
                    node = Node(n,parent=nodes_list[p])
                    nodes_list[n] = node
                    notVisited.remove(n)
                else:
                    continue

    # Generate the corresponding Bayesian network structure
    # print(f"graph_path: {graph_path}")
    graph_path = os.path.join(graph_path, f"BN_chow_liu_tree_{bn_count}.pdf")
    DotExporter(root).to_picture(graph_path)
    
    # Get all possible path starting from root
    whole_tree = list(PreOrderIter(root, filter_=lambda node: node.is_leaf))
    all_path = [str(i)[7:-2].split(i.separator) for i in whole_tree]
    logger.info(f"BN tree graph stored at: {graph_path}")
    
    return nodes_list, [node.name for node in PreOrderIter(root)], all_path
     
def generate_model(itsParents, itsChildren, nodes, nodes_list, nodes_relation):
    global df
    model = ["\tpublic class BayesCardModel\n\t{\n",
            "\t\tpublic Variable<int> TotalNumberOfExamples;\n",
            "\t\tpublic InferenceEngine Engine = new InferenceEngine();\n\n"]
    
    indent = ["\t"]*2
    root = ""
    
    if nodes:
        for n in nodes:
            model.append("".join(indent) + "public VariableArray<int> " + n + ";\n") # define instances
             
            if len(itsParents[n]) == 0: 
                # root node
                root = n
                for typeOf in ["Vector","Dirichlet"]: # Vector defines Probs and CPTs, Dirichlet defines Priors
                    if typeOf == "Dirichlet":
                        n_ = n + "Prior"
                    else:
                        n_ = n
                        
                    model.append("".join(indent) + "public Variable<" + typeOf + "> Prob" + n_ + ";\n")
                
                model.append("".join(indent) + "public Dirichlet Prob" + n + "Posterior;\n") # define Posteriors
                
            elif len(itsParents[n]) == 1:
                # node dependent on single parent
                for typeOf in ["Vector","Dirichlet"]:
                    if typeOf == "Dirichlet":
                        n_ = n + "Prior"
                    else:
                        n_ = n
                        
                    model.append("".join(indent) + "public VariableArray<" + typeOf + "> CPT" + n_ + ";\n")
                    
                model.append("".join(indent) + "public Dirichlet[] CPT" + n + "Posterior;\n") # define Posteriors
            model.append("\n")
            
        model.append("".join(indent) + "public BayesCardModel()\n" + "".join(indent) + "{\n")
        model_innerEnding = "".join(indent)+"}\n"
        indent.append("\t")
        
        model.append("".join(indent) + 
                     'TotalNumberOfExamples = Variable.New<int>().Named("NofE");\n' +
                     "".join(indent) +
                     'Range N = new Range(TotalNumberOfExamples).Named("N");\n\n')
        
        for n in nodes:
            idx = nodes.index(n) + 1
            model.append("".join(indent) + "Range R" + str(idx) + ' = new Range(' + str(max(set(df[n])) + 1)+ ').Named("R' + str(idx) + " " + n + '");\n')
        model.append("\n")
        
        for n in nodes:
            idx = nodes.index(n) + 1
            if len(itsParents[n]) == 0: 
                # root node
                model.append("".join(indent) + 
                             "Prob" + n + 'Prior = Variable.New<Dirichlet>().Named("Prob' + n + 'Prior");\n' +
                             "".join(indent) +
                             "Prob" + n + ' = Variable<Vector>.Random(Prob' + n + 'Prior).Named("Prob' + n + '");\n' +
                             "".join(indent) +
                             "Prob" + n + ".SetValueRange(R"+ str(idx) +");\n")

            elif len(itsParents[n]) == 1:
                # child node
                p_idx = nodes.index(itsParents[n][0]) + 1 # its parent's index
                model.append("".join(indent) + 
                             "CPT" + n + "Prior = Variable.Array<Dirichlet>(R"+ str(p_idx) +').Named("CPT' + n + 'Prior");\n' +
                             "".join(indent) + 
                             "CPT" + n + " = Variable.Array<Vector>(R"+ str(p_idx) +').Named("CPT' + n + '");\n' +
                             "".join(indent) + 
                             "CPT" + n + "[R"+ str(p_idx) +"] = Variable<Vector>.Random(CPT" + n + "Prior[R"+ str(p_idx) +"]);\n" +
                             "".join(indent) + 
                             "CPT" + n + ".SetValueRange(R"+ str(idx) +");\n")
            model.append("\n")
            
        # define network structure 
        for n in nodes:
            if len(itsParents[n]) == 0: 
                # root node
                model.append("".join(indent) + 
                             n + ' = Variable.Array<int>(N).Named("' + n + '");\n' +
                             "".join(indent) +
                             n + "[N] = Variable.Discrete(Prob" + n + ").ForEach(N);\n\n")
            elif len(itsParents[n]) == 1:
                model.append("".join(indent) + 
                             n + " = AddChildFromOneParent(" + itsParents[n][0] + ", CPT" + n + ').Named("' + n + '");\n')
        
        model.append("\n")
            
    model.insert(-1,model_innerEnding)
    
    # add children to parent method
    model.append("".join(indent[:-1]) + "public static VariableArray<int> AddChildFromOneParent(\n" + 
                 "".join(indent) + "VariableArray<int> p1,\n" +
                 "".join(indent) + "VariableArray<Vector> cpt)\n" + 
                 "".join(indent[:-1]) + "{\n" +
                 "".join(indent) + "var n = p1.Range;\n" + 
                 "".join(indent) + "var child = Variable.Array<int>(n);\n" + 
                 "".join(indent) + "using (Variable.ForEach(n))\n" +
                 "".join(indent) + "using (Variable.Switch(p1[n]))\n" + 
                 "".join(indent) + "{\n" + 
                 "".join(indent) + "\tchild[n] = Variable.Discrete(cpt[p1[n]]);\n" + 
                 "".join(indent) + "}\n" + 
                 "".join(indent) + "return child;\n" + 
                 "".join(indent[:-1]) + "}\n\n")
    
    # get probabilities for each node/attribute:
    # Query method
    model.append("".join(indent[:-1]) + "public double[] query(\n")
    model.append("".join(indent) + "string target,\n")
    for n in nodes:
        model.append("".join(indent) + "int? " + n.lower() + ",\n")
                    
    for n in nodes:  
        if itsParents[n]:
            temp = "Dirichlet[] cpt" + n + "Prior"
        else:
            temp = "Dirichlet prob" + n + "Prior"
            
        if n == nodes[-1]:
            model.append("".join(indent) + temp + ")\n")
        else:
            model.append("".join(indent) + temp + ",\n")
            
    model.append("".join(indent[:-1]) + "{\n")
    model.append("".join(indent) + "TotalNumberOfExamples.ObservedValue = 1;\n")
    attr_lst = ['"' + n + '"' for n in nodes]
    model.append("".join(indent) + "string[] attributes = new string[]{ " + ",".join(attr_lst) + " };\n")
    
    for n in nodes:
        model.append("".join(indent) + "if (" + n.lower() + ".HasValue)\n" + \
                     "".join(indent) + "\t" + n + ".ObservedValue = new int[] { " + n.lower() + ".Value };\n" + \
                     "".join(indent) + "else\n" + \
                     "".join(indent) + "\t" + n + ".ClearObservedValue();\n\n")
    
    for n in nodes:
        if itsParents[n]:
            n1 = "CPT"
        else:
            n1 = "Prob"
        model.append("".join(indent) + n1 + n + "Prior" + ".ObservedValue = " + n1.lower() + n + "Prior;\n")
    
    # doing inference
    model.append("".join(indent) + "Discrete[] resultPosterior;\n")
    for m in range(0,len(nodes)):
        if m == 0:
            con_prefix = "if"
            condition = " (target == attributes[" + str(m) + "])\n"
        elif m == len(nodes)-1:
            con_prefix = "else"
            condition = "\n"
        else:
            con_prefix = "else if"
            condition = " (target == attributes[" + str(m) + "])\n"
            
        model.append("".join(indent) + con_prefix + condition + \
                     "".join(indent) + "{\n" + \
                     "".join(indent) + "\t" + nodes[m] + ".ClearObservedValue();\n" + \
                     "".join(indent) + "\tresultPosterior = Engine.Infer<Discrete[]>(" + nodes[m] + ");\n" + \
                     "".join(indent) + "}\n")
    
    model.append("".join(indent) + "return resultPosterior[0].GetProbs().ToArray();\n" + \
                 "".join(indent[:-1]) + "}\n\n")
    
    # learn parameters 1
    depth_arr = {}
    for n in nodes:
        d = nodes_list[n].depth
        if n not in depth_arr.keys():
            depth_arr[n] = d
    
    # diff_depth = set(depth_arr.values())
    
    model.append("".join(indent[:-1]) + "public void LearnParameters(\n")
    for n in nodes:
        model.append("".join(indent) + "int[] " + n.lower() + ",\n")
    
    for n in nodes:
        name_prefix = " prob" if nodes_list[n].is_root else "[] cpt"
        end = ")\n" if nodes.index(n) == len(nodes)-1 else ",\n"
        model.append("".join(indent) + "Dirichlet" + name_prefix + n.lower() + "Prior" + end)
    
    # indent = indent[:-1]
    model.append("".join(indent[:-1]) + "{\n")
    
    model.append("".join(indent) + "TotalNumberOfExamples.ObservedValue = " + nodes[0].lower() + ".Length;\n")
    
    for n in nodes:
        model.append("".join(indent) + n + ".ObservedValue = " + n.lower() + ";\n")
    model.append("\n")
    
    for n in nodes:
        name_prefix1 = "Prob" if nodes_list[n].is_root else "CPT"
        name_prefix2 = "prob" if nodes_list[n].is_root else "cpt"
        model.append("".join(indent) + name_prefix1 + n + "Prior" + ".ObservedValue = " + name_prefix2 + n.lower() + "Prior" + ";\n")
    model.append("\n")
    
    for n in nodes:
        name_prefix1 = "Prob" if nodes_list[n].is_root else "CPT"
        dimension = "" if nodes_list[n].is_root else "[]" 
        model.append("".join(indent) + name_prefix1 + n + "Posterior = Engine.Infer<Dirichlet" + dimension + ">(" + name_prefix1 + n + ");\n" )

    model.append("".join(indent[:-1]) + "}\n\n")
    
    # learn parameters 2
    model.append("".join(indent[:-1]) + "public void LearnParameters(\n")
    for n in nodes:
        end = ")\n" if nodes.index(n) == len(nodes)-1 else ",\n"
        model.append("".join(indent) + "int[] " + n.lower() + end)
    
    model.append("".join(indent[:-1]) + "{\n")
    
    val = {}
    for n in nodes:
        val[n] = max(set(df[n])) + 1
    
    for n in nodes:
        if not itsParents[n]:
            model.append("".join(indent) + "Dirichlet " + n + "Array = Dirichlet.Uniform(" + str(val[n]) + ");\n")
        else:
            p = itsParents[n][0]
            model.append("".join(indent) + "Dirichlet[] " + n + "Array = new Dirichlet[" + str(val[p]) + "];\n")
            model.append("".join(indent) + "for (int i = 0; i < " + str(val[p]) + "; i++)\n" + \
                         "".join(indent) + "\t" + n + "Array[i] = Dirichlet.Uniform(" + str(val[n]) + ");\n")

    # for d in diff_depth: 
    #     if d == 0:
    #         model.append("".join(indent) + "Dirichlet" + " " + str(d+1) + "dArray = Dirichlet.Uniform(2);\n")
    #     else:
    #         model.append("".join(indent) + "Dirichlet" + "[]" * d + " " + str(d+1) + "dArray = Enumerable.Repeat(" + str(d) + "dArray, 2).ToArray();\n")
    
    # model.append("".join(indent) + "Dirichlet arr1 = Dirichlet.Uniform(2);\n")
    # model.append("".join(indent) + "Dirichlet[] arr2 = Enumerable.Repeat(arr1, 2).ToArray();\n")
    
    # for n in nodes:
    #     arr = "arr1" if nodes_list[n].is_root else "arr2"
    #     parameters.append(arr)
    p1 = [n.lower() for n in nodes]
    p2 = [n+"Array" for n in nodes]
    parameters = [*p1, *p2]
        
    model.append("".join(indent) + "LearnParameters" + "(" + ",".join(parameters) + ");\n")
    model.append("".join(indent[:-1]) + "}\n")
    
    # model defining finished
    model.append("\t}\n")
    return "".join(model)

def generate_BayesCardBNmodel_cSharp(relation):
    global df
    itsParents, itsChildren, nodes = tree_structure(relation)
    nodes_list, nodes, allPath = construct_tree(itsParents, itsChildren, nodes) # produce tree graph "BN_Chow_liu_tree.png"
    nodes_relation = findSingleNodePath(allPath) 
    df = df[nodes] # reorder the discrete_table for input automation
    
    lib =  "using System;\n" + \
           "using Microsoft.ML.Probabilistic.Models;\n" + \
           "using Microsoft.ML.Probabilistic.Math;\n" + \
           "using Microsoft.ML.Probabilistic.Distributions;\n" + \
           "using Range = Microsoft.ML.Probabilistic.Models.Range;\n" + \
           "using System.Data;\n" + \
           "using System.IO;\n" + \
           "using Microsoft.Data.Analysis;\n" + \
           "using System.Collections.Generic;\n" + \
           "using System.Diagnostics;\n" + \
           "using Microsoft.ML.Probabilistic.Algorithms;\n" + \
           "using System.Text.Json;\n" + \
           "using System.Linq;\n\n"
            
    model = generate_model(itsParents=itsParents, itsChildren=itsChildren, nodes=nodes, nodes_list=nodes_list, nodes_relation=nodes_relation)
    innerCode = model
    
    return lib + "namespace learningdotnet\n{\n" + innerCode + "}\n"

def write_file(relation):
    save_path = "~/Projects/learningdotnet/learningdotnet"
    file_name = "BayesCardModelAutomation.cs"
    data_name = "Census.csv"
    path = os.path.join(save_path,file_name)
    path2 = os.path.join(save_path,data_name)
    with open(path,"w") as f:
        code = generate_BayesCardBNmodel_cSharp(relation)
        f.write(code)
    f.close()
    
    df.to_csv(path2, index=False, sep=',', encoding='utf-8') 
    df.to_csv("discrete_table.csv", index=False, sep=',', encoding='utf-8') 

def findSingleNodePath(allPath): 
    # find the path to each node starting from root, 
    # where dictionary key is the node, and its value 
    # is the path to it (previous visited node for this path)
    nodesPath = {}  
    paths = [path[:i] for path in allPath for i in range(1, len(path)+1)]

    for path in paths:
        node = path[-1]
        if node not in nodesPath.keys():
            if len(path) == 1:
                nodesPath[node] = []
            else:
                nodesPath[node] = path[:-1]
        else:
            continue
    return nodesPath    

    
# -------------------------- automate Queries --------------------------
if __name__ == "__main__":
    census = [('iLang1', 'dAncstry1'), ('dAncstry1', 'dAncstry2'), ('iLooking', 'iAvail'), ('iRPOB', 'iCitizen'), ('dIndustry', 'iClass'), ('dTravtime', 'dDepart'), ('iDisabl2', 'iDisabl1'), ('iYearwrk', 'iDisabl2'), ('iLang1', 'iEnglish'), ('iRvetserv', 'iFeb55'), ('iRelat1', 'iFertil'), ('dAncstry1', 'dHispanic'), ('iWork89', 'dHour89'), ('iRlabor', 'dHours'), ('iRPOB', 'iImmigr'), ('dRearning', 'dIncome1'), ('iClass', 'dIncome2'), ('dOccup', 'dIncome3'), ('dRpincome', 'dIncome4'), ('dAge', 'dIncome5'), ('dRpincome', 'dIncome6'), ('dAge', 'dIncome7'), ('dRpincome', 'dIncome8'), ('dOccup', 'dIndustry'), ('iRvetserv', 'iKorean'), ('iYearsch', 'iLang1'), ('iRlabor', 'iLooking'), ('iRspouse', 'iMarital'), ('iRvetserv', 'iMay75880'), ('dHours', 'iMeans'), ('iRlabor', 'iMilitary'), ('iLang1', 'iMobility'), ('iDisabl1', 'iMobillim'), ('iYearwrk', 'dOccup'), ('iRvetserv', 'iOthrserv'), ('iMobillim', 'iPerscare'), ('dAncstry1', 'dPOB'), ('iRelat1', 'dPoverty'), ('iRPOB', 'dPwgt1'), ('iFertil', 'iRagechld'), ('dHour89', 'dRearning'), ('iRspouse', 'iRelat1'), ('iRelat1', 'iRelat2'), ('iRrelchld', 'iRemplpar'), ('iMeans', 'iRiders'), ('iYearwrk', 'iRlabor'), ('iRemplpar', 'iRownchld'), ('dRearning', 'dRpincome'), ('dPOB', 'iRPOB'), ('dAge', 'iRrelchld'), ('dAge', 'iRspouse'), ('iMilitary', 'iRvetserv'), ('dAge', 'iSchool'), ('iRvetserv', 'iSept80'), ('iRagechld', 'iSex'), ('iRelat1', 'iSubfam1'), ('iSubfam1', 'iSubfam2'), ('iRlabor', 'iTmpabsnt'), ('iMeans', 'dTravtime'), ('iRvetserv', 'iVietnam'), ('dRearning', 'dWeek89'), ('iYearwrk', 'iWork89'), ('iRlabor', 'iWorklwk'), ('iRvetserv', 'iWWII'), ('dAge', 'iYearsch'), ('dAge', 'iYearwrk'), ('iRvetserv', 'dYrsserv')]
    dmv = [('Record_Type', 'Registration_Class'), ('County', 'State'), ('Registration_Class', 'County'), ('Body_Type', 'Model_Year'), ('Registration_Class', 'Body_Type'), ('Body_Type', 'Fuel_Type'), ('County', 'Scofflaw_Indicator'), ('Body_Type', 'Suspension_Indicator'), ('Registration_Class', 'Revocation_Indicator')]
    parents, children, nodes = tree_structure(relation=census)
    nodes_lis, nodes_name, allPath = construct_tree(parents, children, nodes) # produce tree graph "BN_Chow_liu_tree.png"
    # print(str(nodes_name).replace("'",'"'))
    # print(allPath)
    # nodes_relation = findSingleNodePath(allPath)
    # print(nodes_relation)
    # print(len(nodes_relation))
    # print(len(nodes_relation.keys()))
    # write_file(relation)