from Schemas.graph_representation import SchemaGraph, Table


def gen_census_schema(csv_path):
    """
    census 1990 
    """

    schema = SchemaGraph()

    # tables
    # "caseid","dAge","dAncstry1","dAncstry2","iAvail","iCitizen",
    # "iClass","dDepart","iDisabl1","iDisabl2","iEnglish","iFeb55",
    # "iFertil","dHispanic","dHour89","dHours","iImmigr","dIncome1",
    # "dIncome2","dIncome3","dIncome4","dIncome5","dIncome6","dIncome7","dIncome8",
    # "dIndustry","iKorean","iLang1","iLooking","iMarital","iMay75880","iMeans","iMilitary",
    # "iMobility","iMobillim","dOccup","iOthrserv","iPerscare","dPOB","dPoverty","dPwgt1",
    # "iRagechld","dRearning","iRelat1","iRelat2","iRemplpar","iRiders","iRlabor","iRownchld",
    # "dRpincome","iRPOB","iRrelchld","iRspouse","iRvetserv","iSchool","iSept80","iSex","iSubfam1",
    # "iSubfam2","iTmpabsnt","dTravtime","iVietnam","dWeek89","iWork89","iWorklwk","iWWII",
    # "iYearsch","iYearwrk","dYrsserv"
    
    schema.add_table(Table('Census',
                           attributes=["caseid","dAge","dAncstry1","dAncstry2","iAvail","iCitizen",
                                        "iClass","dDepart","iDisabl1","iDisabl2","iEnglish","iFeb55",
                                        "iFertil","dHispanic","dHour89","dHours","iImmigr","dIncome1",
                                        "dIncome2","dIncome3","dIncome4","dIncome5","dIncome6","dIncome7","dIncome8",
                                        "dIndustry","iKorean","iLang1","iLooking","iMarital","iMay75880","iMeans","iMilitary",
                                        "iMobility","iMobillim","dOccup","iOthrserv","iPerscare","dPOB","dPoverty","dPwgt1",
                                        "iRagechld","dRearning","iRelat1","iRelat2","iRemplpar","iRiders","iRlabor","iRownchld",
                                        "dRpincome","iRPOB","iRrelchld","iRspouse","iRvetserv","iSchool","iSept80","iSex","iSubfam1",
                                        "iSubfam2","iTmpabsnt","dTravtime","iVietnam","dWeek89","iWork89","iWorklwk","iWWII",
                                        "iYearsch","iYearwrk","dYrsserv"],
                           csv_file_location=csv_path.format('census'),
                           irrelevant_attributes=["caseid"],
                           table_size=2458286, primary_key=['caseid'], sample_rate=1,
                           fd_list=[]
                           ))

    return schema