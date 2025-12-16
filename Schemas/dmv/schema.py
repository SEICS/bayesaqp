from Schemas.graph_representation import SchemaGraph, Table

def gen_dmv_schema(csv_path):
    """
    dmv dataset from Bayescard
    """

    schema = SchemaGraph()

    # tables
    # "Record_Type","VIN","Registration_Class","City","State","Zip","County","Model_Year","Make","Body_Type",
    # "Fuel_Type","Unladen_Weight","Maximum_Gross_Weight","Passengers","Reg_Valid_Date","Reg_Expiration_Date",
    # "Color","Scofflaw_Indicator","Suspension_Indicator","Revocation_Indicator"
    schema.add_table(Table('dmv',
                           attributes=["Record_Type","VIN","Registration_Class","City","State","Zip","County","Model_Year","Make","Body_Type",
                                        "Fuel_Type","Unladen_Weight","Maximum_Gross_Weight","Passengers","Reg_Valid_Date","Reg_Expiration_Date",
                                        "Color","Scofflaw_Indicator","Suspension_Indicator","Revocation_Indicator"],
                           csv_file_location=csv_path.format('dmv'),
                           table_size=12176621, primary_key=['f_flightno'], sample_rate=1,
                           fd_list=[]
                           ))

    return schema

