from Schemas.graph_representation import SchemaGraph, Table

def gen_synthetic_schema(csv_path):
    schema = SchemaGraph()
    schema.add_table(Table('synthetic',
                           attributes=['col0', 'col1'],
                           csv_file_location=csv_path.format('synthetic'),
                           table_size=10000000, primary_key=['col0'], sample_rate=1.0
                           ))
    return schema

