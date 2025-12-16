from Schemas.graph_representation import SchemaGraph, Table

def gen_job_light_imdb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'title', 'imdb_index', 'phonetic_code', 'season_nr',
                                                  'imdb_id', 'episode_nr', 'series_years', 'md5sum'],
                           no_compression=['kind_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=24988000))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           irrelevant_attributes=['nr_order', 'note', 'person_id', 'person_role_id'],
                           no_compression=['role_id'],
                           table_size=63475800))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           no_compression=['keyword_id'],
                           table_size=7522600))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           irrelevant_attributes=['note'],
                           no_compression=['company_id', 'company_type_id'],
                           table_size=4958300))

    # relationships
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    return schema

def gen_500gb_ssb_schema(csv_path):
    """
    SSB schema for SF=500.
    """

    schema = SchemaGraph()

    # tables
    # lineorder
    schema.add_table(Table('lineorder',
                           attributes=['lo_orderkey', 'lo_linenumber', 'lo_custkey', 'lo_partkey', 'lo_suppkey',
                                       'lo_orderdate', 'lo_orderpriority', 'lo_shippriority', 'lo_quantity',
                                       'lo_extendedprice', 'lo_ordertotalprice', 'lo_discount', 'lo_revenue',
                                       'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmode'],
                           csv_file_location=csv_path.format('lineorder_sampled'),
                           irrelevant_attributes=['lo_commitdate'],
                           table_size=3000028242, primary_key=['lo_orderkey', 'lo_linenumber'], sample_rate=0.003333))

    # dwdate
    # dwdate.d_dayofweek -> dwdate.d_daynuminweek
    # dwdate.d_dayofweek -> dwdate.d_lastdayinweekfl
    # dwdate.d_month -> dwdate.d_monthnuminyear
    # dwdate.d_monthnuminyear -> dwdate.d_sellingseason
    # dwdate.d_daynuminyear -> dwdate.d_weeknuminyear
    schema.add_table(
        Table('dwdate',
              attributes=['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum', 'd_yearmonth',
                          'd_daynuminweek', 'd_daynuminmonth', 'd_daynuminyear', 'd_monthnuminyear', 'd_weeknuminyear',
                          'd_sellingseason', 'd_lastdayinweekfl', 'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl'],
              csv_file_location=csv_path.format('date'),
              table_size=2556, primary_key=["d_datekey"],
              fd_list=[('d_dayofweek', 'd_daynuminweek'), ('d_dayofweek', 'd_lastdayinweekfl'),
                       ('d_month', 'd_monthnuminyear'), ('d_monthnuminyear', 'd_sellingseason'),
                       ('d_daynuminyear', 'd_weeknuminyear')]))

    # customer
    # customer.c_city -> customer.c_nation
    # customer.c_nation -> customer.c_region
    schema.add_table(
        Table('customer',
              attributes=['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region', 'c_phone',
                          'c_mktsegment'],
              csv_file_location=csv_path.format('customer'),
              table_size=15000000, primary_key=["c_custkey"],
              fd_list=[('c_city', 'c_nation'), ('c_nation', 'c_region')]))

    # part
    # part.p_brand1 -> part.p_category
    # part.p_category -> part.p_mfgr
    schema.add_table(
        Table('part',
              attributes=['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color', 'p_type', 'p_size',
                          'p_container'],
              csv_file_location=csv_path.format('part'),
              table_size=1800000, primary_key=["p_partkey"],
              fd_list=[('p_category', 'p_mfgr'), ('p_brand1', 'p_category')]))

    # supplier
    # supplier.s_city -> supplier.s_nation
    # supplier.s_nation -> supplier.s_region
    schema.add_table(
        Table('supplier', attributes=['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region', 's_phone'],
              csv_file_location=csv_path.format('supplier'),
              table_size=1000000, primary_key=["s_suppkey"],
              fd_list=[('s_city', 's_nation'), ('s_nation', 's_region')]))

    # relationships
    schema.add_relationship('lineorder', 'lo_custkey', 'customer', 'c_custkey')
    schema.add_relationship('lineorder', 'lo_partkey', 'part', 'p_partkey')
    schema.add_relationship('lineorder', 'lo_suppkey', 'supplier', 's_suppkey')
    schema.add_relationship('lineorder', 'lo_orderdate', 'dwdate', 'd_datekey')
    # schema.add_relationship('lineorder', 'lo_commitdate', 'dwdate', 'd_datekey')

    return schema

def gen_imdb_schema(csv_path):
    """
    Specifies full imdb schema. Also tables not in the job-light benchmark.
    """
    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           table_size=24988000))

    # info_type
    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           table_size=63475800))

    # char_name
    schema.add_table(Table('char_name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf',
                                                    'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('char_name'),
                           table_size=4314870))

    # role_type
    schema.add_table(Table('role_type', attributes=['id', 'role'],
                           csv_file_location=csv_path.format('role_type'),
                           table_size=0))

    # complete_cast
    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))

    # comp_cast_type
    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=0))

    # name
    schema.add_table(Table('name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf',
                                               'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('name'),
                           table_size=6379740))

    # aka_name
    schema.add_table(Table('aka_name', attributes=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf',
                                                   'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('aka_name'),
                           table_size=1312270))

    # movie_link, is empty
    # schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
    #                        csv_file_location=csv_path.format('movie_link')))

    # link_type, no relationships
    # schema.add_table(Table('link_type', attributes=['id', 'link'],
    #                        csv_file_location=csv_path.format('link_type')))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           table_size=7522600))

    # keyword
    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=236627))

    # person_info
    schema.add_table(Table('person_info', attributes=['id', 'person_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('person_info'),
                           table_size=4130210))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           table_size=4958300))

    # company_name
    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf',
                                                       'name_pcode_sf', 'md5sum'],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=362131))

    # company_type
    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=0))

    # aka_title
    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                                    'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
                                                    'episode_nr', 'note', 'md5sum'],
                           irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=528268))

    # kind_type
    schema.add_table(Table('kind_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=0))

    # relationships

    # title
    # omit self-join for now
    # schema.add_relationship('title', 'episode_of_id', 'title', 'id')
    schema.add_relationship('title', 'kind_id', 'kind_type', 'id')

    # movie_info_idx
    schema.add_relationship('movie_info_idx', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')

    # movie_info
    schema.add_relationship('movie_info', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')

    # info_type, no relationships

    # cast_info
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'person_id', 'name', 'id')
    schema.add_relationship('cast_info', 'person_role_id', 'char_name', 'id')
    schema.add_relationship('cast_info', 'role_id', 'role_type', 'id')

    # char_name, no relationships

    # role_type, no relationships

    # complete_cast
    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id')
    schema.add_relationship('complete_cast', 'status_id', 'comp_cast_type', 'id')
    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id')

    # comp_cast_type, no relationships

    # name, no relationships

    # aka_name
    schema.add_relationship('aka_name', 'person_id', 'name', 'id')

    # movie_link, is empty
    # schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id')
    # schema.add_relationship('movie_link', 'linked_movie_id', 'title', 'id')
    # schema.add_relationship('movie_link', 'movie_id', 'title', 'id')

    # link_type, no relationships

    # movie_keyword
    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')

    # keyword, no relationships

    # person_info
    schema.add_relationship('person_info', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('person_info', 'person_id', 'name', 'id')

    # movie_companies
    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id')
    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    # company_name, no relationships

    # company_type, no relationships

    # aka_title
    schema.add_relationship('aka_title', 'movie_id', 'title', 'id')
    schema.add_relationship('aka_title', 'kind_id', 'kind_type', 'id')

    # kind_type, no relationships

    return schema
