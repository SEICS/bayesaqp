from Schemas.graph_representation import SchemaGraph, Table

def gen_10gb_tpc_ds_store_sales_schema(csv_path):
    """
    TPCDS 10g schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('store_sales',
                           attributes=['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk',
                                       'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk',
                                       'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                       'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price',
                                       'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax',
                                       'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
                           irrelevant_attributes=['ss_sold_time_sk', 'ss_customer_sk', 'ss_cdemo_sk',
                                                  'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_wholesale_cost',
                                                  'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost',
                                                  'ss_ext_list_price', 'ss_ext_tax', 'ss_net_paid', 'ss_net_paid_inc_tax'],
                           no_compression=['ss_sold_date_sk', 'ss_item_sk', 'ss_store_sk', 'ss_ticket_number', 'ss_quantity', 'ss_list_price', 'ss_sales_price', 'ss_coupon_amt', 'ss_net_profit'],
                           csv_file_location=csv_path.format('store_sales'),
                           table_size=28800991, primary_key=['ss_item_sk', 'ss_ticket_number'],
                           sample_rate=10000000 / 28800991
                           ))
    return schema

def gen_10gb_tpc_ds_schema(csv_path):
    """
    TPCDS 10g schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('store_sales',
                           attributes=['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk',
                                       'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk',
                                       'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                       'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price',
                                       'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax',
                                       'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
                           irrelevant_attributes=['ss_sold_time_sk', 'ss_customer_sk', 'ss_cdemo_sk',
                                                  'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_wholesale_cost',
                                                  'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost',
                                                  'ss_ext_list_price', 'ss_ext_tax', 'ss_net_paid', 'ss_net_paid_inc_tax'],
                           no_compression=['ss_sold_date_sk', 'ss_item_sk', 'ss_store_sk', 'ss_ticket_number', 'ss_quantity', 'ss_list_price', 'ss_sales_price', 'ss_coupon_amt', 'ss_net_profit'],
                           csv_file_location=csv_path.format('store_sales'),
                           table_size=28800991, primary_key=['ss_item_sk', 'ss_ticket_number'],
                           sample_rate=10000000 / 28800991
                           ))
    
    schema.add_table(Table('store',
                           attributes=['s_store_sk', 's_store_id', 's_rec_start_date', 's_rec_end_date', 's_closed_date_sk', 's_store_name',
                                       's_number_employees', 's_floor_space', 's_hours', 'S_manager', 'S_market_id', 'S_geography_class', 
                                       'S_market_desc', 's_market_manager', 's_division_id', 's_division_name', 's_company_id', 's_company_name',
                                       's_street_number', 's_street_name', 's_street_type', 's_suite_number', 's_city', 's_county', 's_state', 
                                       's_zip', 's_country','s_gmt_offset', 's_tax_percentage'],
                           irrelevant_attributes=['s_rec_start_date', 's_rec_end_date', 's_closed_date_sk', 's_store_name',
                                                's_number_employees', 's_floor_space', 's_hours', 'S_manager', 'S_market_id', 'S_geography_class', 
                                                'S_market_desc', 's_market_manager', 's_division_id', 's_division_name', 's_company_id', 's_company_name',
                                                's_street_number', 's_street_name', 's_street_type', 's_suite_number', 's_city', 's_county','s_state', 
                                                's_zip', 's_country', 's_gmt_offset', 's_tax_percentage'],
                           no_compression=['s_store_sk', 's_store_id'],
                           csv_file_location=csv_path.format('store'),
                           table_size=102, primary_key=['s_store_sk'],
                           sample_rate=10000000 / 102
                           ))

    schema.add_relationship('store_sales', 'ss_store_sk', 'store', 's_store_sk') # relationship identifier (Join on ss_store_sk)
    return schema


def gen_1t_tpc_ds_schema(csv_path):
    """
    TPCDS 1t schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('store_sales',
                           attributes=['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk',
                                       'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk',
                                       'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                       'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price',
                                       'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax',
                                       'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
                           irrelevant_attributes=['ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk',
                                                  'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_ticket_number',
                                                  'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                                  'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost',
                                                  'ss_ext_list_price', 'ss_ext_tax',
                                                  'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax',
                                                  'ss_net_profit'],
                           no_compression=['ss_sold_date_sk', 'ss_store_sk', 'ss_sales_price'],
                           csv_file_location=csv_path.format('store_sales_sampled'),
                           table_size=2879987999, primary_key=['ss_item_sk', 'ss_ticket_number'],
                           sample_rate=10000000 / 2879987999
                           ))

    return schema

# suboptimal configuration
# def gen_10gb_tpc_ds_schema(csv_path):
#     """
#     TPCDS 10g schema
#     """
#
#     schema = SchemaGraph()
#     schema.add_table(Table('store_sales',
#                            attributes=['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk',
#                                        'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk',
#                                        'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
#                                        'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price',
#                                        'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax',
#                                        'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
#                            csv_file_location=csv_path.format('store_sales_sampled'),
#                            table_size=28800991, primary_key=['ss_item_sk', 'ss_ticket_number'], sample_rate=0.33
#                            ))
#
#     return schema
