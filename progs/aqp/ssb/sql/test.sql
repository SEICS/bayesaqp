select AVG(lo_extendedprice) from lineorder, dwdate where lo_orderdate = d_datekey and d_year = 1993 and lo_discount>=1 and lo_discount<=3 and lo_quantity < 25;
select AVG(lo_extendedprice) from lineorder, dwdate where lo_orderdate = d_datekey and d_yearmonthnum = 199401 and lo_discount>=4 and lo_discount<=6 and  lo_quantity>=26 and lo_quantity<=35;
select AVG(lo_extendedprice) from lineorder, dwdate where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year = 1994 and lo_discount>=5 and lo_discount<=7 and lo_quantity>=26 and lo_quantity<=35;
