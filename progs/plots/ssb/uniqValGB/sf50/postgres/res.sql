-- -- \copy (select COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=26 and lo_quantity<=35) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q0.csv' With CSV DELIMITER ',' HEADER

-- -- \copy (select p_mfgr, COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=26 and lo_quantity<=35 group by p_mfgr) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q1.csv' With CSV DELIMITER ',' HEADER

-- -- \copy (select p_category, COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=26 and lo_quantity<=35 group by p_category) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q2.csv' With CSV DELIMITER ',' HEADER

-- \copy (select p_type, COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_type) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q0.csv' With CSV DELIMITER ',' HEADER

-- \copy (select p_brand1, COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_brand1) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q1.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_1000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q0.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_2000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q1.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_3000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q2.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_4000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q3.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_5000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q4.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_6000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q5.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_7000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q6.csv' With CSV DELIMITER ',' HEADER

\copy (select p_name, COUNT(*) from lineorder, dwdate, part_p_name_8000, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and s_region = 'AMERICA' and d_weeknuminyear = 6 and lo_quantity>=5 group by p_name) To '~/Desktop/bayestensor/progs/plots/ssb/uniqValGB/sf50/postgres/psql_q7.csv' With CSV DELIMITER ',' HEADER
