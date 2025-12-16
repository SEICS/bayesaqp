-- All queries are modified based on ssb Q3 (indexed from 0, referring to S2.1 in the paper)
-- 3 joins + no GB
\copy (select count(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA') To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/join/postgres/psql_q0.csv' With CSV DELIMITER ',' HEADER

\copy (select avg(lo_revenue) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA') To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/join/postgres/psql_q1.csv' With CSV DELIMITER ',' HEADER

\copy (select sum(lo_revenue) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA') To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/join/postgres/psql_q2.csv' With CSV DELIMITER ',' HEADER

-- 3 joins + 2 GB
\copy (select d_year, p_brand1, count(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA' group by d_year, p_brand1) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/join/postgres/psql_q3.csv' With CSV DELIMITER ',' HEADER

\copy (select d_year, p_brand1, avg(lo_revenue) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA' group by d_year, p_brand1) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/join/postgres/psql_q4.csv' With CSV DELIMITER ',' HEADER

\copy (select d_year, p_brand1, sum(lo_revenue) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA' group by d_year, p_brand1) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/join/postgres/psql_q5.csv' With CSV DELIMITER ',' HEADER