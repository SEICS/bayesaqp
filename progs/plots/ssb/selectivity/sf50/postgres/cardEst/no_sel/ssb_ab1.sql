\copy (select COUNT(*) from lineorder, dwdate where lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q0.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from lineorder, dwdate where lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q1.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from lineorder, dwdate where lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q2.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q3.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q4.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from lineorder, dwdate, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q5.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from customer, lineorder, supplier, dwdate where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q6.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from customer, lineorder, supplier, dwdate where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q7.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from customer, lineorder, supplier, dwdate where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q8.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from customer, lineorder, supplier, dwdate where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q9.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from dwdate, customer, supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q10.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from dwdate, customer, supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q11.csv' With CSV DELIMITER ',' HEADER

\copy (select COUNT(*) from dwdate, customer, supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey) To '~/Desktop/bayestensor/progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel/psql_q12.csv' With CSV DELIMITER ',' HEADER