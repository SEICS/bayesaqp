-- All queries are modified based on ssb Q3 (indexed from 0, referring to S2.1 in the paper)
-- 3 joins + no GB
\copy (SELECT COUNT(*) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/no_join/postgres/psql_q0.csv' With CSV DELIMITER ',' HEADER

\copy (SELECT AVG(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/no_join/postgres/psql_q1.csv' With CSV DELIMITER ',' HEADER

\copy (SELECT SUM(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/no_join/postgres/psql_q2.csv' With CSV DELIMITER ',' HEADER

-- 3 joins + 2 GB
\copy (SELECT lo_suppkey, lo_custkey, lo_orderdate, COUNT(*) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35 GROUP BY lo_suppkey, lo_custkey, lo_orderdate) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/no_join/postgres/psql_q3.csv' With CSV DELIMITER ',' HEADER

\copy (SELECT lo_suppkey, lo_custkey, lo_orderdate, AVG(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35 GROUP BY lo_suppkey, lo_custkey, lo_orderdate) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/no_join/postgres/psql_q4.csv' With CSV DELIMITER ',' HEADER

\copy (SELECT lo_suppkey, lo_custkey, lo_orderdate, SUM(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35 GROUP BY lo_suppkey, lo_custkey, lo_orderdate) To '~/Desktop/bayestensor/progs/plots/ssb/opt_phases/sf50/no_join/postgres/psql_q5.csv' With CSV DELIMITER ',' HEADER