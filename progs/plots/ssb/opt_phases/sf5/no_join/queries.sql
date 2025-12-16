SELECT COUNT(*) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35;
SELECT AVG(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35;
SELECT SUM(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35;
SELECT lo_suppkey, lo_custkey, lo_orderdate, COUNT(*) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35 GROUP BY lo_suppkey, lo_custkey, lo_orderdate;
SELECT lo_suppkey, lo_custkey, lo_orderdate, AVG(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35 GROUP BY lo_suppkey, lo_custkey, lo_orderdate;
SELECT lo_suppkey, lo_custkey, lo_orderdate, SUM(lo_revenue) FROM lineorder WHERE lo_quantity>=26 AND lo_quantity <= 35 GROUP BY lo_suppkey, lo_custkey, lo_orderdate;