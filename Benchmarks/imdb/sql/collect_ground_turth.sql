CREATE TEMPORARY VIEW temp_job_light_results AS
SELECT true_card FROM (

SELECT 0 AS qid, COUNT(*)::bigint AS true_card FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2

UNION ALL
SELECT 1, COUNT(*)::bigint FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2010

UNION ALL
SELECT 2, COUNT(*)::bigint FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2 AND t.production_year>2010

UNION ALL
SELECT 3, COUNT(*)::bigint FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2000

UNION ALL
SELECT 4, COUNT(*)::bigint FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=117

UNION ALL
SELECT 5, COUNT(*)::bigint FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2005

UNION ALL
SELECT 6, COUNT(*)::bigint FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2010

UNION ALL
SELECT 7, COUNT(*)::bigint FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>1990

UNION ALL
SELECT 8, COUNT(*)::bigint FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101

UNION ALL
SELECT 9, COUNT(*)::bigint FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2010 AND mi_idx.info_type_id=101

UNION ALL
SELECT 10, COUNT(*)::bigint FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>1990 AND mi_idx.info_type_id=101

UNION ALL
SELECT 11, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND mc.company_type_id=2

UNION ALL
SELECT 12, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2010 AND mc.company_type_id=2

UNION ALL
SELECT 13, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>1990 AND mc.company_type_id=2

UNION ALL
SELECT 14, COUNT(*)::bigint FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mk.keyword_id=8200

UNION ALL
SELECT 15, COUNT(*)::bigint FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2014

UNION ALL
SELECT 16, COUNT(*)::bigint FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2014 AND mk.keyword_id=8200

UNION ALL
SELECT 17, COUNT(*)::bigint FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mk.keyword_id=8200

UNION ALL
SELECT 18, COUNT(*)::bigint FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000

UNION ALL
SELECT 19, COUNT(*)::bigint FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1995

UNION ALL
SELECT 20, COUNT(*)::bigint FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1984

UNION ALL
SELECT 21, COUNT(*)::bigint FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<2010

UNION ALL
SELECT 22, COUNT(*)::bigint FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2

UNION ALL
SELECT 23, COUNT(*)::bigint FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=4

UNION ALL
SELECT 24, COUNT(*)::bigint FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=7

UNION ALL
SELECT 25, COUNT(*)::bigint FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2015 AND ci.role_id=2

UNION ALL
SELECT 26, COUNT(*)::bigint FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.production_year>2007 AND t.production_year<2010 AND ci.role_id=2

UNION ALL
SELECT 27, COUNT(*)::bigint FROM title t,cast_info ci,movie_companies mc WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2005 AND ci.role_id=1

UNION ALL
SELECT 28, COUNT(*)::bigint FROM title t,cast_info ci,movie_companies mc WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND ci.role_id=1

UNION ALL
SELECT 29, COUNT(*)::bigint FROM title t,cast_info ci,movie_companies mc WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>1990

UNION ALL
SELECT 30, COUNT(*)::bigint FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000

UNION ALL
SELECT 31, COUNT(*)::bigint FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2

UNION ALL
SELECT 32, COUNT(*)::bigint FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>1950

UNION ALL
SELECT 33, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3 AND t.production_year>2005 AND t.production_year<2008 AND mc.company_type_id=2

UNION ALL
SELECT 34, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=113 AND mi.info_type_id=105

UNION ALL
SELECT 35, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3 AND t.production_year>2000 AND t.production_year<2010 AND mc.company_type_id=2

UNION ALL
SELECT 36, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mc.company_type_id=2 AND mi_idx.info_type_id=101 AND mi.info_type_id=16

UNION ALL
SELECT 37, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

UNION ALL
SELECT 38, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

UNION ALL
SELECT 39, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

UNION ALL
SELECT 40, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>2000

UNION ALL
SELECT 41, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2010

UNION ALL
SELECT 42, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>1990

UNION ALL
SELECT 43, COUNT(*)::bigint FROM cast_info ci,title t,movie_keyword mk,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=117

UNION ALL
SELECT 44, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=105 AND mi_idx.info_type_id=100

UNION ALL
SELECT 45, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=101 AND t.production_year>2008 AND t.production_year<2014

UNION ALL
SELECT 46, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100

UNION ALL
SELECT 47, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2009

UNION ALL
SELECT 48, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16

UNION ALL
SELECT 49, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000

UNION ALL
SELECT 50, COUNT(*)::bigint FROM title t,cast_info ci,movie_keyword mk WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>1950 AND t.kind_id=1

UNION ALL
SELECT 51, COUNT(*)::bigint FROM title t,cast_info ci,movie_keyword mk WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND t.kind_id=1

UNION ALL
SELECT 52, COUNT(*)::bigint FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000

UNION ALL
SELECT 53, COUNT(*)::bigint FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010

UNION ALL
SELECT 54, COUNT(*)::bigint FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2010

UNION ALL
SELECT 55, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2008 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

UNION ALL
SELECT 56, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2009 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

UNION ALL
SELECT 57, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010

UNION ALL
SELECT 58, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010 AND mc.company_id=22956

UNION ALL
SELECT 59, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000

UNION ALL
SELECT 60, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100

UNION ALL
SELECT 61, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2010

UNION ALL
SELECT 62, COUNT(*)::bigint FROM title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND t.kind_id=1 AND mi_idx.info_type_id=101

UNION ALL
SELECT 63, COUNT(*)::bigint FROM title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND t.kind_id=1 AND mi_idx.info_type_id=101

UNION ALL
SELECT 64, COUNT(*)::bigint FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year=1998

UNION ALL
SELECT 65, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

UNION ALL
SELECT 66, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

UNION ALL
SELECT 67, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2010 AND mk.keyword_id=7084

UNION ALL
SELECT 68, COUNT(*)::bigint FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2005 AND mk.keyword_id=7084

UNION ALL
SELECT 69, COUNT(*)::bigint FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2000

) q
ORDER BY qid;

\copy (SELECT * FROM temp_job_light_results) TO '/Users/jw/Desktop/FactorJoin/IMDB-JOB/light/true_cards_joblightranges.csv' WITH CSV HEADER;
DROP VIEW temp_job_light_results;
