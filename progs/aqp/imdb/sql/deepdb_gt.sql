-- true_card.sql
-- Output columns: query_no, query, cardinality_true

DROP VIEW IF EXISTS temp_job_light_results;

CREATE TEMPORARY VIEW temp_job_light_results AS
SELECT
  qid AS query_no,
  query_text AS query,
  true_card AS cardinality_true
FROM (
  SELECT
    0 AS qid,
    'SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2;' AS query_text,
    COUNT(*)::bigint AS true_card
  FROM movie_companies mc, title t, movie_info_idx mi_idx
  WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2

  UNION ALL
  SELECT
    1,
    'SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2010;',
    COUNT(*)::bigint
  FROM movie_companies mc, title t, movie_info_idx mi_idx
  WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2005 AND t.production_year<2010

  UNION ALL
  SELECT
    2,
    'SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2 AND t.production_year>2010;',
    COUNT(*)::bigint
  FROM movie_companies mc, title t, movie_info_idx mi_idx
  WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2 AND t.production_year>2010

  UNION ALL
  SELECT
    3,
    'SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2000;',
    COUNT(*)::bigint
  FROM movie_companies mc, title t, movie_info_idx mi_idx
  WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=113 AND mc.company_type_id=2 AND t.production_year>2000

  UNION ALL
  SELECT
    4,
    'SELECT COUNT(*) FROM movie_companies mc,title t,movie_keyword mk WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=117;',
    COUNT(*)::bigint
  FROM movie_companies mc, title t, movie_keyword mk
  WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=117

  UNION ALL
  SELECT
    5,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2005;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2005

  UNION ALL
  SELECT
    6,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2010;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>2010

  UNION ALL
  SELECT
    7,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>1990;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.production_year>1990

  UNION ALL
  SELECT
    8,
    'SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info_idx mi_idx, movie_keyword mk
  WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2005 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    9,
    'SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2010 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info_idx mi_idx, movie_keyword mk
  WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>2010 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    10,
    'SELECT COUNT(*) FROM title t,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>1990 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info_idx mi_idx, movie_keyword mk
  WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.production_year>1990 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    11,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND mc.company_type_id=2;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND mc.company_type_id=2

  UNION ALL
  SELECT
    12,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2010 AND mc.company_type_id=2;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>2010 AND mc.company_type_id=2

  UNION ALL
  SELECT
    13,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>1990 AND mc.company_type_id=2;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.production_year>1990 AND mc.company_type_id=2

  UNION ALL
  SELECT
    14,
    'SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mk.keyword_id=8200;',
    COUNT(*)::bigint
  FROM movie_keyword mk, title t, cast_info ci
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND mk.keyword_id=8200

  UNION ALL
  SELECT
    15,
    'SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2014;',
    COUNT(*)::bigint
  FROM movie_keyword mk, title t, cast_info ci
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2014

  UNION ALL
  SELECT
    16,
    'SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2014 AND mk.keyword_id=8200;',
    COUNT(*)::bigint
  FROM movie_keyword mk, title t, cast_info ci
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2014 AND mk.keyword_id=8200

  UNION ALL
  SELECT
    17,
    'SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mk.keyword_id=8200;',
    COUNT(*)::bigint
  FROM movie_keyword mk, title t, cast_info ci
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND mk.keyword_id=8200

  UNION ALL
  SELECT
    18,
    'SELECT COUNT(*) FROM movie_keyword mk,title t,cast_info ci WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000;',
    COUNT(*)::bigint
  FROM movie_keyword mk, title t, cast_info ci
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000

  UNION ALL
  SELECT
    19,
    'SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1995;',
    COUNT(*)::bigint
  FROM cast_info ci, title t
  WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1995

  UNION ALL
  SELECT
    20,
    'SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1984;',
    COUNT(*)::bigint
  FROM cast_info ci, title t
  WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1984

  UNION ALL
  SELECT
    21,
    'SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<2010;',
    COUNT(*)::bigint
  FROM cast_info ci, title t
  WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<2010

  UNION ALL
  SELECT
    22,
    'SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2;',
    COUNT(*)::bigint
  FROM cast_info ci, title t, movie_companies mc
  WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=2

  UNION ALL
  SELECT
    23,
    'SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=4;',
    COUNT(*)::bigint
  FROM cast_info ci, title t, movie_companies mc
  WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=4

  UNION ALL
  SELECT
    24,
    'SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=7;',
    COUNT(*)::bigint
  FROM cast_info ci, title t, movie_companies mc
  WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.role_id=7

  UNION ALL
  SELECT
    25,
    'SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2015 AND ci.role_id=2;',
    COUNT(*)::bigint
  FROM cast_info ci, title t, movie_companies mc
  WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND t.production_year<2015 AND ci.role_id=2

  UNION ALL
  SELECT
    26,
    'SELECT COUNT(*) FROM cast_info ci,title t,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.production_year>2007 AND t.production_year<2010 AND ci.role_id=2;',
    COUNT(*)::bigint
  FROM cast_info ci, title t, movie_companies mc
  WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.production_year>2007 AND t.production_year<2010 AND ci.role_id=2

  UNION ALL
  SELECT
    27,
    'SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2005 AND ci.role_id=1;',
    COUNT(*)::bigint
  FROM title t, cast_info ci, movie_companies mc
  WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2005 AND ci.role_id=1

  UNION ALL
  SELECT
    28,
    'SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND ci.role_id=1;',
    COUNT(*)::bigint
  FROM title t, cast_info ci, movie_companies mc
  WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>2010 AND ci.role_id=1

  UNION ALL
  SELECT
    29,
    'SELECT COUNT(*) FROM title t,cast_info ci,movie_companies mc WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>1990;',
    COUNT(*)::bigint
  FROM title t, cast_info ci, movie_companies mc
  WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.production_year>1990

  UNION ALL
  SELECT
    30,
    'SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000;',
    COUNT(*)::bigint
  FROM title t, movie_keyword mk, movie_companies mc
  WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000

  UNION ALL
  SELECT
    31,
    'SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2;',
    COUNT(*)::bigint
  FROM title t, movie_keyword mk, movie_companies mc
  WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2

  UNION ALL
  SELECT
    32,
    'SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>1950;',
    COUNT(*)::bigint
  FROM title t, movie_keyword mk, movie_companies mc
  WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.production_year>1950

  UNION ALL
  SELECT
    33,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3 AND t.production_year>2005 AND t.production_year<2008 AND mc.company_type_id=2;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3 AND t.production_year>2005 AND t.production_year<2008 AND mc.company_type_id=2

  UNION ALL
  SELECT
    34,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=113 AND mi.info_type_id=105;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=113 AND mi.info_type_id=105

  UNION ALL
  SELECT
    35,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3 AND t.production_year>2000 AND t.production_year<2010 AND mc.company_type_id=2;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3 AND t.production_year>2000 AND t.production_year<2010 AND mc.company_type_id=2

  UNION ALL
  SELECT
    36,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,movie_info_idx mi_idx WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mc.company_type_id=2 AND mi_idx.info_type_id=101 AND mi.info_type_id=16;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, movie_info_idx mi_idx
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mc.company_type_id=2 AND mi_idx.info_type_id=101 AND mi.info_type_id=16

  UNION ALL
  SELECT
    37,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2010 AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    38,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    39,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    40,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>2000;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>2000

  UNION ALL
  SELECT
    41,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2010;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2010

  UNION ALL
  SELECT
    42,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>1990;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.production_year>1990

  UNION ALL
  SELECT
    43,
    'SELECT COUNT(*) FROM cast_info ci,title t,movie_keyword mk,movie_companies mc WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=117;',
    COUNT(*)::bigint
  FROM cast_info ci, title t, movie_keyword mk, movie_companies mc
  WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=117

  UNION ALL
  SELECT
    44,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=105 AND mi_idx.info_type_id=100;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, cast_info ci
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=105 AND mi_idx.info_type_id=100

  UNION ALL
  SELECT
    45,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=101 AND t.production_year>2008 AND t.production_year<2014;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, cast_info ci
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=101 AND t.production_year>2008 AND t.production_year<2014

  UNION ALL
  SELECT
    46,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, cast_info ci
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100

  UNION ALL
  SELECT
    47,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2009;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2005 AND t.production_year<2009

  UNION ALL
  SELECT
    48,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16

  UNION ALL
  SELECT
    49,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000

  UNION ALL
  SELECT
    50,
    'SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>1950 AND t.kind_id=1;',
    COUNT(*)::bigint
  FROM title t, cast_info ci, movie_keyword mk
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>1950 AND t.kind_id=1

  UNION ALL
  SELECT
    51,
    'SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND t.kind_id=1;',
    COUNT(*)::bigint
  FROM title t, cast_info ci, movie_keyword mk
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.production_year>2000 AND t.kind_id=1

  UNION ALL
  SELECT
    52,
    'SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000;',
    COUNT(*)::bigint
  FROM title t, movie_keyword mk, movie_companies mc, movie_info mi
  WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2000

  UNION ALL
  SELECT
    53,
    'SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010;',
    COUNT(*)::bigint
  FROM title t, movie_keyword mk, movie_companies mc, movie_info mi
  WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>2000 AND t.production_year<2010

  UNION ALL
  SELECT
    54,
    'SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2010;',
    COUNT(*)::bigint
  FROM title t, movie_keyword mk, movie_companies mc, movie_info mi
  WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year>1950 AND t.production_year<2010

  UNION ALL
  SELECT
    55,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2008 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2008 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    56,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2009 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2009 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    57,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010

  UNION ALL
  SELECT
    58,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010 AND mc.company_id=22956;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2010 AND mc.company_id=22956

  UNION ALL
  SELECT
    59,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000

  UNION ALL
  SELECT
    60,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100

  UNION ALL
  SELECT
    61,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2010;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2010

  UNION ALL
  SELECT
    62,
    'SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND t.kind_id=1 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, cast_info ci, movie_keyword mk, movie_info_idx mi_idx
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2000 AND t.kind_id=1 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    63,
    'SELECT COUNT(*) FROM title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND t.kind_id=1 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, cast_info ci, movie_keyword mk, movie_info_idx mi_idx
  WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.production_year>2005 AND t.kind_id=1 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    64,
    'SELECT COUNT(*) FROM title t,movie_keyword mk,movie_companies mc,movie_info mi WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year=1998;',
    COUNT(*)::bigint
  FROM title t, movie_keyword mk, movie_companies mc, movie_info mi
  WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.id=mi.movie_id AND mk.keyword_id=398 AND mc.company_type_id=2 AND t.production_year=1998

  UNION ALL
  SELECT
    65,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2000 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    66,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,movie_keyword mk,movie_companies mc WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, movie_companies mc
  WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.production_year>2005 AND mi.info_type_id=8 AND mi_idx.info_type_id=101

  UNION ALL
  SELECT
    67,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2010 AND mk.keyword_id=7084;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2010 AND mk.keyword_id=7084

  UNION ALL
  SELECT
    68,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_companies mc,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2005 AND mk.keyword_id=7084;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_companies mc, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND ci.role_id=2 AND mi.info_type_id=16 AND t.production_year>2000 AND t.production_year<2005 AND mk.keyword_id=7084

  UNION ALL
  SELECT
    69,
    'SELECT COUNT(*) FROM title t,movie_info mi,movie_info_idx mi_idx,cast_info ci,movie_keyword mk WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2000;',
    COUNT(*)::bigint
  FROM title t, movie_info mi, movie_info_idx mi_idx, cast_info ci, movie_keyword mk
  WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id=100 AND t.production_year>2000
) q
ORDER BY query_no;

\copy (SELECT query_no, query, cardinality_true FROM temp_job_light_results ORDER BY query_no) TO '~/Desktop/bayestensor/progs/cardest/imdb/sql/deepdb_job_light_true_cardinalities.csv' WITH (FORMAT csv, HEADER true);

DROP VIEW temp_job_light_results;