create table part_p_name_1000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

create table part_p_name_2000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

create table part_p_name_3000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

create table part_p_name_4000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

create table part_p_name_5000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

create table part_p_name_6000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

create table part_p_name_7000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

create table part_p_name_8000 (
  p_partkey       integer     not null,
  p_name          varchar(22) not null,
  p_mfgr          varchar(6)  not null,
  p_category      varchar(7)  not null,
  p_brand1        varchar(9)  not null,
  p_color         varchar(11) not null,
  p_type          varchar(25) not null,
  p_size          integer     not null,
  p_container     varchar(10) not null,
  primary key (p_partkey)
);

\copy part_p_name_1000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_1000.csv' DELIMITER '|' CSV HEADER;

\copy part_p_name_2000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_2000.csv' DELIMITER '|' CSV HEADER;

\copy part_p_name_3000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_3000.csv' DELIMITER '|' CSV HEADER;

\copy part_p_name_4000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_4000.csv' DELIMITER '|' CSV HEADER;

\copy part_p_name_5000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_5000.csv' DELIMITER '|' CSV HEADER;

\copy part_p_name_6000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_6000.csv' DELIMITER '|' CSV HEADER;

\copy part_p_name_7000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_7000.csv' DELIMITER '|' CSV HEADER;

\copy part_p_name_8000(p_partkey,p_name,p_mfgr,p_category,p_brand1,p_color,p_type,p_size,p_container) FROM '~/Desktop/data/ssb/sf50/part_p_name_8000.csv' DELIMITER '|' CSV HEADER;

-- CREATE INDEX part_idx_1k ON part_p_name_1000(p_partkey);
-- CREATE INDEX part_idx_2k ON part_p_name_2000(p_partkey);
-- CREATE INDEX part_idx_3k ON part_p_name_3000(p_partkey);
-- CREATE INDEX part_idx_4k ON part_p_name_4000(p_partkey);
-- CREATE INDEX part_idx_5k ON part_p_name_5000(p_partkey);
-- CREATE INDEX part_idx_6k ON part_p_name_6000(p_partkey);
-- CREATE INDEX part_idx_7k ON part_p_name_7000(p_partkey);
-- CREATE INDEX part_idx_8k ON part_p_name_8000(p_partkey);