CREATE DATABASE IF NOT EXISTS airline_analysis
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE airline_analysis;

-- 创建 RFM 分析结果表
CREATE TABLE IF NOT EXISTS rfm_analysis (
    rfm_id          BIGINT          AUTO_INCREMENT  PRIMARY KEY     COMMENT 'RFM记录唯一标识，自增主键',
    member_no       VARCHAR(32)     NOT NULL                        COMMENT '会员编号（航空公司原始会员号）',
    batch_no        VARCHAR(32)     NOT NULL                        COMMENT '分析批次编号（如 20260701_001）',
    r_value         DECIMAL(12,2)                                   COMMENT 'R值 - 最近一次消费距今天数（天）',
    f_value         DECIMAL(12,2)                                   COMMENT 'F值 - 消费频率（次）',
    m_value         DECIMAL(12,2)                                   COMMENT 'M值 - 消费总金额（元）',
    analysis_date   DATE            NOT NULL                        COMMENT '分析执行日期',
    created_at      DATETIME        DEFAULT CURRENT_TIMESTAMP       COMMENT '记录创建时间',

    UNIQUE KEY  uk_member_batch    (member_no, batch_no),
    INDEX       idx_batch_no       (batch_no),
    INDEX       idx_analysis_date  (analysis_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='RFM分析结果表';

-- 创建客户分群表
CREATE TABLE IF NOT EXISTS customer_clusters (
    cluster_id          BIGINT          AUTO_INCREMENT  PRIMARY KEY     COMMENT '分群记录唯一标识，自增主键',
    member_no           VARCHAR(32)     NOT NULL                        COMMENT '会员编号（航空公司原始会员号）',
    batch_no            VARCHAR(32)     NOT NULL                        COMMENT '聚类批次编号（如 CLUSTER_20260701_120000）',
    cluster_label       INT             NOT NULL                        COMMENT '聚类标签编号（如 0, 1, 2, 3...）',
    customer_type       VARCHAR(32)                                     COMMENT '客户类型：高价值客户/一般客户/低价值客户/...',
    cluster_date        DATE            NOT NULL                        COMMENT '聚类执行日期',
    created_at          DATETIME        DEFAULT CURRENT_TIMESTAMP       COMMENT '记录创建时间',

    UNIQUE KEY  uk_member_cluster_batch  (member_no, batch_no),
    INDEX       idx_batch_no             (batch_no),
    INDEX       idx_cluster_label        (cluster_label),
    INDEX       idx_customer_type        (customer_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='客户分群表';

-- 创建原始客户数据表（清洗后）
CREATE TABLE IF NOT EXISTS member_data (
    data_id             BIGINT          AUTO_INCREMENT  PRIMARY KEY     COMMENT '数据记录唯一标识，自增主键',
    member_no           VARCHAR(32)     NOT NULL                        COMMENT '会员编号（航空公司原始会员号）',
    batch_no            VARCHAR(32)     NOT NULL                        COMMENT '数据批次编号（如 20260701_001）',

    -- 基础信息字段
    ffp_date            VARCHAR(32)                                     COMMENT '常旅客计划入会日期',
    first_flight_date   VARCHAR(32)                                     COMMENT '第一次乘机日期',
    gender              VARCHAR(8)                                      COMMENT '性别',
    ffp_tier            INT                                             COMMENT '常旅客计划等级',
    work_city           VARCHAR(64)                                     COMMENT '工作城市',
    work_province       VARCHAR(64)                                     COMMENT '工作省份',
    work_country        VARCHAR(32)                                     COMMENT '工作国家',
    age                 INT                                             COMMENT '年龄',
    load_time           VARCHAR(32)                                     COMMENT '数据加载时间',

    -- 乘机行为字段
    flight_count        INT                                             COMMENT '总乘机次数',
    seg_km_sum          DECIMAL(14,2)                                   COMMENT '总飞行公里数',
    weighted_seg_km     DECIMAL(14,2)                                   COMMENT '加权飞行公里数',
    avg_flight_count    DECIMAL(10,2)                                   COMMENT '平均乘机次数（年均）',
    last_flight_date    VARCHAR(32)                                     COMMENT '最后一次乘机日期',
    begin_to_first      DECIMAL(10,2)                                   COMMENT '入会到首次乘机间隔（天数）',
    last_to_end         DECIMAL(10,2)                                   COMMENT '最后乘机距数据截止日间隔（天数）',
    avg_interval        DECIMAL(10,2)                                   COMMENT '平均乘机间隔（天数）',
    max_interval        INT                                             COMMENT '最大乘机间隔（天数）',
    p1y_flight_count    INT                                             COMMENT '前一年乘机次数',
    l1y_flight_count    INT                                             COMMENT '最近一年乘机次数',
    ration_l1y_flight_count DECIMAL(10,6)                               COMMENT '最近一年乘机次数占比',
    ration_p1y_flight_count DECIMAL(10,6)                               COMMENT '前一年乘机次数占比',

    -- 积分与消费字段
    bp_sum              DECIMAL(14,2)                                   COMMENT '基本积分总和（飞行里程积分）',
    avg_bp_sum          DECIMAL(14,2)                                   COMMENT '平均基本积分总和（年均）',
    ep_sum              DECIMAL(14,2)                                   COMMENT '精英积分总和',
    ep_sum_yr_1         DECIMAL(14,2)                                   COMMENT '第一年精英积分总和',
    ep_sum_yr_2         DECIMAL(14,2)                                   COMMENT '第二年精英积分总和',
    add_point_sum       DECIMAL(14,2)                                   COMMENT '额外积分总和',
    eli_add_point_sum   DECIMAL(14,2)                                   COMMENT '有效额外积分总和',
    l1y_eli_add_points  DECIMAL(14,2)                                   COMMENT '最近一年有效额外积分',
    add_points_sum_yr_1 DECIMAL(14,2)                                   COMMENT '第一年额外积分总和',
    add_points_sum_yr_2 DECIMAL(14,2)                                   COMMENT '第二年额外积分总和',
    points_sum          DECIMAL(14,2)                                   COMMENT '总积分',
    l1y_points_sum      DECIMAL(14,2)                                   COMMENT '最近一年总积分',
    l1y_bp_sum          DECIMAL(14,2)                                   COMMENT '最近一年基本积分总和',
    p1y_bp_sum          DECIMAL(14,2)                                   COMMENT '前一年基本积分总和',
    point_not_flight    DECIMAL(14,2)                                   COMMENT '非乘机积分',
    ration_p1y_bps      DECIMAL(10,6)                                   COMMENT '前一年基本积分占比',
    ration_l1y_bps      DECIMAL(10,6)                                   COMMENT '最近一年基本积分占比',

    -- 消费与折扣字段
    sum_yr_1            DECIMAL(14,2)                                   COMMENT '第一年票价总额',
    sum_yr_2            DECIMAL(14,2)                                   COMMENT '第二年票价总额',
    exchange_count      INT                                             COMMENT '积分兑换次数',
    avg_discount        DECIMAL(10,4)                                   COMMENT '平均折扣率',

    import_date         DATE            NOT NULL                        COMMENT '数据导入日期',
    created_at          DATETIME        DEFAULT CURRENT_TIMESTAMP       COMMENT '记录创建时间',

    UNIQUE KEY  uk_member_data_batch  (member_no, batch_no),
    INDEX       idx_batch_no          (batch_no),
    INDEX       idx_import_date       (import_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='原始客户数据表（清洗后）';
