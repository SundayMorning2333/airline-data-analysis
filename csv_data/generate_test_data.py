"""
生成约500万条测试数据，基于原始 air.csv 的结构和分布。

脏数据设计（约15%会被默认清洗逻辑清理掉）：
  - 重复 MEMBER_NO（~7.5%）：去重步骤会移除
  - 含缺失值的行（~7.5%）：drop 策略会移除
  - 部分数值列存在 IQR 异常值（被 clip 策略裁剪，不移除行）

默认清洗参数: missing_strategy='drop', outlier_method='iqr', outlier_strategy='clip'
"""

import numpy as np
import pandas as pd
import time
import os

# ============================================================
# 配置
# ============================================================
TOTAL_ROWS = 5_000_000
DUPLICATE_RATIO = 0.075        # ~7.5% 重复 MEMBER_NO
MISSING_RATIO = 0.075          # ~7.5% 含缺失值的行
RANDOM_SEED = 42
OUTPUT_FILE = 'air_test.csv'

# 原始数据的一些统计特征（从 air.csv 观察到）
MEMBER_NO_MAX = 62_999         # 原始 MEMBER_NO 范围约 0~62999
MEMBER_NO_OFFSET = 100_000     # 测试数据从 100000 开始编号，避免与原始冲突

# 性别选项及权重
GENDERS = ['男', '女']
GENDER_WEIGHTS = [0.57, 0.33]  # 剩余 10% 会被设为缺失

# FFP_TIER 分布
TIERS = [4, 5, 6]
TIER_WEIGHTS = [0.15, 0.25, 0.60]

# 工作城市/省份/国家 样本
CITIES_CN = [
    '北京', '上海', '广州', '深圳', '成都', '杭州', '武汉', '南京',
    '重庆', '西安', '苏州', '天津', '长沙', '郑州', '青岛', '大连',
    '昆明', '厦门', '乌鲁木齐', '哈尔滨', '济南', '福州', '合肥',
    '贵阳', '兰州', '南昌', '太原', '石家庄', '呼和浩特', '拉萨',
]
PROVINCES_CN = [
    '北京', '上海', '广东', '四川', '浙江', '湖北', '江苏',
    '重庆', '陕西', '山东', '辽宁', '云南', '福建', '河南',
    '湖南', '安徽', '贵州', '甘肃', '江西', '河北', '山西',
    '内蒙古', '西藏', '黑龙江', '吉林', '新疆', '广西', '海南',
]
CITIES_INTL = [
    ('Los Angeles', 'CA', 'US'), ('San Francisco', 'CA', 'US'),
    ('New York', 'NY', 'US'), ('PARIS', 'PARIS', 'FR'),
    ('London', 'London', 'UK'), ('Tokyo', 'Tokyo', 'JP'),
    ('Seoul', 'Seoul', 'KR'), ('Singapore', 'Singapore', 'SG'),
    ('Sydney', 'NSW', 'AU'), ('Toronto', 'ON', 'CA'),
    ('DRANCY', 'ILE-DE-FRANCE', 'FR'), ('Berlin', 'Berlin', 'DE'),
]


def generate_dates(n, start_year=2004, end_year=2014):
    """生成随机日期字符串 (YYYY/M/D 格式)。"""
    start = pd.Timestamp(f'{start_year}-01-01')
    end = pd.Timestamp(f'{end_year}-12-31')
    days_range = (end - start).days
    random_days = np.random.randint(0, days_range, size=n)
    dates = start + pd.to_timedelta(random_days, unit='D')
    return dates.strftime('%Y/%-m/%-d')


def generate_test_data():
    """生成测试数据。"""
    print(f"开始生成 {TOTAL_ROWS:,} 条测试数据...")
    t0 = time.time()
    np.random.seed(RANDOM_SEED)

    # ----------------------------------------------------------
    # 1. 计算各类脏数据数量
    # ----------------------------------------------------------
    n_duplicate = int(TOTAL_ROWS * DUPLICATE_RATIO)   # ~375,000
    n_missing = int(TOTAL_ROWS * MISSING_RATIO)        # ~375,000
    n_clean = TOTAL_ROWS - n_missing                    # 干净行数（含部分后续会变成重复的）
    n_unique_member = TOTAL_ROWS - n_duplicate          # 唯一 MEMBER_NO 数

    print(f"  目标: 唯一MEMBER_NO {n_unique_member:,}, 重复行 {n_duplicate:,}, 缺失值行 {n_missing:,}")

    # ----------------------------------------------------------
    # 2. 生成唯一 MEMBER_NO
    # ----------------------------------------------------------
    member_nos = np.arange(MEMBER_NO_OFFSET, MEMBER_NO_OFFSET + n_unique_member)
    # 随机打乱顺序
    np.random.shuffle(member_nos)

    # ----------------------------------------------------------
    # 3. 生成属性数据（向量化）
    # ----------------------------------------------------------
    print("  生成属性数据...")

    # 性别：85% 有值，15% 为空（其中一部分在 missing 行，一部分随机）
    gender_pool = np.random.choice(GENDERS + [''], size=TOTAL_ROWS,
                                   p=[0.52, 0.30, 0.18])

    # FFP_TIER
    tiers = np.random.choice(TIERS, size=TOTAL_ROWS, p=TIER_WEIGHTS)

    # 工作地点（80% 国内，15% 国际，5% 空）- 向量化生成
    intl_cities = [c[0] for c in CITIES_INTL]
    intl_provinces = [c[1] for c in CITIES_INTL]
    intl_countries = [c[2] for c in CITIES_INTL]

    rand_loc = np.random.random(TOTAL_ROWS)
    mask_cn = rand_loc < 0.80
    mask_intl = (rand_loc >= 0.80) & (rand_loc < 0.95)
    # mask_empty = rand_loc >= 0.95  (implicit)

    cn_indices = np.random.randint(0, min(len(CITIES_CN), len(PROVINCES_CN)), TOTAL_ROWS)
    intl_indices = np.random.randint(0, len(CITIES_INTL), TOTAL_ROWS)

    city_arr = np.where(mask_cn, np.array(CITIES_CN)[cn_indices],
               np.where(mask_intl, np.array(intl_cities)[intl_indices], ''))
    province_arr = np.where(mask_cn, np.array(PROVINCES_CN)[cn_indices],
                   np.where(mask_intl, np.array(intl_provinces)[intl_indices], ''))
    country_arr = np.where(mask_cn, 'CN',
                   np.where(mask_intl, np.array(intl_countries)[intl_indices], 'CN'))

    # 年龄（正态分布，均值38，标准差12，裁剪到18~80）
    ages = np.clip(np.random.normal(38, 12, TOTAL_ROWS), 18, 80).astype(int)

    # 日期字段
    print("  生成日期字段...")
    ffp_dates = generate_dates(TOTAL_ROWS, 2003, 2013)
    first_flight_dates = generate_dates(TOTAL_ROWS, 2003, 2014)
    load_time = np.array(['2014/3/31'] * TOTAL_ROWS)  # 固定值，与原始数据一致

    # LAST_FLIGHT_DATE：大部分在2013~2014，少量在更早
    last_flight_dates = generate_dates(TOTAL_ROWS, 2012, 2014)

    # ----------------------------------------------------------
    # 4. 生成数值字段（基于合理的分布）
    # ----------------------------------------------------------
    print("  生成数值字段...")

    flight_count = np.random.negative_binomial(5, 0.15, TOTAL_ROWS).astype(float)
    bp_sum = np.clip(np.random.lognormal(10.5, 1.2, TOTAL_ROWS), 0, 800000)
    ep_sum_yr1 = np.where(np.random.random(TOTAL_ROWS) < 0.6, 0,
                          np.random.exponential(15000, TOTAL_ROWS))
    ep_sum_yr2 = np.where(np.random.random(TOTAL_ROWS) < 0.5, 0,
                          np.random.exponential(20000, TOTAL_ROWS))
    sum_yr1 = np.clip(np.random.lognormal(10.2, 1.3, TOTAL_ROWS), 0, 500000)
    sum_yr2 = np.clip(np.random.lognormal(10.3, 1.3, TOTAL_ROWS), 0, 500000)
    seg_km_sum = np.clip(np.random.lognormal(11.5, 1.1, TOTAL_ROWS), 0, 900000)
    weighted_seg_km = seg_km_sum * np.random.uniform(0.7, 1.3, TOTAL_ROWS)

    avg_flight_count = flight_count / np.random.uniform(2, 12, TOTAL_ROWS)
    avg_bp_sum = bp_sum / np.maximum(flight_count, 1)

    begin_to_first = np.random.exponential(10, TOTAL_ROWS).astype(int)
    last_to_end = np.random.exponential(40, TOTAL_ROWS).astype(int)
    avg_interval = np.where(flight_count > 1,
                            np.random.uniform(1, 30, TOTAL_ROWS), 0)
    max_interval = avg_interval + np.random.exponential(15, TOTAL_ROWS)

    add_points_yr1 = np.where(np.random.random(TOTAL_ROWS) < 0.7, 0,
                              np.random.exponential(5000, TOTAL_ROWS))
    add_points_yr2 = np.where(np.random.random(TOTAL_ROWS) < 0.7, 0,
                              np.random.exponential(5000, TOTAL_ROWS))
    exchange_count = np.random.poisson(5, TOTAL_ROWS).astype(float)

    avg_discount = np.clip(np.random.normal(0.85, 0.15, TOTAL_ROWS), 0.3, 1.5)

    p1y_flight = np.random.negative_binomial(3, 0.2, TOTAL_ROWS).astype(float)
    l1y_flight = flight_count - p1y_flight
    l1y_flight = np.maximum(l1y_flight, 0)

    p1y_bp_sum = np.clip(np.random.lognormal(9.5, 1.2, TOTAL_ROWS), 0, 400000)
    l1y_bp_sum = bp_sum - p1y_bp_sum
    l1y_bp_sum = np.maximum(l1y_bp_sum, 0)

    ep_sum = ep_sum_yr1 + ep_sum_yr2
    add_point_sum = add_points_yr1 + add_points_yr2
    eli_add_point = add_point_sum * np.random.uniform(0.8, 1.0, TOTAL_ROWS)
    l1y_eli_add = eli_add_point * np.random.uniform(0.7, 1.0, TOTAL_ROWS)

    points_sum = np.clip(np.random.lognormal(11, 1.3, TOTAL_ROWS), 0, 800000)
    l1y_points_sum = points_sum * np.random.uniform(0.3, 0.7, TOTAL_ROWS)

    ration_l1y = l1y_flight / np.maximum(flight_count, 1)
    ration_p1y = p1y_flight / np.maximum(flight_count, 1)
    ration_p1y_bp = p1y_bp_sum / np.maximum(bp_sum, 1)
    ration_l1y_bp = l1y_bp_sum / np.maximum(bp_sum, 1)

    point_notflight = np.random.poisson(10, TOTAL_ROWS).astype(float)

    # ----------------------------------------------------------
    # 5. 注入 IQR 异常值（约2%的数值点，不会被 remove，会被 clip）
    # ----------------------------------------------------------
    print("  注入 IQR 异常值...")
    outlier_cols_idx = [0, 1, 5, 6]  # flight_count, bp_sum, sum_yr1, sum_yr2
    all_numeric = [
        flight_count, bp_sum, ep_sum_yr1, ep_sum_yr2, sum_yr1, sum_yr2,
        seg_km_sum, weighted_seg_km, avg_flight_count, avg_bp_sum,
        begin_to_first, last_to_end, avg_interval, max_interval,
        add_points_yr1, add_points_yr2, exchange_count, avg_discount,
        p1y_flight, l1y_flight, p1y_bp_sum, l1y_bp_sum,
        ep_sum, add_point_sum, eli_add_point, l1y_eli_add,
        points_sum, l1y_points_sum, ration_l1y, ration_p1y,
        ration_p1y_bp, ration_l1y_bp, point_notflight,
    ]

    n_outlier_inject = int(TOTAL_ROWS * 0.02)
    outlier_indices = np.random.choice(TOTAL_ROWS, n_outlier_inject, replace=False)
    for idx in outlier_cols_idx:
        arr = all_numeric[idx]
        arr[outlier_indices] *= np.random.uniform(5, 20, len(outlier_indices))

    # ----------------------------------------------------------
    # 6. 组装 DataFrame
    # ----------------------------------------------------------
    print("  组装 DataFrame...")

    # 计算各列中需要设为空的比例
    # 目标：约 n_missing 行至少有一个关键字段为空
    # 关键字段：GENDER, AGE, WORK_CITY
    missing_indices = np.random.choice(TOTAL_ROWS, n_missing, replace=False)

    # 复制性别数组并注入缺失
    gender_col = gender_pool.copy()
    gender_col[missing_indices[:n_missing // 3]] = ''

    age_col = ages.astype(float).copy()
    age_col[missing_indices[n_missing // 3: 2 * n_missing // 3]] = np.nan

    city_col = city_arr.copy()
    city_col[missing_indices[2 * n_missing // 3:]] = ''

    # 为了让 drop 策略真正移除行，需要确保这些行有 pandas NaN
    # GENDER 和 WORK_CITY 的 '' 不会被 dropna 视为 NaN
    # 所以我们额外在 SUM_YR_1 列注入 NaN（数值列的 NaN 会被 dropna 识别）
    sum_yr1_col = sum_yr1.copy()
    sum_yr1_col[missing_indices] = np.nan

    # MEMBER_NO 数组：前 n_unique_member 个唯一值 + 从前面随机抽取的重复值
    member_col = np.empty(TOTAL_ROWS, dtype=int)
    member_col[:n_unique_member] = member_nos
    # 重复值：从已有 MEMBER_NO 中随机抽取
    dup_sources = np.random.choice(member_nos, n_duplicate, replace=True)
    member_col[n_unique_member:] = dup_sources
    # 随机打乱全部 MEMBER_NO 的顺序（让重复行散布在各处）
    shuffle_idx = np.random.permutation(TOTAL_ROWS)
    member_col = member_col[shuffle_idx]

    # 同步打乱所有列
    def shuffle_arr(arr):
        return arr[shuffle_idx]

    df = pd.DataFrame({
        'MEMBER_NO': member_col,
        'FFP_DATE': shuffle_arr(ffp_dates),
        'FIRST_FLIGHT_DATE': shuffle_arr(first_flight_dates),
        'GENDER': shuffle_arr(gender_col),
        'FFP_TIER': shuffle_arr(tiers),
        'WORK_CITY': shuffle_arr(city_col),
        'WORK_PROVINCE': shuffle_arr(province_arr),
        'WORK_COUNTRY': shuffle_arr(country_arr),
        'AGE': shuffle_arr(age_col),
        'LOAD_TIME': shuffle_arr(load_time),
        'FLIGHT_COUNT': shuffle_arr(all_numeric[0]),
        'BP_SUM': shuffle_arr(all_numeric[1]),
        'EP_SUM_YR_1': shuffle_arr(all_numeric[2]),
        'EP_SUM_YR_2': shuffle_arr(all_numeric[3]),
        'SUM_YR_1': shuffle_arr(sum_yr1_col),
        'SUM_YR_2': shuffle_arr(all_numeric[5]),
        'SEG_KM_SUM': shuffle_arr(all_numeric[6]),
        'WEIGHTED_SEG_KM': shuffle_arr(all_numeric[7]),
        'LAST_FLIGHT_DATE': shuffle_arr(last_flight_dates),
        'AVG_FLIGHT_COUNT': shuffle_arr(all_numeric[8]),
        'AVG_BP_SUM': shuffle_arr(all_numeric[9]),
        'BEGIN_TO_FIRST': shuffle_arr(all_numeric[10]),
        'LAST_TO_END': shuffle_arr(all_numeric[11]),
        'AVG_INTERVAL': shuffle_arr(all_numeric[12]),
        'MAX_INTERVAL': shuffle_arr(all_numeric[13]),
        'ADD_POINTS_SUM_YR_1': shuffle_arr(all_numeric[14]),
        'ADD_POINTS_SUM_YR_2': shuffle_arr(all_numeric[15]),
        'EXCHANGE_COUNT': shuffle_arr(all_numeric[16]),
        'avg_discount': shuffle_arr(all_numeric[17]),
        'P1Y_Flight_Count': shuffle_arr(all_numeric[18]),
        'L1Y_Flight_Count': shuffle_arr(all_numeric[19]),
        'P1Y_BP_SUM': shuffle_arr(all_numeric[20]),
        'L1Y_BP_SUM': shuffle_arr(all_numeric[21]),
        'EP_SUM': shuffle_arr(all_numeric[22]),
        'ADD_Point_SUM': shuffle_arr(all_numeric[23]),
        'Eli_Add_Point_Sum': shuffle_arr(all_numeric[24]),
        'L1Y_ELi_Add_Points': shuffle_arr(all_numeric[25]),
        'Points_Sum': shuffle_arr(all_numeric[26]),
        'L1Y_Points_Sum': shuffle_arr(all_numeric[27]),
        'Ration_L1Y_Flight_Count': shuffle_arr(all_numeric[28]),
        'Ration_P1Y_Flight_Count': shuffle_arr(all_numeric[29]),
        'Ration_P1Y_BPS': shuffle_arr(all_numeric[30]),
        'Ration_L1Y_BPS': shuffle_arr(all_numeric[31]),
        'Point_NotFlight': shuffle_arr(all_numeric[32]),
    })

    # 数值列取合理精度
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].round(6)
    for col in ['FLIGHT_COUNT', 'FFP_TIER', 'EXCHANGE_COUNT', 'BEGIN_TO_FIRST',
                'LAST_TO_END', 'Point_NotFlight', 'AGE',
                'P1Y_Flight_Count', 'L1Y_Flight_Count']:
        if col in df.columns:
            df[col] = df[col].round(0)

    # ----------------------------------------------------------
    # 7. 写入 CSV
    # ----------------------------------------------------------
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    print(f"  写入 CSV: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # ----------------------------------------------------------
    # 8. 输出统计
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("生成完成！")
    print(f"  总行数:       {len(df):>12,}")
    print(f"  唯一MEMBER_NO: {df['MEMBER_NO'].nunique():>12,}")
    print(f"  重复行数:     {df.duplicated(subset=['MEMBER_NO']).sum():>12,}")
    print(f"  SUM_YR_1 NaN: {df['SUM_YR_1'].isna().sum():>12,}")
    print(f"  AGE NaN:      {df['AGE'].isna().sum():>12,}")
    print(f"  文件大小:     {file_size_mb:>12.1f} MB")
    print(f"  耗时:         {elapsed:>12.1f} 秒")

    # 模拟默认清洗
    print("\n--- 模拟默认清洗 (drop + iqr + clip) ---")
    sim = df.copy()
    before = len(sim)

    # 去重
    sim = sim.drop_duplicates(subset=['MEMBER_NO'], keep='first')
    after_dedup = len(sim)
    removed_dedup = before - after_dedup

    # drop 缺失值
    sim = sim.dropna()
    after_drop = len(sim)
    removed_missing = after_dedup - after_drop

    total_removed = before - after_drop
    print(f"  去重移除:     {removed_dedup:>12,} ({removed_dedup/before*100:.2f}%)")
    print(f"  缺失值移除:   {removed_missing:>12,} ({removed_missing/before*100:.2f}%)")
    print(f"  合计移除:     {total_removed:>12,} ({total_removed/before*100:.2f}%)")
    print(f"  最终保留:     {len(sim):>12,}")
    print("=" * 60)

    return df


if __name__ == '__main__':
    generate_test_data()
