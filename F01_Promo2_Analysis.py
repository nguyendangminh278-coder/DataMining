# Notebook export: 03_F01_Promo2_Analysis
# Original source: https://colab.research.google.com/drive/19K7REYOXk6DfUlXBEf50IWaSfo_a5GK6?hl=vi
# Automatically extracted code cells only. The .ipynb is authoritative.
# Notebook magics/shell commands, if any, are commented out.

# %% [markdown cell 1] ```

# %% [code cell 2]
import pandas as pd

train = pd.read_csv("train_clean.csv", dtype={"StateHoliday": "string"})
test = pd.read_csv("test_clean.csv", dtype={"StateHoliday": "string"})
store = pd.read_csv("store_clean.csv")

print("TRAIN:", train.shape)
print("TEST :", test.shape)
print("STORE:", store.shape)


# %% [code cell 3]
print("TRAIN duplicate key:", train.duplicated(subset=["Store", "Date"]).sum())
print("TEST duplicate key :", test.duplicated(subset=["Store", "Date"]).sum())
print("STORE duplicate key:", store.duplicated(subset=["Store"]).sum())

# %% [markdown cell 4] duplicate = không có dòng trùng theo business key.
# %% [markdown cell 5] check Open và Sales

# %% [code cell 6]
closed_days = train[train["Open"] == 0]

open_days = train[train["Open"] == 1]

open_but_zero_sales = train[
    (train["Open"] == 1) &
    (train["Sales"] == 0)
]

closed_but_has_sales = train[
    (train["Open"] == 0) &
    (train["Sales"] > 0)
]

print("Closed days:", len(closed_days))
print("Open days:", len(open_days))
print("Open but Sales = 0:", len(open_but_zero_sales))
print("Closed but Sales > 0:", len(closed_but_has_sales))



# %% [code cell 7]
train_open = train[train["Open"] == 1].copy()

print("Original train:", train.shape)
print("Filtered train_open:", train_open.shape)


# %% [code cell 8]
open_but_zero_sales.head(10)


# %% [code cell 9]
print("Số dòng Open=1, Sales=0:", len(open_but_zero_sales))
print("Trong đó Customers=0:", (open_but_zero_sales["Customers"] == 0).sum())


# %% [code cell 10]
open_zero_with_customers = open_but_zero_sales[
    open_but_zero_sales["Customers"] > 0
]

open_zero_with_customers

# %% [markdown cell 11] ---
# %% [markdown cell 12] P08 - inconsistent values

# %% [code cell 13]
print("TRAIN")
print(train.dtypes)

print("\nTEST")
print(test.dtypes)

print("\nSTORE")
print(store.dtypes)

# %% [markdown cell 14] Kiểm tra các giá trị thực tế trong các cột dễ inconsistent

# %% [code cell 15]
print("TRAIN - StateHoliday")
print(train["StateHoliday"].value_counts(dropna=False))

print("\nTEST - StateHoliday")
print(test["StateHoliday"].value_counts(dropna=False))

print("\nTRAIN - Open")
print(train["Open"].value_counts(dropna=False))

print("\nTEST - Open")
print(test["Open"].value_counts(dropna=False))

print("\nSTORE - StoreType")
print(store["StoreType"].value_counts(dropna=False))

print("\nSTORE - Assortment")
print(store["Assortment"].value_counts(dropna=False))

# %% [markdown cell 16] Check các cột thời gian trong store

# %% [code cell 17]
print("CompetitionOpenSinceMonth")
print(store["CompetitionOpenSinceMonth"].value_counts(dropna=False).sort_index())

print("\nCompetitionOpenSinceYear")
print(store["CompetitionOpenSinceYear"].value_counts(dropna=False).sort_index())

print("\nPromo2SinceWeek")
print(store["Promo2SinceWeek"].value_counts(dropna=False).sort_index())

print("\nPromo2SinceYear")
print(store["Promo2SinceYear"].value_counts(dropna=False).sort_index())

# %% [markdown cell 18] có 1900.0 với 1961.0 có thể fix lại dữ liệu P05 để clean hơn.
# %% [markdown cell 19] Kiểm tra PromoInterval

# %% [code cell 20]
print(store["PromoInterval"].value_counts(dropna=False))

# %% [markdown cell 21] Kiểm tra Promo2 và PromoInterval

# %% [code cell 22]
print(pd.crosstab(
    store["Promo2"],
    store["PromoInterval"].isna(),
    rownames=["Promo2"],
    colnames=["PromoInterval is NaN"]
))

# %% [markdown cell 23] Checking 3 cột Promo2 còn lại

# %% [code cell 24]
print(pd.crosstab(
    store["Promo2"],
    store["Promo2SinceWeek"].isna(),
    rownames=["Promo2"],
    colnames=["Promo2SinceWeek is NaN"]
))

print()

print(pd.crosstab(
    store["Promo2"],
    store["Promo2SinceYear"].isna(),
    rownames=["Promo2"],
    colnames=["Promo2SinceYear is NaN"]
))

# %% [markdown cell 25] check space, hoa thường column category

# %% [code cell 26]
print("StoreType:", store["StoreType"].dropna().unique())
print("Assortment:", store["Assortment"].dropna().unique())
print("PromoInterval:", store["PromoInterval"].dropna().unique())
print("Train StateHoliday:", train["StateHoliday"].dropna().unique())
print("Test StateHoliday:", test["StateHoliday"].dropna().unique())

# %% [markdown cell 27] ---
# %% [markdown cell 28] P09 Xử lý Inconsistent value

# %% [code cell 29]
train_p09 = train.copy()
test_p09 = test.copy()
store_p09 = store.copy()

print("Đã tạo bản sao để xử lý P09")


# %% [code cell 30]
test_p09["Open"] = test_p09["Open"].astype("int64")

print(test_p09["Open"].dtype)
print(test_p09["Open"].value_counts())


# %% [code cell 31]
cols_to_int = [
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear"
]

for col in cols_to_int:
    store_p09[col] = store_p09[col].astype("Int64")

print(store_p09[cols_to_int].dtypes)


# %% [code cell 32]
print(store_p09[[
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear"
]].isna().sum())


# %% [code cell 33]
print("TRAIN:", train_p09.duplicated(subset=["Store", "Date"]).sum())
print("TEST :", test_p09.duplicated(subset=["Store", "Date"]).sum())
print("STORE:", store_p09.duplicated(subset=["Store"]).sum())


# %% [code cell 34]
print("TRAIN:", train.shape, "->", train_p09.shape)
print("TEST :", test.shape, "->", test_p09.shape)
print("STORE:", store.shape, "->", store_p09.shape)


# %% [code cell 35]
print("TEST")
print(test_p09.dtypes)

print("\nSTORE")
print(store_p09.dtypes)


# %% [code cell 36]
train_p09.to_csv("train_p09.csv", index=False)
test_p09.to_csv("test_p09.csv", index=False)
store_p09.to_csv("store_p09.csv", index=False)

print("Đã xuất 3 file P09")

# %% [markdown cell 37] ---
# %% [markdown cell 38] P10: Xử lý col reduction

# %% [code cell 39]
print("TRAIN columns:")
print(train_p09.columns.tolist())

print("\nTEST columns:")
print(test_p09.columns.tolist())

print("\nChỉ có trong TRAIN:")
print(sorted(set(train_p09.columns) - set(test_p09.columns)))

print("\nChỉ có trong TEST:")
print(sorted(set(test_p09.columns) - set(train_p09.columns)))

# %% [markdown cell 40] các cột có thể dùng chung cho train và test

# %% [code cell 41]
common_cols = [
    col for col in train_p09.columns
    if col in test_p09.columns
]

print("Các cột chung giữa train và test:")
print(common_cols)

# %% [markdown cell 42] Check column Date

# %% [code cell 43]
print(train_p09["Date"].head())
print()
print("Kiểu dữ liệu Date:", train_p09["Date"].dtype)

# %% [markdown cell 44] Check Column Open

# %% [code cell 45]
print("TRAIN Open:")
print(train_p09["Open"].value_counts())

print("\nTEST Open:")
print(test_p09["Open"].value_counts())

# %% [markdown cell 46] Promo và SchoolHoliday

# %% [code cell 47]
print("TRAIN Promo:")
print(train_p09["Promo"].value_counts())

print("\nTRAIN SchoolHoliday:")
print(train_p09["SchoolHoliday"].value_counts())

print("\nTEST Promo:")
print(test_p09["Promo"].value_counts())

print("\nTEST SchoolHoliday:")
print(test_p09["SchoolHoliday"].value_counts())


# %% [code cell 48]
selected_features = [
    "Store",
    "DayOfWeek",
    "Date",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday"
]

target = "Sales"

excluded_from_features = [
    "Customers",  # chỉ có ở train
    "Id"          # chỉ dùng định danh trong test
]

print("Features giữ lại:")
print(selected_features)

print("\nTarget:")
print(target)

print("\nKhông dùng làm feature:")
print(excluded_from_features)

# %% [markdown cell 49] ---
# %% [markdown cell 50] P11: Column Reduction

# %% [code cell 51]
train_p11 = train_p09[
    [
        "Store",
        "DayOfWeek",
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "Sales"
    ]
].copy()

test_p11 = test_p09[
    [
        "Id",
        "Store",
        "DayOfWeek",
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday"
    ]
].copy()

print("TRAIN P11:", train_p11.shape)
print("TEST P11 :", test_p11.shape)

print("\nTRAIN columns:")
print(train_p11.columns.tolist())

print("\nTEST columns:")
print(test_p11.columns.tolist())

# %% [markdown cell 52] check feature train và test có khớp nhau không

# %% [code cell 53]
train_features = set(train_p11.columns) - {"Sales"}
test_features = set(test_p11.columns) - {"Id"}

print("Feature train:", sorted(train_features))
print("Feature test :", sorted(test_features))
print("\nHai tập feature có giống nhau không?")
print(train_features == test_features)

# %% [markdown cell 54] ---
# %% [markdown cell 55] P12 + P13: Xử lý outliers

# %% [code cell 56]
print(train_p11["Sales"].describe())

# %% [markdown cell 57] tính ngưỡng outlier theo IQR (khoảng tứ phân vị trong thống kê.)

# %% [code cell 58]
Q1 = train_p11["Sales"].quantile(0.25)
Q3 = train_p11["Sales"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

sales_outliers = train_p11[
    (train_p11["Sales"] < lower_bound) |
    (train_p11["Sales"] > upper_bound)
]

print("Số dòng bị đánh dấu outlier theo IQR:", len(sales_outliers))


# %% [code cell 59]
print("Outlier theo trạng thái Open:")
print(sales_outliers["Open"].value_counts())

print("\nOutlier theo Promo:")
print(sales_outliers["Promo"].value_counts())

print("\nOutlier theo StateHoliday:")
print(sales_outliers["StateHoliday"].value_counts())

print("\n10 dòng Sales cao nhất:")
display(
    train_p11.sort_values("Sales", ascending=False).head(10)
)

# %% [markdown cell 60] Check outlier có tập trung ở một số Store không

# %% [code cell 61]
print("Top 15 Store có nhiều outlier Sales nhất:")

print(
    sales_outliers["Store"]
    .value_counts()
    .head(15)
)

# %% [markdown cell 62] tính IQR theo từng Store

# %% [code cell 63]
def mark_store_outliers(group):
    q1 = group["Sales"].quantile(0.25)
    q3 = group["Sales"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (group["Sales"] < lower) | (group["Sales"] > upper)

store_outlier_mask = (
    train_p11
    .groupby("Store", group_keys=False)
    .apply(mark_store_outliers)
)

print("Số outlier theo IQR từng Store:", store_outlier_mask.sum())

# %% [markdown cell 64] tính IQR theo từng Store nhưng chỉ trên ngày mở cửa

# %% [code cell 65]
train_open_p12 = train_p11[
    train_p11["Open"] == 1
].copy()

def mark_outliers_series(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (s < lower) | (s > upper)

store_open_outlier_mask = (
    train_open_p12
    .groupby("Store")["Sales"]
    .transform(mark_outliers_series)
)

print("Số ngày mở cửa:", len(train_open_p12))
print("Số outlier theo IQR từng Store, chỉ ngày mở cửa:",
      store_open_outlier_mask.sum())

# %% [markdown cell 66] bảng outlier

# %% [code cell 67]
store_open_outliers = train_open_p12[
    store_open_outlier_mask
].copy()

print("Số outlier:", len(store_open_outliers))

print("\nTheo Promo:")
print(store_open_outliers["Promo"].value_counts())

print("\nTheo StateHoliday:")
print(store_open_outliers["StateHoliday"].value_counts())

print("\n10 dòng Sales cao nhất trong nhóm outlier:")
display(
    store_open_outliers
    .sort_values("Sales", ascending=False)
    .head(10)
)

# %% [markdown cell 68] Xong P12 nốt P13

# %% [code cell 69]
train_p13 = train_p11.copy()

train_p13["SalesOutlier"] = 0

train_p13.loc[
    train_open_p12.index[store_open_outlier_mask],
    "SalesOutlier"
] = 1

print(train_p13["SalesOutlier"].value_counts())


# %% [code cell 70]
print("Outlier có Open = 1:")
print(
    train_p13.loc[
        train_p13["SalesOutlier"] == 1,
        "Open"
    ].value_counts()
)

print("\nSố dòng trước P13:", len(train_p11))
print("Số dòng sau P13 :", len(train_p13))

# %% [markdown cell 71] ---
# %% [markdown cell 72] P14: OUTPUT
# %% [markdown cell 73] xác định cột nào cần normalize

# %% [code cell 74]
print("TRAIN P13")
print(train_p13.dtypes)
print("\nSố giá trị khác nhau:")
print(train_p13.nunique())

print("\nTEST P11")
print(test_p11.dtypes)
print("\nSố giá trị khác nhau:")
print(test_p11.nunique())

print("\nSTORE P09")
print(store_p09.dtypes)
print("\nSố giá trị khác nhau:")
print(store_p09.nunique())

# %% [markdown cell 75] Store, Id → mã định danh
# %% [markdown cell 76] Check CompetitionDistance

# %% [code cell 77]
print(store_p09["CompetitionDistance"].describe())

print("\nMissing CompetitionDistance:",
      store_p09["CompetitionDistance"].isna().sum())

# %% [markdown cell 78] normalize CompetitionDistance

# %% [code cell 79]
import numpy as np

store_p14 = store_p09.copy()

store_p14["CompetitionDistance"] = np.log1p(
    store_p14["CompetitionDistance"]
)

print(store_p14["CompetitionDistance"].describe())


# %% [code cell 80]
train_p14 = train_p13.copy()
test_p14 = test_p11.copy()

print("TRAIN P14:", train_p14.shape)
print("TEST P14 :", test_p14.shape)
print("STORE P14:", store_p14.shape)


# %% [code cell 81]
train_p14.to_csv("train_P14.csv", index=False)
test_p14.to_csv("test_P14.csv", index=False)
store_p14.to_csv("store_P14.csv", index=False)

print("Đã xuất:")
print("- train_P14.csv")
print("- test_P14.csv")
print("- store_P14.csv")


# %% [code cell 82]
train_p14 = train_p13.drop(columns=["SalesOutlier"]).copy()

train_p14.to_csv("train_P14.csv", index=False)
test_p14.to_csv("test_P14.csv", index=False)
store_p14.to_csv("store_P14.csv", index=False)

print("TRAIN P14:", train_p14.shape)
print("TEST P14 :", test_p14.shape)
print("STORE P14:", store_p14.shape)

# %% [markdown cell 83] ---
# %% [markdown cell 84] F01 - Feature Engineering
# %% [markdown cell 85] bảng làm việc bằng cách nối train_P14 với thông tin Promo2 của store_P14

# %% [code cell 86]
f01 = train_p14.merge(
    store_p14[
        [
            "Store",
            "StoreType",
            "Assortment",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval"
        ]
    ],
    on="Store",
    how="left",
    validate="many_to_one"
)

f01["Date"] = pd.to_datetime(f01["Date"])

print("F01 shape:", f01.shape)
print("\nMissing sau merge:")
print(
    f01[
        [
            "StoreType",
            "Assortment",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval"
        ]
    ].isna().sum()
)

# %% [markdown cell 87] xác nhận missing Promo2 có đúng logic không

# %% [code cell 88]
promo2_missing_check = f01.groupby("Promo2").agg(
    Rows=("Store", "size"),
    Missing_Promo2SinceWeek=("Promo2SinceWeek", lambda x: x.isna().sum()),
    Missing_Promo2SinceYear=("Promo2SinceYear", lambda x: x.isna().sum()),
    Missing_PromoInterval=("PromoInterval", lambda x: x.isna().sum())
)

display(promo2_missing_check)

# %% [markdown cell 89] tạo ngày bắt đầu Promo2 và xác định tháng hiện tại có thuộc PromoInterval hay không

# %% [code cell 90]
# Tạo ngày bắt đầu Promo2 từ Promo2SinceYear + Promo2SinceWeek
f01["Promo2StartDate"] = pd.NaT

mask_promo2 = f01["Promo2"] == 1

promo_year = (
    f01.loc[mask_promo2, "Promo2SinceYear"]
    .astype("Int64")
    .astype(str)
)

promo_week = (
    f01.loc[mask_promo2, "Promo2SinceWeek"]
    .astype("Int64")
    .astype(str)
    .str.zfill(2)
)

f01.loc[mask_promo2, "Promo2StartDate"] = pd.to_datetime(
    promo_year + "-W" + promo_week + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)

# Tên tháng phải khớp với PromoInterval
month_map = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sept",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}

f01["PromoMonth"] = f01["Date"].dt.month.map(month_map)

print("Missing Promo2StartDate khi Promo2 = 1:",
      f01.loc[mask_promo2, "Promo2StartDate"].isna().sum())

display(
    f01.loc[mask_promo2, [
        "Date",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "Promo2StartDate",
        "PromoInterval",
        "PromoMonth"
    ]].head(10)
)

# %% [markdown cell 91] tạo IsPromo2ActiveThisMonth

# %% [code cell 92]
# Kiểm tra tháng hiện tại có nằm trong chu kỳ Promo2 không
f01["PromoMonthInInterval"] = (
    (
        f01["PromoInterval"].eq("Jan,Apr,Jul,Oct")
        & f01["PromoMonth"].isin(["Jan", "Apr", "Jul", "Oct"])
    )
    |
    (
        f01["PromoInterval"].eq("Feb,May,Aug,Nov")
        & f01["PromoMonth"].isin(["Feb", "May", "Aug", "Nov"])
    )
    |
    (
        f01["PromoInterval"].eq("Mar,Jun,Sept,Dec")
        & f01["PromoMonth"].isin(["Mar", "Jun", "Sept", "Dec"])
    )
)

# Promo2 active khi đã bắt đầu và đang đúng tháng chạy Promo2
f01["IsPromo2ActiveThisMonth"] = (
    (f01["Promo2"] == 1)
    & (f01["Date"] >= f01["Promo2StartDate"])
    & (f01["PromoMonthInInterval"])
).astype(int)

print("IsPromo2ActiveThisMonth:")
print(f01["IsPromo2ActiveThisMonth"].value_counts())

print("\nKiểm tra theo Promo2:")
print(
    pd.crosstab(
        f01["Promo2"],
        f01["IsPromo2ActiveThisMonth"]
    )
)

# %% [markdown cell 93] xác định các cửa hàng đủ điều kiện để so sánh within-store

# %% [code cell 94]
f01_within = f01[
    (f01["Promo2"] == 1)
    & (f01["Date"] >= f01["Promo2StartDate"])
    & (f01["Open"] == 1)
].copy()

store_active_check = (
    f01_within
    .groupby("Store")["IsPromo2ActiveThisMonth"]
    .nunique()
)

eligible_stores = store_active_check[
    store_active_check == 2
].index

print("Số dòng sau filter:", len(f01_within))
print("Số store Promo2 sau mốc bắt đầu:", f01_within["Store"].nunique())
print("Số store có cả active và inactive:", len(eligible_stores))

# %% [markdown cell 95] so sánh Sales active vs inactive trong chính từng Store

# %% [code cell 96]
f01_compare = (
    f01_within[
        f01_within["Store"].isin(eligible_stores)
    ]
    .groupby(
        ["Store", "IsPromo2ActiveThisMonth"]
    )["Sales"]
    .mean()
    .unstack()
)

f01_compare.columns = [
    "Sales_Inactive",
    "Sales_Active"
]

f01_compare["Difference"] = (
    f01_compare["Sales_Active"]
    - f01_compare["Sales_Inactive"]
)

f01_compare["Pct_Difference"] = (
    f01_compare["Difference"]
    / f01_compare["Sales_Inactive"]
    * 100
)

print("Số store:", len(f01_compare))

print("\nTóm tắt chênh lệch %:")
print(f01_compare["Pct_Difference"].describe())

print("\nSố store Sales active cao hơn:")
print((f01_compare["Difference"] > 0).sum())

print("\nSố store Sales active thấp hơn:")
print((f01_compare["Difference"] < 0).sum())

display(f01_compare.head(10))

# %% [markdown cell 97] F01: chuyển sang mức Store × Month

# %% [code cell 98]
# Tạo biến Year-Month
f01_within["YearMonth"] = f01_within["Date"].dt.to_period("M")

# Tổng hợp theo Store × Month
f01_monthly = (
    f01_within
    .groupby(["Store", "YearMonth"])
    .agg(
        AvgSales=("Sales", "mean"),
        OpenDays=("Sales", "size"),
        IsPromo2Active=("IsPromo2ActiveThisMonth", "max"),
        PromoRate=("Promo", "mean"),
        SchoolHolidayRate=("SchoolHoliday", "mean")
    )
    .reset_index()
)

# Chỉ giữ Store-Month có ít nhất 5 ngày mở cửa
f01_monthly = f01_monthly[
    f01_monthly["OpenDays"] >= 5
].copy()

print("Số Store-Month còn lại:", len(f01_monthly))

print("\nTheo trạng thái Promo2:")
print(f01_monthly["IsPromo2Active"].value_counts())

print("\nSố store có cả active và inactive:")
print(
    (
        f01_monthly
        .groupby("Store")["IsPromo2Active"]
        .nunique()
        == 2
    ).sum()
)

# %% [markdown cell 99] so sánh lại Sales ở cấp tháng

# %% [code cell 100]
monthly_compare = (
    f01_monthly
    .groupby(["Store", "IsPromo2Active"])["AvgSales"]
    .mean()
    .unstack()
)

monthly_compare.columns = [
    "Sales_Inactive_Month",
    "Sales_Active_Month"
]

monthly_compare["Difference"] = (
    monthly_compare["Sales_Active_Month"]
    - monthly_compare["Sales_Inactive_Month"]
)

monthly_compare["Pct_Difference"] = (
    monthly_compare["Difference"]
    / monthly_compare["Sales_Inactive_Month"]
    * 100
)

print("Số store:", len(monthly_compare))

print("\nTóm tắt % chênh lệch:")
print(monthly_compare["Pct_Difference"].describe())

print("\nStore Sales active cao hơn:")
print((monthly_compare["Difference"] > 0).sum())

print("\nStore Sales active thấp hơn:")
print((monthly_compare["Difference"] < 0).sum())

display(monthly_compare.head(10))

# %% [markdown cell 101] 426/571 store (74,6%) có Sales thấp hơn trong tháng Promo2 active.
# %% [markdown cell 102] tạo các biến điều chỉnh ở cấp Store × Month

# %% [code cell 103]
# Đánh dấu ngày có StateHoliday
f01_within["HasStateHoliday"] = (
    f01_within["StateHoliday"] != "0"
).astype(int)

# Tổng hợp lại ở mức Store × Month
f01_monthly_adj = (
    f01_within
    .groupby(["Store", "YearMonth"])
    .agg(
        AvgSales=("Sales", "mean"),
        OpenDays=("Sales", "size"),
        IsPromo2Active=("IsPromo2ActiveThisMonth", "max"),
        PromoRate=("Promo", "mean"),
        SchoolHolidayRate=("SchoolHoliday", "mean"),
        StateHolidayRate=("HasStateHoliday", "mean")
    )
    .reset_index()
)

# Giữ các tháng có ít nhất 5 ngày mở cửa
f01_monthly_adj = f01_monthly_adj[
    f01_monthly_adj["OpenDays"] >= 5
].copy()

# Tạo thêm biến thời gian
f01_monthly_adj["Year"] = f01_monthly_adj["YearMonth"].dt.year
f01_monthly_adj["Month"] = f01_monthly_adj["YearMonth"].dt.month

print(f01_monthly_adj.shape)

display(
    f01_monthly_adj[
        [
            "Store",
            "YearMonth",
            "AvgSales",
            "IsPromo2Active",
            "PromoRate",
            "SchoolHolidayRate",
            "StateHolidayRate",
            "Year",
            "Month"
        ]
    ].head(10)
)


# %% [code cell 104]


# %% [markdown cell 105] chạy mô hình điều chỉnh within-store
# %% [markdown cell 106] C(Store) → so sánh trong cùng store, không để store mạnh/yếu làm nhiễu.

# %% [code cell 107]
import numpy as np
import statsmodels.formula.api as smf

# Log Sales để hệ số dễ diễn giải theo %
f01_monthly_adj["LogAvgSales"] = np.log(
    f01_monthly_adj["AvgSales"]
)

model_f01 = smf.ols(
    formula="""
        LogAvgSales ~
        IsPromo2Active
        + PromoRate
        + SchoolHolidayRate
        + StateHolidayRate
        + C(Store)
        + C(Year)
        + C(Month)
    """,
    data=f01_monthly_adj
).fit(
    cov_type="cluster",
    cov_kwds={"groups": f01_monthly_adj["Store"]}
)

beta = model_f01.params["IsPromo2Active"]
p_value = model_f01.pvalues["IsPromo2Active"]

pct_effect = (np.exp(beta) - 1) * 100

print("Coefficient IsPromo2Active:", beta)
print("Estimated % difference:", pct_effect)
print("p-value:", p_value)

print("\n95% CI:")
print(model_f01.conf_int().loc["IsPromo2Active"])

# %% [markdown cell 108] Promo2 active vẫn gắn với Sales thấp hơn khoảng 0,76%.
# %% [markdown cell 109] robustness check theo từng Year-Month

# %% [code cell 110]
f01_monthly_adj["YearMonthStr"] = (
    f01_monthly_adj["YearMonth"].astype(str)
)

model_f01_robust = smf.ols(
    formula="""
        LogAvgSales ~
        IsPromo2Active
        + PromoRate
        + SchoolHolidayRate
        + StateHolidayRate
        + C(Store)
        + C(YearMonthStr)
    """,
    data=f01_monthly_adj
).fit(
    cov_type="cluster",
    cov_kwds={"groups": f01_monthly_adj["Store"]}
)

beta_r = model_f01_robust.params["IsPromo2Active"]
pct_effect_r = (np.exp(beta_r) - 1) * 100

print("Coefficient:", beta_r)
print("Estimated % difference:", pct_effect_r)
print("p-value:", model_f01_robust.pvalues["IsPromo2Active"])

print("\n95% CI:")
print(
    model_f01_robust
    .conf_int()
    .loc["IsPromo2Active"]
)

# %% [markdown cell 111] Mô hình trước: khoảng -0,76%
# %% [markdown cell 112] kiểm tra hiệu ứng có khác theo StoreType không

# %% [code cell 113]
# Gắn StoreType vào bảng monthly
store_type_map = (
    f01[["Store", "StoreType"]]
    .drop_duplicates()
)

f01_monthly_type = f01_monthly_adj.merge(
    store_type_map,
    on="Store",
    how="left",
    validate="many_to_one"
)

result_by_type = []

for st in sorted(f01_monthly_type["StoreType"].unique()):

    temp = f01_monthly_type[
        f01_monthly_type["StoreType"] == st
    ].copy()

    model = smf.ols(
        formula="""
            LogAvgSales ~
            IsPromo2Active
            + PromoRate
            + SchoolHolidayRate
            + StateHolidayRate
            + C(Store)
            + C(YearMonthStr)
        """,
        data=temp
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": temp["Store"]}
    )

    beta = model.params["IsPromo2Active"]

    result_by_type.append({
        "StoreType": st,
        "Stores": temp["Store"].nunique(),
        "EstimatedPctDifference": (np.exp(beta) - 1) * 100,
        "PValue": model.pvalues["IsPromo2Active"]
    })

result_by_type = pd.DataFrame(result_by_type)

display(result_by_type)

# %% [markdown cell 114] StoreType a: khoảng -0,85%, có ý nghĩa thống kê.
# %% [markdown cell 115] kiểm tra theo Assortment

# %% [code cell 116]
assortment_map = (
    f01[["Store", "Assortment"]]
    .drop_duplicates()
)

f01_monthly_assort = f01_monthly_adj.merge(
    assortment_map,
    on="Store",
    how="left",
    validate="many_to_one"
)

result_by_assortment = []

for assort in sorted(f01_monthly_assort["Assortment"].unique()):

    temp = f01_monthly_assort[
        f01_monthly_assort["Assortment"] == assort
    ].copy()

    model = smf.ols(
        formula="""
            LogAvgSales ~
            IsPromo2Active
            + PromoRate
            + SchoolHolidayRate
            + StateHolidayRate
            + C(Store)
            + C(YearMonthStr)
        """,
        data=temp
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": temp["Store"]}
    )

    beta = model.params["IsPromo2Active"]

    result_by_assortment.append({
        "Assortment": assort,
        "Stores": temp["Store"].nunique(),
        "EstimatedPctDifference": (np.exp(beta) - 1) * 100,
        "PValue": model.pvalues["IsPromo2Active"]
    })

result_by_assortment = pd.DataFrame(result_by_assortment)

display(result_by_assortment)

# %% [markdown cell 117] Assortment a: -0,92%, có ý nghĩa thống kê.
# %% [markdown cell 118] bảng tổng hợp

# %% [code cell 119]
summary_f01 = pd.DataFrame({
    "Metric": [
        "Stores eligible",
        "Raw mean difference (%)",
        "Raw median difference (%)",
        "Adjusted difference (%)",
        "Robust adjusted difference (%)",
        "Stores with lower Sales when active",
        "Stores with higher Sales when active"
    ],
    "Value": [
        571,
        monthly_compare["Pct_Difference"].mean(),
        monthly_compare["Pct_Difference"].median(),
        -0.756539328465966,
        -0.7866826328371657,
        (monthly_compare["Difference"] < 0).sum(),
        (monthly_compare["Difference"] > 0).sum()
    ]
})

display(summary_f01)

# %% [markdown cell 120] ---
# %% [markdown cell 121] ```
# %% [markdown cell 122] so sánh thô Promo2=1 và Promo2=0

# %% [code cell 123]
promo2_raw_compare = (
    f01[f01["Open"] == 1]
    .groupby("Promo2")["Sales"]
    .agg(["mean", "median", "count"])
)

display(promo2_raw_compare)

mean_p0 = promo2_raw_compare.loc[0, "mean"]
mean_p1 = promo2_raw_compare.loc[1, "mean"]

pct_diff = (mean_p1 - mean_p0) / mean_p0 * 100

print("Chênh lệch Sales trung bình Promo2=1 vs Promo2=0:")
print(f"{pct_diff:.2f}%")

# %% [markdown cell 124] Nếu chỉ so thô giữa các cửa hàng:- Promo2 = 0: Sales trung bình ≈ 7.350,56
# %% [markdown cell 125] Correlation Sales vs CompetitionDistance

# %% [code cell 126]
# Sales trung bình theo từng Store, chỉ lấy ngày Open = 1
store_sales_corr = (
    f01[f01["Open"] == 1]
    .groupby("Store", as_index=False)["Sales"]
    .mean()
    .rename(columns={"Sales": "AvgSales"})
)

# Ghép CompetitionDistance
store_sales_corr = store_sales_corr.merge(
    store_p14[["Store", "CompetitionDistance"]],
    on="Store",
    how="left"
)

# Pearson correlation
corr = store_sales_corr[
    ["AvgSales", "CompetitionDistance"]
].corr()

display(corr)

print(
    "Pearson r:",
    store_sales_corr["AvgSales"]
    .corr(store_sales_corr["CompetitionDistance"])
)

# %% [markdown cell 127] Cũ: Pearson r = -0.04165 (sang pp mới -0.10458107366786652)  nghĩa là gần như không có mối tương qua
# %% [markdown cell 128] scatter plot để kiểm tra quan hệ cong và outlier

# %% [code cell 129]
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.scatter(
    store_sales_corr["CompetitionDistance"],
    store_sales_corr["AvgSales"],
    alpha=0.4
)

plt.xlabel("CompetitionDistance")
plt.ylabel("Average Sales")
plt.title("Average Sales vs CompetitionDistance")

plt.show()

# %% [markdown cell 130] - Không thấy xu hướng tuyến tính rõ ràng.
# %% [markdown cell 131] chuẩn bị dữ liệu cho Association Rules

# %% [code cell 132]
# Chỉ dùng ngày cửa hàng mở
assoc = f01[f01["Open"] == 1].copy()

# Median Sales riêng cho từng Store
assoc["StoreMedianSales"] = (
    assoc.groupby("Store")["Sales"]
    .transform("median")
)

# Sales High / Low so với median của chính Store
assoc["SalesLevel"] = (
    assoc["Sales"] > assoc["StoreMedianSales"]
).map({
    True: "Sales_High",
    False: "Sales_Low"
})

print(assoc["SalesLevel"].value_counts())

display(
    assoc[
        [
            "Store",
            "Sales",
            "StoreMedianSales",
            "SalesLevel",
            "Promo",
            "IsPromo2ActiveThisMonth",
            "DayOfWeek",
            "StoreType",
            "Assortment",
            "StateHoliday"
        ]
    ].head(10)
)

# %% [markdown cell 133] chuyển các biến thành item cho Association Rules

# %% [code cell 134]
assoc_rules = assoc.copy()

assoc_rules["PromoItem"] = (
    "Promo_" + assoc_rules["Promo"].astype(str)
)

assoc_rules["Promo2Item"] = (
    "Promo2Active_" +
    assoc_rules["IsPromo2ActiveThisMonth"].astype(str)
)

assoc_rules["DayItem"] = (
    "DayOfWeek_" + assoc_rules["DayOfWeek"].astype(str)
)

assoc_rules["StoreTypeItem"] = (
    "StoreType_" + assoc_rules["StoreType"].astype(str)
)

assoc_rules["AssortmentItem"] = (
    "Assortment_" + assoc_rules["Assortment"].astype(str)
)

assoc_rules["HolidayItem"] = (
    "StateHoliday_" + assoc_rules["StateHoliday"].astype(str)
)

item_cols = [
    "SalesLevel",
    "PromoItem",
    "Promo2Item",
    "DayItem",
    "StoreTypeItem",
    "AssortmentItem",
    "HolidayItem"
]

display(assoc_rules[item_cols].head(10))

print("Số dòng dùng cho Association Rules:", len(assoc_rules))

# %% [markdown cell 135] One-hot encoding cho Association Rules

# %% [code cell 136]
basket = pd.concat(
    [
        pd.get_dummies(assoc_rules[col])
        for col in item_cols
    ],
    axis=1
).astype(bool)

print("Shape:", basket.shape)

print("\nCác item:")
print(basket.columns.tolist())

display(basket.head())

# %% [markdown cell 137] tìm frequent itemsets bằng Apriori

# %% [code cell 138]
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(
    basket,
    min_support=0.02,
    use_colnames=True,
    max_len=3,
    low_memory=True
)

frequent_itemsets = frequent_itemsets.sort_values(
    "support",
    ascending=False
).reset_index(drop=True)

print("Số frequent itemsets:", len(frequent_itemsets))

display(frequent_itemsets.head(20))

# %% [markdown cell 139] 524 frequent itemsets. Cảnh báo DeprecationWarning kia không ảnh hưởng kết quả
# %% [markdown cell 140] Association Rules và lọc Lift > 1

# %% [code cell 141]
from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

# Chỉ giữ rule có consequent liên quan đến Sales
sales_rules = rules[
    rules["consequents"].apply(
        lambda x: ("Sales_Low" in x) or ("Sales_High" in x)
    )
].copy()

# Sắp xếp theo Lift giảm dần
sales_rules = sales_rules.sort_values(
    ["lift", "confidence"],
    ascending=False
).reset_index(drop=True)

print("Số rule liên quan đến Sales có Lift > 1:",
      len(sales_rules))

display(
    sales_rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(30)
)

# %% [markdown cell 142] lọc riêng rule Promo2 active → Sales

# %% [code cell 143]
promo2_sales_rules = sales_rules[
    sales_rules["antecedents"].apply(
        lambda x: "Promo2Active_1" in x
    )
    &
    sales_rules["consequents"].apply(
        lambda x: x == frozenset({"Sales_Low"})
        or x == frozenset({"Sales_High"})
    )
].copy()

promo2_sales_rules = promo2_sales_rules.sort_values(
    ["lift", "confidence"],
    ascending=False
)

display(
    promo2_sales_rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]
)

# %% [markdown cell 144] Promo2Active_1 → Sales_Low: Lift = 1.006, Confidence ≈ 50.36%
# %% [markdown cell 145] Selection bias: Promo2=1 vs Promo2=0 chênh -10.78%
# %% [markdown cell 146] ---
# %% [markdown cell 147] E01 - robustness test của phân tích 1
# %% [markdown cell 148] bảng kết quả theo từng Store

# %% [code cell 149]
e01_store = monthly_compare.reset_index().copy()

# Active > Inactive nếu % chênh lệch > 0
e01_store["ActiveHigher"] = (
    e01_store["Pct_Difference"] > 0
)

# Thêm StoreType và Assortment
e01_store = e01_store.merge(
    store_p14[
        ["Store", "StoreType", "Assortment"]
    ],
    on="Store",
    how="left"
)

print("Số Store:", len(e01_store))

print(
    "Số Store Active > Inactive:",
    e01_store["ActiveHigher"].sum()
)

print(
    "Tỷ lệ Active > Inactive:",
    round(e01_store["ActiveHigher"].mean() * 100, 2),
    "%"
)

print("\nTheo StoreType:")
display(
    e01_store.groupby("StoreType").agg(
        Stores=("Store", "count"),
        ActiveHigherStores=("ActiveHigher", "sum"),
        ActiveHigherRate=("ActiveHigher", "mean"),
        MeanPctDifference=("Pct_Difference", "mean"),
        MedianPctDifference=("Pct_Difference", "median")
    )
)

print("\nTheo Assortment:")
display(
    e01_store.groupby("Assortment").agg(
        Stores=("Store", "count"),
        ActiveHigherStores=("ActiveHigher", "sum"),
        ActiveHigherRate=("ActiveHigher", "mean"),
        MeanPctDifference=("Pct_Difference", "mean"),
        MedianPctDifference=("Pct_Difference", "median")
    )
)

# %% [markdown cell 150] Chỉ 145/571 Store = 25,39% có Sales trung bình tháng Promo2 active cao hơn inactive; tức khoảng 74,6
# %% [markdown cell 151] Nhìn chung, association âm của Promo2 không chỉ xuất hiện ở một StoreType hoặc Assortment duy nhất.
# %% [markdown cell 152] xác định Store outlier

# %% [code cell 153]
q1 = e01_store["Pct_Difference"].quantile(0.25)
q3 = e01_store["Pct_Difference"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

e01_outliers = e01_store[
    (e01_store["Pct_Difference"] < lower_bound) |
    (e01_store["Pct_Difference"] > upper_bound)
].copy()

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

print("Số Store outlier:", len(e01_outliers))

display(
    e01_outliers[
        [
            "Store",
            "StoreType",
            "Assortment",
            "Pct_Difference"
        ]
    ].sort_values("Pct_Difference")
)

# %% [markdown cell 154] IQR bounds: [-10,47%; 6,32%]
# %% [markdown cell 155] outlier không chỉ kéo kết quả xuống; thực tế phần lớn outlier đang kéo kết quả lên. Vì vậy nếu bỏ ch
# %% [markdown cell 156] Bỏ outlier + StoreType b rồi chạy lại đúng mô hình OLS

# %% [code cell 157]
# Danh sách Store outlier
outlier_store_ids = e01_outliers["Store"].tolist()

# Gắn StoreType vào bảng monthly regression
e01_robust = f01_monthly_adj.merge(
    store_p14[["Store", "StoreType"]],
    on="Store",
    how="left"
)

# Bỏ Store outlier và StoreType b
e01_robust = e01_robust[
    (~e01_robust["Store"].isin(outlier_store_ids)) &
    (e01_robust["StoreType"] != "b")
].copy()

# YearMonth dạng text cho fixed effects
e01_robust["YearMonthStr"] = e01_robust["YearMonth"].astype(str)

# Log Sales
e01_robust["LogAvgSales"] = np.log(e01_robust["AvgSales"])

# Chạy lại cùng specification với M01
robust_model = smf.ols(
    """
    LogAvgSales ~ IsPromo2Active
    + PromoRate
    + SchoolHolidayRate
    + StateHolidayRate
    + C(Store)
    + C(YearMonthStr)
    """,
    data=e01_robust
).fit(
    cov_type="cluster",
    cov_kwds={"groups": e01_robust["Store"]}
)

beta = robust_model.params["IsPromo2Active"]
pvalue = robust_model.pvalues["IsPromo2Active"]
ci = robust_model.conf_int().loc["IsPromo2Active"]

pct_effect = (np.exp(beta) - 1) * 100

print("Số Store còn lại:", e01_robust["Store"].nunique())
print("Số Store-Month còn lại:", len(e01_robust))

print("\nBeta:", beta)
print("Pct effect:", pct_effect, "%")
print("p-value:", pvalue)
print("95% CI:", ci.tolist())

# %% [markdown cell 158] robustness check 1 giữ nguyên chiều và thậm chí âm mạnh hơn nhẹ:
# %% [markdown cell 159] robustness check 2: đổi cách xác định tháng active.
# %% [markdown cell 160] tạo định nghĩa active chặt hơn

# %% [code cell 161]
e01_strict = f01[
    (f01["Promo2"] == 1) &
    (f01["Open"] == 1) &
    (f01["Date"] >= f01["Promo2StartDate"])
].copy()

# Tháng của từng quan sát
e01_strict["YearMonth"] = e01_strict["Date"].dt.to_period("M")

# Ngày đầu tháng
e01_strict["MonthStart"] = (
    e01_strict["Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Loại tháng bắt đầu Promo2 giữa tháng
e01_strict = e01_strict[
    e01_strict["Promo2StartDate"] <= e01_strict["MonthStart"]
].copy()

# Tổng hợp lại Store × Month
e01_strict_monthly = (
    e01_strict
    .groupby(["Store", "YearMonth"], as_index=False)
    .agg(
        AvgSales=("Sales", "mean"),
        OpenDays=("Sales", "size"),
        IsPromo2Active=("IsPromo2ActiveThisMonth", "max")
    )
)

# Giữ tháng có ít nhất 5 ngày mở cửa
e01_strict_monthly = e01_strict_monthly[
    e01_strict_monthly["OpenDays"] >= 5
].copy()

print("Số Store:", e01_strict_monthly["Store"].nunique())
print("Số Store-Month:", len(e01_strict_monthly))

print("\nActive / Inactive months:")
print(e01_strict_monthly["IsPromo2Active"].value_counts())

# Store có cả active và inactive
status_count = (
    e01_strict_monthly
    .groupby("Store")["IsPromo2Active"]
    .nunique()
)

eligible_strict = status_count[
    status_count == 2
].index

print(
    "\nSố Store có cả active và inactive:",
    len(eligible_strict)
)

# %% [markdown cell 162] - Vẫn đủ 571 Store.
# %% [markdown cell 163] tính lại within-store difference theo định nghĩa strict

# %% [code cell 164]
strict_compare = (
    e01_strict_monthly
    .pivot_table(
        index="Store",
        columns="IsPromo2Active",
        values="AvgSales",
        aggfunc="mean"
    )
    .dropna()
    .rename(
        columns={
            0: "InactiveSales",
            1: "ActiveSales"
        }
    )
)

strict_compare["Pct_Difference"] = (
    (
        strict_compare["ActiveSales"]
        - strict_compare["InactiveSales"]
    )
    / strict_compare["InactiveSales"]
    * 100
)

print("Số Store:", len(strict_compare))

print(
    "Mean % difference:",
    strict_compare["Pct_Difference"].mean()
)

print(
    "Median % difference:",
    strict_compare["Pct_Difference"].median()
)

print(
    "Store Active > Inactive:",
    (strict_compare["Pct_Difference"] > 0).sum()
)

print(
    "Store Active < Inactive:",
    (strict_compare["Pct_Difference"] < 0).sum()
)

print(
    "Tỷ lệ Active > Inactive:",
    round(
        (strict_compare["Pct_Difference"] > 0).mean() * 100,
        2
    ),
    "%"
)

# %% [markdown cell 165] - Baseline mean difference: -1,69%
# %% [markdown cell 166] ---
# %% [markdown cell 167] D01 – Business Memo: cửa hàng/chiến dịch nào hoạt động kém.
# %% [markdown cell 168] bảng hiệu quả Promo2 theo từng Store

# %% [code cell 169]
# Trung bình Sales + số tháng Active/Inactive của từng Store
d01_store = (
    f01_monthly
    .groupby(["Store", "IsPromo2Active"])
    .agg(
        AvgSales=("AvgSales", "mean"),
        Months=("YearMonth", "nunique")
    )
    .reset_index()
)

# Đưa Active / Inactive thành các cột riêng
sales_wide = d01_store.pivot(
    index="Store",
    columns="IsPromo2Active",
    values="AvgSales"
).rename(columns={
    0: "InactiveSales",
    1: "ActiveSales"
})

months_wide = d01_store.pivot(
    index="Store",
    columns="IsPromo2Active",
    values="Months"
).rename(columns={
    0: "InactiveMonths",
    1: "ActiveMonths"
})

# Ghép lại
d01_store_perf = (
    sales_wide
    .join(months_wide)
    .dropna()
    .reset_index()
)

# % chênh lệch Sales
d01_store_perf["Pct_Difference"] = (
    (
        d01_store_perf["ActiveSales"]
        - d01_store_perf["InactiveSales"]
    )
    / d01_store_perf["InactiveSales"]
    * 100
)

# Thêm thông tin cửa hàng
d01_store_perf = d01_store_perf.merge(
    store_p14[
        [
            "Store",
            "StoreType",
            "Assortment",
            "PromoInterval"
        ]
    ],
    on="Store",
    how="left"
)

print("Số Store:", len(d01_store_perf))

display(
    d01_store_perf[
        [
            "Store",
            "StoreType",
            "Assortment",
            "PromoInterval",
            "ActiveMonths",
            "InactiveMonths",
            "InactiveSales",
            "ActiveSales",
            "Pct_Difference"
        ]
    ].head(10)
)

print("\nPhân bố số tháng Active:")
print(d01_store_perf["ActiveMonths"].describe())

print("\nPhân bố số tháng Inactive:")
print(d01_store_perf["InactiveMonths"].describe())

# %% [markdown cell 170] có Store chỉ có 1 tháng active hoặc 1 tháng inactive, nên nếu xếp hạng trực tiếp thì rất dễ bị nhiễu
# %% [markdown cell 171] ***
# %% [markdown cell 172] lọc Store đủ dữ liệu và tìm nhóm kém nhất

# %% [code cell 173]
# Chỉ giữ Store có đủ số tháng để so sánh
d01_eligible = d01_store_perf[
    (d01_store_perf["ActiveMonths"] >= 3) &
    (d01_store_perf["InactiveMonths"] >= 3)
].copy()

# Sắp xếp từ kém nhất đến tốt nhất
d01_eligible = d01_eligible.sort_values(
    "Pct_Difference",
    ascending=True
)

print("Số Store đủ điều kiện:", len(d01_eligible))

print("\n10 Store có kết quả Promo2 kém nhất:")
display(
    d01_eligible[
        [
            "Store",
            "StoreType",
            "Assortment",
            "PromoInterval",
            "ActiveMonths",
            "InactiveMonths",
            "InactiveSales",
            "ActiveSales",
            "Pct_Difference"
        ]
    ].head(10)
)

# %% [markdown cell 174] 560 Store đủ điều kiện
# %% [markdown cell 175] so sánh theo PromoInterval

# %% [code cell 176]
d01_campaign = (
    d01_eligible
    .groupby("PromoInterval")
    .agg(
        Stores=("Store", "count"),
        MeanPctDifference=("Pct_Difference", "mean"),
        MedianPctDifference=("Pct_Difference", "median"),
        PositiveStores=("Pct_Difference", lambda x: (x > 0).sum()),
        NegativeStores=("Pct_Difference", lambda x: (x < 0).sum())
    )
    .reset_index()
)

d01_campaign["PositiveRate"] = (
    d01_campaign["PositiveStores"]
    / d01_campaign["Stores"]
    * 100
)

d01_campaign["NegativeRate"] = (
    d01_campaign["NegativeStores"]
    / d01_campaign["Stores"]
    * 100
)

d01_campaign = d01_campaign.sort_values(
    "MeanPctDifference"
)

display(d01_campaign)

# %% [markdown cell 177] Jan,Apr,Jul,Oct: mean -3,25%, 89,73% Store âm.
# %% [markdown cell 178] chưa được kết luận lịch thứ ba “tốt” và hai lịch đầu “phế”, vì mỗi PromoInterval gắn với các tháng k
# %% [markdown cell 179] Bước 4: kiểm tra từng PromoInterval sau khi control seasonality

# %% [code cell 180]
# Gắn PromoInterval vào bảng monthly dùng cho regression
d01_interval_reg = f01_monthly_adj.merge(
    store_p14[["Store", "PromoInterval"]],
    on="Store",
    how="left"
)

# Chỉ giữ 560 Store đủ điều kiện của D01
eligible_store_ids = d01_eligible["Store"].unique()

d01_interval_reg = d01_interval_reg[
    d01_interval_reg["Store"].isin(eligible_store_ids)
].copy()

# Chuẩn bị biến cho regression
d01_interval_reg["YearMonthStr"] = (
    d01_interval_reg["YearMonth"].astype(str)
)

d01_interval_reg["LogAvgSales"] = np.log(
    d01_interval_reg["AvgSales"]
)

interval_results = []

for interval in sorted(
    d01_interval_reg["PromoInterval"].dropna().unique()
):

    temp = d01_interval_reg[
        d01_interval_reg["PromoInterval"] == interval
    ].copy()

    model = smf.ols(
        """
        LogAvgSales ~ IsPromo2Active
        + PromoRate
        + SchoolHolidayRate
        + StateHolidayRate
        + C(Store)
        + C(YearMonthStr)
        """,
        data=temp
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": temp["Store"]}
    )

    beta = model.params["IsPromo2Active"]
    ci = model.conf_int().loc["IsPromo2Active"]

    interval_results.append({
        "PromoInterval": interval,
        "Stores": temp["Store"].nunique(),
        "Beta": beta,
        "PctEffect": (np.exp(beta) - 1) * 100,
        "p_value": model.pvalues["IsPromo2Active"],
        "CI_Lower": (np.exp(ci.iloc[0]) - 1) * 100,
        "CI_Upper": (np.exp(ci.iloc[1]) - 1) * 100
    })

d01_interval_adjusted = pd.DataFrame(interval_results)

display(d01_interval_adjusted)

# %% [markdown cell 181] khi tách riêng từng PromoInterval, trạng thái active gần như được quyết định bởi chính tháng trong n
# %% [markdown cell 182] sửa: pooled model với interaction theo lịch Promo2

# %% [code cell 183]
d01_interval_reg = f01_monthly_adj.merge(
    store_p14[["Store", "PromoInterval"]],
    on="Store",
    how="left"
)

# Chỉ giữ các Store đủ điều kiện của D01
eligible_store_ids = d01_eligible["Store"].unique()

d01_interval_reg = d01_interval_reg[
    d01_interval_reg["Store"].isin(eligible_store_ids)
].copy()

# Chuẩn bị biến cho regression
d01_interval_reg["YearMonthStr"] = (
    d01_interval_reg["YearMonth"].astype(str)
)

d01_interval_reg["LogAvgSales"] = np.log(
    d01_interval_reg["AvgSales"]
)

print("Shape:", d01_interval_reg.shape)
print("Số Store:", d01_interval_reg["Store"].nunique())

print("\nPromoInterval:")
print(d01_interval_reg["PromoInterval"].value_counts())

# %% [markdown cell 184] đm colab lỗi lòn dm colab
# %% [markdown cell 185] check kiểm tra session

# %% [code cell 186]
check_vars = [
    "train_p14",
    "store_p14",
    "f01",
    "f01_monthly",
    "f01_monthly_adj",
    "d01_store_perf",
    "d01_eligible"
]

for var in check_vars:
    print(
        var,
        "✅ còn" if var in globals() else "❌ mất"
    )

# %% [markdown cell 187] đang check lại mẫu

# %% [code cell 188]
print("f01_monthly_adj shape:", f01_monthly_adj.shape)
print(
    "Store trong f01_monthly_adj:",
    f01_monthly_adj["Store"].nunique()
)

print("\nd01_eligible:")
print("Rows:", len(d01_eligible))
print(
    "Stores:",
    d01_eligible["Store"].nunique()
)

missing_stores = set(d01_eligible["Store"]) - set(
    f01_monthly_adj["Store"]
)

print(
    "\nSố Store D01 bị thiếu trong f01_monthly_adj:",
    len(missing_stores)
)

print(
    "Một số Store bị thiếu:",
    sorted(list(missing_stores))[:30]
)

# %% [markdown cell 189] f01_monthly_adj hiện vẫn có đủ 571 Store, nhưng d01_eligible chỉ còn 539 Store. Nghĩa là biến d01_el
# %% [markdown cell 190] Đọaun sau đang check đã, nào ấy xong thì note thêm sau

# %% [code cell 191]
print("f01_monthly shape:", f01_monthly.shape)
print("Số Store:", f01_monthly["Store"].nunique())

print("\nColumns:")
print(f01_monthly.columns.tolist())

print("\nActive / Inactive:")
print(f01_monthly["IsPromo2Active"].value_counts())

# %% [markdown cell 192] pooled regression theo từng lịch Promo2

# %% [code cell 193]
# Tạo lại bảng regression cho D01
d01_interval_reg = f01_monthly_adj.merge(
    store_p14[["Store", "PromoInterval"]],
    on="Store",
    how="left"
)

# Chỉ giữ các Store đủ điều kiện của D01
eligible_store_ids = d01_eligible["Store"].unique()

d01_interval_reg = d01_interval_reg[
    d01_interval_reg["Store"].isin(eligible_store_ids)
].copy()

# Chuẩn bị biến
d01_interval_reg["YearMonthStr"] = (
    d01_interval_reg["YearMonth"].astype(str)
)

d01_interval_reg["LogAvgSales"] = np.log(
    d01_interval_reg["AvgSales"]
)

# Tạo biến active riêng cho 3 lịch Promo2
d01_interval_reg["Active_JanAprJulOct"] = (
    (d01_interval_reg["PromoInterval"] == "Jan,Apr,Jul,Oct")
    & (d01_interval_reg["IsPromo2Active"] == 1)
).astype(int)

d01_interval_reg["Active_FebMayAugNov"] = (
    (d01_interval_reg["PromoInterval"] == "Feb,May,Aug,Nov")
    & (d01_interval_reg["IsPromo2Active"] == 1)
).astype(int)

d01_interval_reg["Active_MarJunSeptDec"] = (
    (d01_interval_reg["PromoInterval"] == "Mar,Jun,Sept,Dec")
    & (d01_interval_reg["IsPromo2Active"] == 1)
).astype(int)

# Pooled regression
d01_schedule_model = smf.ols(
    """
    LogAvgSales ~
    Active_JanAprJulOct
    + Active_FebMayAugNov
    + Active_MarJunSeptDec
    + PromoRate
    + SchoolHolidayRate
    + StateHolidayRate
    + C(Store)
    + C(YearMonthStr)
    """,
    data=d01_interval_reg
).fit(
    cov_type="cluster",
    cov_kwds={"groups": d01_interval_reg["Store"]}
)

# In kết quả
schedule_vars = [
    "Active_JanAprJulOct",
    "Active_FebMayAugNov",
    "Active_MarJunSeptDec"
]

for var in schedule_vars:
    beta = d01_schedule_model.params[var]
    ci = d01_schedule_model.conf_int().loc[var]

    print("\n", var)
    print("Beta:", beta)
    print(
        "Pct effect:",
        (np.exp(beta) - 1) * 100,
        "%"
    )
    print(
        "p-value:",
        d01_schedule_model.pvalues[var]
    )
    print(
        "95% CI:",
        [
            (np.exp(ci.iloc[0]) - 1) * 100,
            (np.exp(ci.iloc[1]) - 1) * 100
        ]
    )


# %% [code cell 194]
# Tạo lại bảng monthly adjusted từ f01

temp_adj = f01[
    (f01["Promo2"] == 1) &
    (f01["Date"] >= f01["Promo2StartDate"]) &
    (f01["Open"] == 1)
].copy()

# Biến thời gian
temp_adj["YearMonth"] = temp_adj["Date"].dt.to_period("M")

# Có StateHoliday hay không
temp_adj["HasStateHoliday"] = (
    temp_adj["StateHoliday"].astype(str) != "0"
).astype(int)

# Tổng hợp Store × Month
f01_monthly_adj = (
    temp_adj
    .groupby(["Store", "YearMonth"], as_index=False)
    .agg(
        AvgSales=("Sales", "mean"),
        OpenDays=("Sales", "size"),
        IsPromo2Active=("IsPromo2ActiveThisMonth", "max"),
        PromoRate=("Promo", "mean"),
        SchoolHolidayRate=("SchoolHoliday", "mean"),
        StateHolidayRate=("HasStateHoliday", "mean")
    )
)

# Loại Store-Month có dưới 5 ngày mở cửa
f01_monthly_adj = f01_monthly_adj[
    f01_monthly_adj["OpenDays"] >= 5
].copy()

# Tạo Year và Month
f01_monthly_adj["Year"] = (
    f01_monthly_adj["YearMonth"].dt.year
)

f01_monthly_adj["Month"] = (
    f01_monthly_adj["YearMonth"].dt.month
)

print("Shape:", f01_monthly_adj.shape)
print("Số Store:", f01_monthly_adj["Store"].nunique())

print("\nActive / Inactive:")
print(
    f01_monthly_adj["IsPromo2Active"].value_counts()
)

# %% [markdown cell 195] xác định nhóm Store “underperforming nặng”

# %% [code cell 196]
# Ngưỡng bottom 10%
cutoff = d01_eligible["Pct_Difference"].quantile(0.10)

d01_bottom10 = d01_eligible[
    d01_eligible["Pct_Difference"] <= cutoff
].copy()

print("Ngưỡng bottom 10%:", cutoff)
print("Số Store bottom 10%:", len(d01_bottom10))

print("\nTheo PromoInterval:")
display(
    d01_bottom10.groupby("PromoInterval").agg(
        Stores=("Store", "count"),
        MeanPctDifference=("Pct_Difference", "mean"),
        MedianPctDifference=("Pct_Difference", "median")
    ).sort_values("Stores", ascending=False)
)

print("\nTheo StoreType:")
display(
    d01_bottom10.groupby("StoreType").agg(
        Stores=("Store", "count"),
        MeanPctDifference=("Pct_Difference", "mean")
    ).sort_values("Stores", ascending=False)
)

print("\nTheo Assortment:")
display(
    d01_bottom10.groupby("Assortment").agg(
        Stores=("Store", "count"),
        MeanPctDifference=("Pct_Difference", "mean")
    ).sort_values("Stores", ascending=False)
)

# %% [markdown cell 197] c: 35 Store, mean difference -6,39%
# %% [markdown cell 198] => Bottom 10% có tổng 56 Store. Assortment c xuất hiện nhiều hơn, nhưng mức giảm trung bình giữa a v
# %% [markdown cell 199] tính tỷ lệ bottom 10% trong từng nhóm

# %% [code cell 200]
# PromoInterval
promo_total = (
    d01_eligible
    .groupby("PromoInterval")["Store"]
    .count()
    .rename("TotalStores")
)

promo_bottom = (
    d01_bottom10
    .groupby("PromoInterval")["Store"]
    .count()
    .rename("Bottom10Stores")
)

promo_rate = pd.concat(
    [promo_total, promo_bottom],
    axis=1
).fillna(0)

promo_rate["Bottom10Rate"] = (
    promo_rate["Bottom10Stores"]
    / promo_rate["TotalStores"]
    * 100
)

print("Theo PromoInterval:")
display(promo_rate)


# StoreType
type_total = (
    d01_eligible
    .groupby("StoreType")["Store"]
    .count()
    .rename("TotalStores")
)

type_bottom = (
    d01_bottom10
    .groupby("StoreType")["Store"]
    .count()
    .rename("Bottom10Stores")
)

type_rate = pd.concat(
    [type_total, type_bottom],
    axis=1
).fillna(0)

type_rate["Bottom10Rate"] = (
    type_rate["Bottom10Stores"]
    / type_rate["TotalStores"]
    * 100
)

print("\nTheo StoreType:")
display(type_rate)


# Assortment
assort_total = (
    d01_eligible
    .groupby("Assortment")["Store"]
    .count()
    .rename("TotalStores")
)

assort_bottom = (
    d01_bottom10
    .groupby("Assortment")["Store"]
    .count()
    .rename("Bottom10Stores")
)

assort_rate = pd.concat(
    [assort_total, assort_bottom],
    axis=1
).fillna(0)

assort_rate["Bottom10Rate"] = (
    assort_rate["Bottom10Stores"]
    / assort_rate["TotalStores"]
    * 100
)

print("\nTheo Assortment:")
display(assort_rate)


# %% [code cell 201]
import pandas as pd

store_p14 = pd.read_csv("/content/store_P14.csv")

print(store_p14[store_p14["Promo2"] == 1]["PromoInterval"].value_counts())


# %% [code cell 202]
import pandas as pd

train = pd.read_csv("/content/train_P14.csv")
store = pd.read_csv("/content/store_P14.csv")

train["Date"] = pd.to_datetime(train["Date"])

df = train.merge(
    store[["Store", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]],
    on="Store",
    how="left"
)

# Ngày bắt đầu Promo2
df["Promo2StartDate"] = pd.to_datetime(
    df["Promo2SinceYear"].astype("Int64").astype(str)
    + "-W"
    + df["Promo2SinceWeek"].astype("Int64").astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)

month_name = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",
    5:"May", 6:"Jun", 7:"Jul", 8:"Aug",
    9:"Sept", 10:"Oct", 11:"Nov", 12:"Dec"
}

df["MonthName"] = df["Date"].dt.month.map(month_name)

df["Active"] = (
    (df["Promo2"] == 1) &
    (df["Date"] >= df["Promo2StartDate"]) &
    df.apply(
        lambda x: x["MonthName"] in str(x["PromoInterval"]).split(","),
        axis=1
    )
).astype(int)

# Gom về Store - Year - Month
check = (
    df.groupby(["Store", df["Date"].dt.year.rename("Year"),
                df["Date"].dt.month.rename("Month")])["Active"]
      .max()
      .reset_index()
)

# Tìm cùng Store + cùng tháng nhưng các năm có trạng thái Active khác nhau
result = (
    check.groupby(["Store", "Month"])
         .filter(lambda x: x["Active"].nunique() > 1)
         .sort_values(["Store", "Month", "Year"])
)

print(result.head(50))

