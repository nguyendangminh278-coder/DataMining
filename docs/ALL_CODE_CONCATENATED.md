# ALL CODE CELLS — reference only

This file concatenates code-cell exports for search and review. **Do not run this as one script.** Execute notebooks in their intended environments and phase sequence.

---

## scripts/01_P05_Clean_Data.py

```python
# Notebook export: 01_P05_Clean_Data
# Original source: https://drive.google.com/file/d/1Y53YsoRV7zRtJhLEgjOn6Xrp_Ds5irs2/view
# Automatically extracted code cells only. The .ipynb is authoritative.
# Notebook magics/shell commands, if any, are commented out.

# %% [markdown cell 1] # P05 Làm sạch dữ liệu đơn giản
# %% [markdown cell 2] ## 1 Đọc ba file gốc

# %% [code cell 3]
from pathlib import Path
import pandas as pd
root = Path.cwd()
train = pd.read_csv(root / "input/train.csv", dtype={"StateHoliday": "string"})
test = pd.read_csv(root / "input/test.csv", dtype={"StateHoliday": "string"})
store = pd.read_csv(root / "input/store.csv")
# Đọc riêng từng bảng để giữ nguyên cấu trúc gốc, không ghép hoặc tạo cột mới.
print("Train, test, store:", train.shape, test.shape, store.shape)

# %% [markdown cell 4] Lý do: train là lịch sử bán hàng, test là các ngày cần dự báo, store là thông tin cửa hàng. Giữ riên
# %% [markdown cell 5] ## 2 Kiểm tra ô trống và dòng trùng

# %% [code cell 6]
for name, data in [("train", train), ("test", test), ("store", store)]:
    print(name, "missing:", data.isna().sum().loc[lambda x: x > 0].to_dict())
    key = ["Store"] if name == "store" else ["Store", "Date"]
    assert not data.duplicated(key).any(), f"{name}: cần kiểm tra khóa bị trùng"
# Không có khóa trùng nên không xóa dòng. Nếu nguồn thay đổi, dừng để kiểm tra.

# %% [markdown cell 7] Lý do: cần biết chính xác ô nào bị thiếu trước khi điền. Cùng Store và Date chỉ được có một dòng. Kế
# %% [markdown cell 8] ## 3 Kiểm tra lịch mở cửa rồi điền Open

# %% [code cell 9]
missing = test["Open"].isna()
history = train.loc[(train.Store == 622) & (train.DayOfWeek <= 6) & (train.StateHoliday == "0"), "Open"]
assert missing.sum() == 11 and test.loc[missing, "Store"].eq(622).all()
assert test.loc[missing, "DayOfWeek"].le(6).all() and test.loc[missing, "StateHoliday"].eq("0").all()
assert len(history) > 0 and history.eq(1).all()
print("Ngày lịch sử phù hợp:", len(history), "Tỷ lệ mở:", history.mean())
test.loc[missing, "Open"] = 1
# Chỉ điền 11 ô thiếu vì cửa hàng 622 luôn mở vào ngày thường tương ứng trong train.

# %% [markdown cell 10] Lý do: 11 ô trống đều thuộc Store 622 vào thứ Hai đến thứ Bảy, không phải ngày lễ. Trong tập train c
# %% [markdown cell 11] ## 4 Điền khoảng cách bằng trung vị

# %% [code cell 12]
distance = store["CompetitionDistance"].median()
store["CompetitionDistance"] = store["CompetitionDistance"].fillna(distance)
# Trung vị ít bị kéo lên bởi vài khoảng cách rất lớn; không hiểu ô thiếu là 0 mét.
print("Khoảng cách dùng để điền:", distance, "mét")

# %% [markdown cell 13] Lý do: lấy giá trị đứng giữa các khoảng cách đã biết trong store, bằng 2.325 mét. Điền cùng giá trị 
# %% [markdown cell 14] ## 5 Giữ các ô không áp dụng hoặc chưa biết

# %% [code cell 15]
assert store.loc[store.Promo2 == 0, ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]].isna().all().all()
# Không tham gia Promo2 thì không có lịch tham gia; tháng/năm đối thủ chưa biết cũng giữ trống.
assert test.Open.notna().all() and store.CompetitionDistance.notna().all()

# %% [markdown cell 16] Lý do: 544 cửa hàng không tham gia Promo2 nên không có tuần, năm và lịch Promo2. 354 cửa hàng chưa b
# %% [markdown cell 17] ## 6 Xuất ba file sạch

# %% [code cell 18]
(root / "output").mkdir(exist_ok=True)
for name, data in [("train", train), ("test", test), ("store", store)]:
    data.to_csv(root / "output" / f"{name}_clean.csv", index=False, encoding="utf-8-sig")
    print(f"{name}_clean.csv:", data.shape)
# Giữ toàn bộ dòng và cột gốc; P06 bỏ, không sampling, không Feature Engineering.

# %% [markdown cell 19] Lý do: lưu output riêng để không đè file gốc. Kết quả gồm train_clean.csv (1.017.209 dòng, 9 cột), t
# %% [markdown cell 20] ## Các bước không làm trong bản này
# %% [markdown cell 21] ## Đối chiếu với EDA ngày 20/10/2026

```

---

## scripts/02_U02_EDA_RootCause.py

```python
# Notebook export: 02_U02_EDA_RootCause
# Original source: https://drive.google.com/file/d/1xeItsAqHjJCWqU9LMK6Iz8jo8Cygg1mJ/view
# Automatically extracted code cells only. The .ipynb is authoritative.
# Notebook magics/shell commands, if any, are commented out.

# %% [markdown cell 1] # U02 - EDA

# %% [code cell 2]
# JUPYTER_ONLY: %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
pd.set_option('display.max_columns', 50)


# %% [markdown cell 3] ## 0. Load dữ liệu & Data Dictionary tóm tắt

# %% [code cell 4]
train = pd.read_csv('train_ross.csv', parse_dates=['Date'])
test  = pd.read_csv('test_ross.csv',  parse_dates=['Date'])
store = pd.read_csv('store_ross.csv')

print("train :", train.shape)
print("test  :", test.shape)
print("store :", store.shape)
print()
print("train date range:", train.Date.min().date(), "->", train.Date.max().date())
print("test  date range:", test.Date.min().date(),  "->", test.Date.max().date())
gap = (test.Date.min() - train.Date.max()).days
span_weeks = (test.Date.max() - test.Date.min()).days / 7
print(f"Khoảng cách train_max -> test_min: {gap} ngày | test span: {span_weeks:.1f} tuần")


# %% [markdown cell 5] **Nhận xét:** test.csv bắt đầu ngay sau train.csv (cách 1 ngày) và trải dài ~6.7 tuần. → Mission 2 p
# %% [markdown cell 6] ---

# %% [code cell 7]
print("train.csv missing")
print(train.isna().sum()[train.isna().sum() > 0])
print("\ntest.csv missing")
print(test.isna().sum()[test.isna().sum() > 0])
print("\nstore.csv missing")
print(store.isna().sum())



# %% [code cell 8]
n_missing_interval = store['PromoInterval'].isna().sum()
n_promo2_0 = (store['Promo2']==0).sum()
print(f"PromoInterval missing: {n_missing_interval}  |  Promo2=0: {n_promo2_0}  |  Khớp: {n_missing_interval==n_promo2_0}")

# %% [markdown cell 9] **Nhận xét:**
# %% [markdown cell 10] ### 1.2. StateHoliday

# %% [code cell 11]
for v in train['StateHoliday'].unique():
    print(f"  value={v!r:8}  type={type(v).__name__}")
print()
print(train['StateHoliday'].value_counts(dropna=False))



# %% [code cell 12]
train['StateHoliday'] = train['StateHoliday'].astype(str).replace('0.0', '0')
test['StateHoliday']  = test['StateHoliday'].astype(str).replace('0.0', '0')
print(train['StateHoliday'].value_counts())


# %% [markdown cell 13] **Nhận xét:** '0' (string) và 0 (số) bị tách thành 2 nhóm khác nhau khi đọc bằng pandas → **bắt buộc
# %% [markdown cell 14] ### 1.3. Logic Open ↔ Sales

# %% [code cell 15]
open0_sales_nonzero = ((train.Open==0) & (train.Sales!=0)).sum()
open1_sales_zero    = ((train.Open==1) & (train.Sales==0)).sum()
print(f"Open=0 nhưng Sales!=0: {open0_sales_nonzero} dòng (kỳ vọng 0)")
print(f"Open=1 nhưng Sales=0 : {open1_sales_zero} dòng (case lạ)")

anomaly = train[(train.Open==1) & (train.Sales==0)]
print("\nTheo DayOfWeek:"); print(anomaly['DayOfWeek'].value_counts().sort_index())


# %% [markdown cell 16] **Nhận xét:** 0 dòng vi phạm Open=0 → Sales=0 (Mục đích kiểm tra: Có ngày nào cửa hàng đóng cửa (Ope
# %% [markdown cell 17] ### 1.4. PromoInterval

# %% [code cell 18]
print(store['PromoInterval'].dropna().unique())
print()
print(store[store.Promo2==1]['PromoInterval'].value_counts())


# %% [markdown cell 19] **Nhận xét:** Toàn bộ 571 CH Promo2=1 chỉ rơi vào đúng 3 chu kỳ (`Jan,Apr,Jul,Oct` / `Feb,May,Aug,No
# %% [markdown cell 20] ### 1.5. Sample size theo StoreType / Assortment

# %% [code cell 21]
fig, axes = plt.subplots(1, 2, figsize=(11,4))
st_counts = store['StoreType'].value_counts().sort_index()
axes[0].bar(st_counts.index, st_counts.values, color='#4472C4')
for i,v in enumerate(st_counts.values): axes[0].text(i, v+10, str(v), ha='center')
axes[0].set_title('Số cửa hàng theo StoreType (tổng 1.115)')

as_counts = store['Assortment'].value_counts().sort_index()
axes[1].bar(as_counts.index, as_counts.values, color='#ED7D31')
for i,v in enumerate(as_counts.values): axes[1].text(i, v+10, str(v), ha='center')
axes[1].set_title('Số cửa hàng theo Assortment (tổng 1.115)')
plt.tight_layout(); plt.show()
print(st_counts); print(); print(as_counts)


# %% [markdown cell 22] **Nhận xét:** StoreType `b` (17 CH, 1.5%) và Assortment `b` (9 CH, 0.8%) quá nhỏ → mọi kết luận liên
# %% [markdown cell 23] ### 1.6. Store coverage của test.csv 0 bao phủ hết 1.115 cửa hàng

# %% [code cell 24]
n_train_stores = train.Store.nunique()
n_test_stores  = test.Store.nunique()
missing_from_test = sorted(set(train.Store) - set(test.Store))
print(f"Số CH trong train: {n_train_stores}")
print(f"Số CH trong test : {n_test_stores}")
print(f"Số CH có trong train nhưng KHÔNG có trong test: {len(missing_from_test)}")
print(f"Số CH trong test nhưng không có trong train: {len(set(test.Store)-set(train.Store))}")
print(f"Vài mã CH bị thiếu trong test: {missing_from_test[:10]} ...")


# %% [markdown cell 25] **Phát hiện quan trọng, dễ bị bỏ sót:** `test.csv` chỉ có **856/1.115 CH**, thiếu **259 cửa hàng**.
# %% [markdown cell 26] ### 1.7. Đóng cửa dài ngày (khả năng do cải tạo/refurbishment)

# %% [code cell 27]
ts = train.sort_values(['Store','Date']).copy()
ts['closed'] = (ts.Open==0).astype(int)
ts['grp'] = (ts['closed'] != ts.groupby('Store')['closed'].shift()).cumsum()
runs = ts[ts.closed==1].groupby(['Store','grp']).size()

print("Phân phối độ dài các chuỗi ngày đóng cửa liên tục:")
print(runs.describe())
long_runs = runs[runs>=10]
n_stores_long = long_runs.reset_index()['Store'].nunique()
print(f"\nSố lần đóng cửa liên tục >=10 ngày: {(runs>=10).sum()}")
print(f"Số CH có ít nhất 1 lần đóng cửa dài (>=10 ngày): {n_stores_long}")
print(f"Chuỗi đóng cửa dài nhất: {runs.max()} ngày liên tục")


# %% [markdown cell 28] **Phát hiện quan trọng:** có **72 cửa hàng** từng đóng cửa liên tục ≥10 ngày, cá biệt có chuỗi lên t
# %% [markdown cell 29] ### 1.8. CompetitionOpenSinceYear

# %% [code cell 30]
print(store['CompetitionOpenSinceYear'].describe())
print()
print("Các dòng có năm bất thường (<1990):")
print(store[store['CompetitionOpenSinceYear']<1990][['Store','CompetitionOpenSinceYear','CompetitionOpenSinceMonth']])


# %% [markdown cell 31] **Nhận xét:** phần lớn `CompetitionOpenSinceYear` nằm trong khoảng hợp lý (1990-2015), nhưng có vài 
# %% [markdown cell 32] ### 1.9. Duplicates

# %% [code cell 33]
# Kiểm tra khóa chính (Store, Date)
print(f"Duplicate (Store, Date) trong train: {train.duplicated(subset=['Store','Date']).sum()}")
print(f"Duplicate (Store, Date) trong test : {test.duplicated(subset=['Store','Date']).sum()}")
print("=> (Store, Date) là khóa chính duy nhất, mỗi dòng = đúng 1 CH x 1 ngày.")


# %% [markdown cell 34] **Xác nhận:** Không có dòng nào trùng lặp. Cặp `(Store, Date)` là khóa chính xác định duy nhất, 
# %% [markdown cell 35] ### 1.10. Tính liên tục thời gian

# %% [code cell 36]
# Kiểm tra xem mỗi CH có đủ 942 ngày (01/01/2013 -> 31/07/2015) không
days_per_store = train.groupby('Store')['Date'].count()
n_full = (days_per_store == 942).sum()
n_short = (days_per_store < 942).sum()

print(f"Giai đoạn train (01/01/2013 -> 31/07/2015) = 942 ngày.")
print(f"Số CH có đủ 942 ngày: {n_full}")
print(f"Số CH thiếu ngày    : {n_short} (min = {days_per_store.min()} ngày)")
print(f"\n10 CH có ít ngày nhất:")
print(days_per_store[days_per_store < 942].sort_values().head(10))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(days_per_store.values, bins=50, color='#4472C4', edgecolor='white')
ax.axvline(942, color='red', linestyle='--', linewidth=1.5, label=f'Full = 942 days ({n_full} CH)')
ax.axvline(days_per_store.min(), color='orange', linestyle='--', linewidth=1.5, label=f'Min = {days_per_store.min()} days ({n_short} CH thiếu)')
ax.set_xlabel('Số ngày có dữ liệu trong train')
ax.set_ylabel('Số cửa hàng')
ax.set_title(f'Time Continuity: {n_short}/1.115 CH thiếu ngày')
ax.legend()
plt.tight_layout(); plt.show()


# %% [markdown cell 37] **Phát hiện:** **181 CH chỉ có 758/942 ngày**: các CH này bắt đầu có dữ liệu muộn hơn (khoảng giữa n
# %% [markdown cell 38] ### 1.11. Promo (ngắn hạn) chỉ chạy Thứ 2 đến Thứ 6

# %% [code cell 39]
# Kiểm tra Promo theo DayOfWeek
print(pd.crosstab(train['DayOfWeek'], train['Promo']))

# %% [markdown cell 40] ==> `Promo` **chỉ chạy các ngày trong tuần (Thứ 2 đến Thứ 6)**, 
# %% [markdown cell 41] ### 1.12. missing Open thuộc duy nhất Store 622

# %% [code cell 42]
# Xác định chính xác 11 dòng missing Open trong test.csv
missing_open = test[test['Open'].isna()]
print(missing_open[['Id','Store','DayOfWeek','Date']])

s622 = train[train.Store == 622]
print(f"\nLịch sử Open của Store 622 theo DayOfWeek:")
print(s622.groupby('DayOfWeek')['Open'].mean())

# %% [markdown cell 43] **Phát hiện cụ thể:** Cả 11 dòng missing `Open` **đều thuộc duy nhất `Store 622`**, rơi vào các ngày
# %% [markdown cell 44] ### 1.13. Đối thủ mới mở cửa trong giai đoạn nghiên cứu (2013-2015)

# %% [code cell 45]
# Kiểm tra bao nhiêu CH có đối thủ mới mở cửa TRONG giai đoạn 2013-2015
comp = store.dropna(subset=['CompetitionOpenSinceYear','CompetitionOpenSinceMonth']).copy()
comp['CompOpenDate'] = pd.to_datetime(
    comp['CompetitionOpenSinceYear'].astype(int).astype(str) + '-' + 
    comp['CompetitionOpenSinceMonth'].astype(int).astype(str) + '-01'
)
new_comp = comp[(comp['CompOpenDate'] >= '2013-01-01') & (comp['CompOpenDate'] <= '2015-09-30')]
print(f"Số CH có đối thủ MỚI mở cửa trong giai đoạn 2013-2015: {len(new_comp)} ({len(new_comp)/len(store)*100:.1f}%)")
print(f"\nPhân bố theo năm:")
print(new_comp['CompetitionOpenSinceYear'].value_counts().sort_index())


# %% [markdown cell 46] **Phát hiện:** **191 cửa hàng (17.1%)** có đối thủ cạnh tranh **mới mở cửa ngay trong giai đoạn dữ l
# %% [markdown cell 47] ### 1.14. Phân phối Sales lệch phải -> ảnh hưởng RMSPE

# %% [code cell 48]
# Phân tích phân phối Sales và hiệu quả của log-transform
# Lưu ý: Lọc Open=1 & Sales>0 (loại 54 dòng ngoại lai Sales=0 đã phân tích ở mục 1.3)
open_sales_all = train[train.Open == 1]['Sales']
open_sales = train[(train.Open == 1) & (train.Sales > 0)]['Sales']

log_sales_all = np.log1p(open_sales_all)
log_sales = np.log1p(open_sales)

print(f"1. Sales gốc (Open=1 & Sales>0): mean={open_sales.mean():.0f}, median={open_sales.median():.0f}, skew={open_sales.skew():.2f}")
print(f"   - Max Sales: {train['Sales'].max():,}, Min Sales (Sales>0): {open_sales.min():,}")
print(f"\nPercentiles Sales (Open=1 & Sales>0):")
print(open_sales.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

print(f"\n2. Hiệu quả sau log-transform (log(Sales + 1)):")
print(f"   - Nếu tính cả 54 dòng Sales=0: skew = {log_sales_all.skew():.2f} (bị 54 điểm log(1)=0 kéo lệch trái)")
print(f"   - Khi loại bỏ 54 dòng Sales=0 (chuẩn): skew giảm từ {open_sales.skew():.2f} xuống chỉ còn {log_sales.skew():.2f} (gần như đối xứng hoàn hảo!)")

# Trực quan hóa so sánh
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

axes[0].hist(open_sales, bins=80, color='#4472C4', edgecolor='white', alpha=0.8)
axes[0].axvline(open_sales.mean(), color='red', linestyle='--', label=f'Mean = {open_sales.mean():.0f}')
axes[0].axvline(open_sales.median(), color='orange', linestyle='--', label=f'Median = {open_sales.median():.0f}')
axes[0].set_title(f'Phân phối Sales (Open=1 & Sales>0) | Skew = {open_sales.skew():.2f}')
axes[0].set_xlabel('Sales (€)'); axes[0].set_ylabel('Tần suất')
axes[0].legend()

axes[1].hist(log_sales, bins=80, color='#70AD47', edgecolor='white', alpha=0.8)
axes[1].axvline(log_sales.mean(), color='red', linestyle='--', label=f'Mean = {log_sales.mean():.2f}')
axes[1].axvline(log_sales.median(), color='orange', linestyle='--', label=f'Median = {log_sales.median():.2f}')
axes[1].set_title(f'Phân phối log(Sales+1) | Skew = {log_sales.skew():.2f}')
axes[1].set_xlabel('log(Sales + 1)'); axes[1].set_ylabel('Tần suất')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown cell 49] **Phát hiện bản chất phân phối và giải thích chi tiết về Skewness:**
# %% [markdown cell 50] ### 1.15. 225 CH có dữ liệu trước VÀ sau khi tham gia Promo2 trong train

# %% [code cell 51]
# Xác định bao nhiêu CH tham gia Promo2 TRONG giai đoạn train (có dữ liệu before & after)
p2 = store[store.Promo2 == 1].copy()
p2['Promo2StartDate'] = pd.to_datetime(
    p2['Promo2SinceYear'].astype(int).astype(str) + '-W' + 
    p2['Promo2SinceWeek'].astype(int).astype(str).str.zfill(2) + '-1', 
    format='%Y-W%W-%w'
)
before = (p2['Promo2StartDate'] < '2013-01-01').sum()
during = ((p2['Promo2StartDate'] >= '2013-01-01') & (p2['Promo2StartDate'] <= '2015-07-31')).sum()

print(f"- Tham gia TRƯỚC 01/01/2013 (chỉ có 'after' trong train): {before} CH")
print(f"- Tham gia TRONG giai đoạn train 2013-2015 (có cả before & after): {during} CH")
print(f"\n=> {during} CH có dữ liệu trước VÀ sau khi gia nhập Promo2 ngay trong train.csv")
print(f"-> Nhóm quý giá nhất để phân tích nhân quả (before-after / DiD) cho Mission 1.")


# %% [markdown cell 52] **Phát hiện quan trọng cho Mission 1:** Trong 571 CH tham gia Promo2:
# %% [markdown cell 53] ---

# %% [code cell 54]
df = train[train.Open==1].merge(store, on='Store', how='left')
print("Shape sau khi lọc Open=1 và merge store:", df.shape)


# %% [markdown cell 55] ### 2.1. Promo (khuyến mãi ngắn hạn) vs Sales

# %% [code cell 56]
promo_cmp = df.groupby('Promo')['Sales'].agg(['mean','count'])
pct = (promo_cmp.loc[1,'mean'] / promo_cmp.loc[0,'mean'] - 1) * 100

fig, ax = plt.subplots(figsize=(5,4))
bars = ax.bar(['Không Promo','Có Promo'], promo_cmp['mean'], color=['#A6A6A6','#4472C4'])
for b, n in zip(bars, promo_cmp['count']):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+80, f"n={n:,}", ha='center')
ax.set_ylabel('Sales trung bình / ngày')
ax.set_title(f'Promo ngắn hạn: +{pct:.1f}% doanh số')
plt.tight_layout(); plt.show()
print(promo_cmp); print(f"Chênh lệch: {pct:.1f}%")


# %% [markdown cell 57] **Nhận xét:** Ngày có Promo ngắn hạn, Sales TB cao hơn rõ rệt (~+38%), mẫu rất lớn ở cả 2 nhóm → tín
# %% [markdown cell 58] ### 2.2. CompetitionDistance vs Sales

# %% [code cell 59]
d = df.dropna(subset=['CompetitionDistance']).copy()
corr = d['CompetitionDistance'].corr(d['Sales'])
print(f"Tương quan CompetitionDistance vs Sales: {corr:.4f}")

bins = [0, 500, 1000, 2500, 5000, 10000, 1e9]
labels = ['<500m','500-1000m','1-2.5km','2.5-5km','5-10km','>10km']
d['dist_bin'] = pd.cut(d['CompetitionDistance'], bins=bins, labels=labels)
g = d.groupby('dist_bin')['Sales'].agg(['mean','count'])

fig, ax = plt.subplots(figsize=(7,4))
ax.bar(g.index.astype(str), g['mean'], color='#70AD47')
for i,(m,n) in enumerate(zip(g['mean'], g['count'])):
    ax.text(i, m+80, f"n={n:,}", ha='center', fontsize=8)
ax.set_ylabel('Sales trung bình / ngày'); ax.set_title('Sales TB theo khoảng cách đối thủ')
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
print(g)


# %% [markdown cell 60] **Giả thuyết bị bác bỏ:** tương quan chỉ ≈ -0.036, không có xu hướng đơn điệu rõ ràng theo nhóm khoả
# %% [markdown cell 61] ### 2.3. StoreType & Assortment vs Sales

# %% [code cell 62]
g_st = df.groupby('StoreType')['Sales'].mean(); n_st = store.groupby('StoreType').size()
g_as = df.groupby('Assortment')['Sales'].mean(); n_as = store.groupby('Assortment').size()

fig, axes = plt.subplots(1,2, figsize=(11,4))
bars = axes[0].bar(g_st.index, g_st.values, color='#4472C4')
for b, n in zip(bars, n_st[g_st.index]): axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+80, f"n={n}", ha='center')
axes[0].set_title('Sales TB theo StoreType')

bars2 = axes[1].bar(g_as.index, g_as.values, color='#ED7D31')
for b, n in zip(bars2, n_as[g_as.index]): axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+80, f"n={n}", ha='center')
axes[1].set_title('Sales TB theo Assortment')
plt.tight_layout(); plt.show()


# %% [markdown cell 63] **Nhận xét:** StoreType `b` (n=17) và Assortment `b` (n=9) cao vượt trội, rất có thể là hiệu ứng nhó

# %% [code cell 64]
# Đào sâu bản chất StoreType b: Khách hàng, Giỏ hàng (Basket size) và Lịch mở cửa Chủ nhật
df['BasketSize'] = df['Sales'] / df['Customers']

# Tỷ lệ mở cửa ngày Chủ nhật (tính trên tập train gốc gồm cả Open=0)
train_st = train[train.DayOfWeek == 7].merge(store[['Store', 'StoreType']], on='Store')
sun_open_rate = train_st.groupby('StoreType')['Open'].mean() * 100

st_summary = pd.DataFrame({
    'Số CH (n)': store.groupby('StoreType').size(),
    'Tỷ trọng CH (%)': (store.groupby('StoreType').size() / len(store) * 100).round(1),
    'Tỷ lệ mở CN (%)': sun_open_rate.round(1),
    'Khách TB / ngày': df.groupby('StoreType')['Customers'].mean().round(0).astype(int),
    'Giỏ hàng TB (€)': df.groupby('StoreType')['BasketSize'].mean().round(2),
    'Sales TB / ngày (€)': df.groupby('StoreType')['Sales'].mean().round(0).astype(int)
})

print("=== BẢNG ĐẶC ĐIỂM STORETYPE ===")
print(st_summary)
print("\n* Lưu ý: Toàn bộ 9 cửa hàng Assortment 'b' (extra) đều thuộc StoreType 'b'.")

# Trực quan hóa so sánh 3 khía cạnh: Lượng khách, Giỏ hàng và Tỷ lệ mở Chủ nhật
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

bars1 = axes[0].bar(st_summary.index, st_summary['Khách TB / ngày'], color=['#4472C4', '#ED7D31', '#4472C4', '#4472C4'])
for b in bars1: axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+35, f"{int(b.get_height()):,}", ha='center', fontsize=9)
axes[0].set_title('Khách TB / ngày (khi Open=1)')
axes[0].set_ylabel('Số khách')

bars2 = axes[1].bar(st_summary.index, st_summary['Giỏ hàng TB (€)'], color=['#4472C4', '#ED7D31', '#4472C4', '#4472C4'])
for b in bars2: axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{b.get_height():.2f}€", ha='center', fontsize=9)
axes[1].set_title('Giá trị giỏ hàng TB (Sales/Customers)')
axes[1].set_ylabel('EUR (€)')

bars3 = axes[2].bar(st_summary.index, st_summary['Tỷ lệ mở CN (%)'], color=['#4472C4', '#ED7D31', '#4472C4', '#4472C4'])
for b in bars3: axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+2, f"{b.get_height():.1f}%", ha='center', fontsize=9)
axes[2].set_title('Tỷ lệ mở cửa ngày Chủ nhật (%)')
axes[2].set_ylabel('%')

plt.tight_layout()
plt.show()


# %% [markdown cell 65] **Đào sâu bản chất: Liệu StoreType `b` / Assortment `b` có phải là phân khúc 'Premium' (cao cấp)?**
# %% [markdown cell 66] ### 2.4. DayOfWeek

# %% [code cell 67]
open_rate = train.groupby('DayOfWeek')['Open'].mean()
sales_by_dow = df.groupby('DayOfWeek')['Sales'].mean()

fig, axes = plt.subplots(1,2, figsize=(11,4))
axes[0].bar(open_rate.index, open_rate.values, color='#4472C4')
axes[0].set_title('Tỷ lệ mở cửa theo DayOfWeek (1=Mon...7=Sun)')
axes[0].set_ylabel('Tỷ lệ Open')

axes[1].bar(sales_by_dow.index, sales_by_dow.values, color='#ED7D31')
axes[1].set_title('Sales TB theo DayOfWeek (khi Open=1)')
plt.tight_layout(); plt.show()
print("Tỷ lệ Open:\n", open_rate.round(3))
print("\nSales TB (Open=1):\n", sales_by_dow)


# %% [markdown cell 68] **Phát hiện:** Chủ nhật (DayOfWeek=7) chỉ mở cửa **2.5%** số lần, phần lớn cửa hàng đóng cửa Chủ nhậ
# %% [markdown cell 69] ### 2.5. SchoolHoliday vs Sales

# %% [code cell 70]
g = df.groupby('SchoolHoliday')['Sales'].agg(['mean','count'])
print(g)
pct = (g.loc[1,'mean']/g.loc[0,'mean']-1)*100
print(f"Chênh lệch: {pct:+.1f}%")


# %% [markdown cell 71] **Nhận xét:** Sales TB ngày có SchoolHoliday cao hơn nhẹ (~+4.4%), mẫu đủ lớn ở cả 2 nhóm. Tín hiệu 
# %% [markdown cell 72] ### 2.6. StateHoliday vs Sales, chỉ khi cửa hàng thực sự mở cửa (rất hiếm)

# %% [code cell 73]
print("Tỷ lệ Open theo StateHoliday:")
print(train.groupby('StateHoliday')['Open'].mean())
print()
g = df.groupby('StateHoliday')['Sales'].agg(['mean','count'])
print("Sales TB khi Open=1, theo loại StateHoliday:")
print(g)


# %% [markdown cell 74] **Nhận xét:** Cửa hàng gần như luôn đóng cửa vào ngày lễ (tỷ lệ Open chỉ 1.7-3.4% cho loại a/b/c). N

# %% [code cell 75]
# Đào sâu bản chất ngày lễ: Bóc tách doanh số theo từng StoreType để kiểm tra giả thuyết
df['IsHoliday'] = df['StateHoliday'] != '0'

st_comp = df.groupby(['StoreType', 'IsHoliday'])['Sales'].agg(['count', 'mean']).unstack()
st_comp.columns = ['Count_Normal', 'Count_Holiday', 'Sales_Normal', 'Sales_Holiday']
st_comp['Holiday_Pct_of_Total_Open'] = (st_comp['Count_Holiday'] / st_comp['Count_Holiday'].sum() * 100).round(1)
st_comp['Sales_Diff_%'] = ((st_comp['Sales_Holiday'] - st_comp['Sales_Normal']) / st_comp['Sales_Normal'] * 100).round(1)

summary_display = pd.DataFrame({
    'Số lượt mở ngày lễ': st_comp['Count_Holiday'],
    'Tỷ trọng trong ngày lễ (%)': st_comp['Holiday_Pct_of_Total_Open'],
    'Sales TB Ngày thường (€)': st_comp['Sales_Normal'].round(0).astype(int),
    'Sales TB Ngày lễ (€)': st_comp['Sales_Holiday'].round(0).astype(int),
    'Chênh lệch (%)': st_comp['Sales_Diff_%']
})

print("=== BẢNG BÓC TÁCH DOANH SỐ NGÀY LỄ VS NGÀY THƯỜNG THEO STORETYPE ===")
print(summary_display)

# Trực quan hóa Nghịch lý Simpson trong ngày lễ
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
x = np.arange(len(st_comp.index))
width = 0.35

bars1 = axes[0].bar(x - width/2, st_comp['Sales_Normal'], width, label='Ngày thường (StateHoliday=0)', color='#4472C4')
bars2 = axes[0].bar(x + width/2, st_comp['Sales_Holiday'], width, label='Ngày lễ (StateHoliday a/b/c)', color='#ED7D31')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"StoreType {st}" for st in st_comp.index])
axes[0].set_ylabel('Sales TB / ngày (€)')
axes[0].set_title("Nghịch lý Simpson: Chỉ StoreType 'b' tăng, 'a' & 'd' giảm mạnh ngày lễ")
axes[0].legend()
for b in bars1: axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+120, f"{int(b.get_height()):,}", ha='center', fontsize=8.5)
for b in bars2: axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+120, f"{int(b.get_height()):,}", ha='center', fontsize=8.5, fontweight='bold')

bars3 = axes[1].bar(st_comp.index, st_comp['Holiday_Pct_of_Total_Open'], color=['#A6A6A6', '#C00000', '#A6A6A6', '#A6A6A6'])
axes[1].set_title("Cơ cấu mở cửa ngày lễ: StoreType 'b' chiếm gần 50% số lượt")
axes[1].set_ylabel('% trong tổng số lượt mở ngày lễ')
axes[1].set_xlabel('StoreType')
for b in bars3: axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.1f}%", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()


# %% [markdown cell 76] **Đào sâu chứng minh bản chất: Bóc trần Nghịch lý Simpson (Simpson's Paradox) trong ngày lễ:**
# %% [markdown cell 77] ### 2.7. Customers vs Sales

# %% [code cell 78]
corr_cs = df['Sales'].corr(df['Customers'])
print(f"Tương quan Sales vs Customers: {corr_cs:.3f}")

d = df.copy()
d['basket'] = d['Sales'] / d['Customers'].replace(0, np.nan)
print("\nBasket size (Sales/Customers) — thống kê mô tả:")
print(d['basket'].describe())

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(d['basket'].dropna(), bins=60, range=(0,25), color='#4472C4', edgecolor='white')
ax.set_xlabel('Basket size (Sales / Customers), đơn vị tiền/khách')
ax.set_ylabel('Số dòng (ngày-cửa hàng)')
ax.set_title('Phân phối basket size toàn hệ thống')
plt.tight_layout(); plt.show()


# %% [markdown cell 79] **Nhận xét:** Sales và Customers tương quan rất mạnh (r≈0.82). Dễ hiểu vì khách càng đông thì doanh 
# %% [markdown cell 80] ---
# %% [markdown cell 81] ### 3.1. Tái hiện con số của Giám đốc Vận hành (so sánh cross-sectional thô)

# %% [code cell 82]
store_level = df.groupby(['Store','Promo2'])['Sales'].mean().reset_index()
naive = store_level.groupby('Promo2')['Sales'].agg(['mean','count'])
naive.index = ['Promo2 = 0','Promo2 = 1']
naive.columns = ['Sales TB / ngày (theo CH)','Số CH (n)']
gap_pct = (naive.iloc[1,0]/naive.iloc[0,0]-1)*100

fig, ax = plt.subplots(figsize=(5,4))
bars = ax.bar(naive.index, naive['Sales TB / ngày (theo CH)'], color=['#A6A6A6','#C00000'])
for b, n in zip(bars, naive['Số CH (n)']):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+80, f"n={n}", ha='center')
ax.set_ylabel('Sales TB/ngày (trung bình theo CH)')
ax.set_title(f'So sánh thô: Promo2=1 thấp hơn {abs(gap_pct):.1f}%')
plt.tight_layout(); plt.show()
print(naive); print(f"\nChênh lệch: {gap_pct:.1f}%")


# %% [markdown cell 83] **Đã tái hiện đúng con số của HĐQT:** chênh lệch khoảng **10-11%**. Đây là **symptom**. Câu hỏi thật
# %% [markdown cell 84] ### 3.2. Hypothesis 1 - Selection bias: 2 nhóm CH vốn dĩ khác nhau về đặc điểm nền?

# %% [code cell 85]
ct_st = pd.crosstab(store.Promo2, store.StoreType, normalize='index')*100
ct_as = pd.crosstab(store.Promo2, store.Assortment, normalize='index')*100
comp_dist = store.groupby('Promo2')['CompetitionDistance'].mean()

fig, axes = plt.subplots(1,3, figsize=(14,4))
ct_st.T.plot(kind='bar', ax=axes[0], color=['#A6A6A6','#C00000'])
axes[0].set_title('% StoreType theo nhóm Promo2'); axes[0].legend(['Promo2=0','Promo2=1'])
ct_as.T.plot(kind='bar', ax=axes[1], color=['#A6A6A6','#C00000'])
axes[1].set_title('% Assortment theo nhóm Promo2'); axes[1].legend(['Promo2=0','Promo2=1'])
axes[2].bar(['Promo2=0','Promo2=1'], comp_dist.values, color=['#A6A6A6','#C00000'])
axes[2].set_title('CompetitionDistance TB theo nhóm Promo2')
plt.tight_layout(); plt.show()

print("StoreType %:\n", ct_st.round(1))
print("\nAssortment %:\n", ct_as.round(1))
print("\nCompetitionDistance TB:\n", comp_dist)


# %% [markdown cell 86] **Nhận xét:**
# %% [markdown cell 87] ### 3.3. Xây dựng biến `IsPromo2ActiveThisMonth` (bước chuẩn bị quan trọng nhất)

# %% [code cell 88]
month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,
             'Sep':9,'Sept':9,'Oct':10,'Nov':11,'Dec':12}

def parse_interval(s):
    if pd.isna(s): return set()
    return set(month_map[m.strip()] for m in s.split(','))

d2 = df[df.Promo2==1].copy()
d2['interval_months'] = d2['PromoInterval'].map(parse_interval)
d2['Year']  = d2.Date.dt.isocalendar().year
d2['Week']  = d2.Date.dt.isocalendar().week
d2['Month'] = d2.Date.dt.month

def past_since(row):
    y, w = row['Year'], row['Week']
    sy, sw = row['Promo2SinceYear'], row['Promo2SinceWeek']
    if pd.isna(sy) or pd.isna(sw): return False
    return (y > sy) or (y == sy and w >= sw)

d2['past_since']         = d2.apply(past_since, axis=1)
d2['month_in_interval']  = d2.apply(lambda r: r['Month'] in r['interval_months'], axis=1)
d2['IsPromo2ActiveThisMonth'] = (d2['past_since'] & d2['month_in_interval']).astype(int)

print("Điều kiện: Promo2=1 AND đã qua mốc SinceWeek/Year AND tháng nằm trong PromoInterval")
print(d2['IsPromo2ActiveThisMonth'].value_counts())


# %% [markdown cell 89] **Test case thủ công**

# %% [code cell 90]
row = store[store.Store==6].iloc[0]
print(f"Store 6: Promo2={row.Promo2}, SinceWeek={row.Promo2SinceWeek}, "
      f"SinceYear={row.Promo2SinceYear}, PromoInterval={row.PromoInterval}")
print()
sample = d2[d2.Store==6][['Date','Year','Week','Month','past_since','month_in_interval','IsPromo2ActiveThisMonth']]
print(sample[sample.Month.isin([1,2,3,4])].drop_duplicates('Month').sort_values('Date'))


# %% [markdown cell 91] **Nhận xét:** logic hoạt động đúng. `month_in_interval` chỉ True ở đúng các tháng trong PromoInterva
# %% [markdown cell 92] ### 3.4. So sánh within-store: tháng active vs tháng nghỉ (chưa xét mùa vụ)

# %% [code cell 93]
post = d2[d2['past_since']].copy()

# cảnh báo: loại các store-month có quá ít ngày mở cửa (khả năng do đóng cửa dài — mục 1.7)
days_per_sm = post.groupby(['Store','Year','Month']).size()
thin = days_per_sm[days_per_sm<5]
print(f"Số tháng-CH có <5 ngày mở cửa (khả năng do đóng cửa dài hạn): {len(thin)}/{len(days_per_sm)}")
post = post.set_index(['Store','Year','Month'])
post = post[~post.index.isin(thin.index)].reset_index()
print(f"Giữ lại {len(days_per_sm)-len(thin)} tháng-CH sau khi loại các tháng quá thưa dữ liệu.")

grp = post.groupby(['Store','IsPromo2ActiveThisMonth'])['Sales'].mean().unstack().dropna()
grp.columns = ['inactive_month_sales','active_month_sales']
grp['pct_diff'] = (grp['active_month_sales'] - grp['inactive_month_sales']) / grp['inactive_month_sales'] * 100

fig, ax = plt.subplots(figsize=(7,4))
ax.hist(grp['pct_diff'], bins=40, color='#4472C4', edgecolor='white')
ax.axvline(0, color='black', linewidth=1)
ax.axvline(grp['pct_diff'].mean(), color='red', linestyle='--', label=f"TB = {grp['pct_diff'].mean():.2f}%")
ax.set_xlabel('% chênh lệch Sales (active - nghỉ), theo từng CH')
ax.set_ylabel('Số cửa hàng')
ax.set_title(f"n={len(grp)} CH, {(grp['pct_diff']>0).mean()*100:.1f}% CH có active > nghỉ")
ax.legend(); plt.tight_layout(); plt.show()
print(grp['pct_diff'].describe())


# %% [markdown cell 94] **Nhận xét:** Ngay cả trong cùng 1 CH, tháng active vẫn thấp hơn tháng nghỉ trung bình **~1.7%**, ch
# %% [markdown cell 95] ### 3.5. Mùa vụ thị trường

# %% [code cell 96]
seasonal = df.groupby(df.Date.dt.month)['Sales'].mean()
overall_mean = df['Sales'].mean()
seasonal_idx = seasonal / overall_mean

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(seasonal_idx.index, seasonal_idx.values, marker='o', color='#4472C4')
ax.axhline(1.0, color='gray', linestyle=':')
ax.set_xticks(range(1,13)); ax.set_xlabel('Tháng'); ax.set_ylabel('Chỉ số mùa vụ (=1 là trung bình năm)')
ax.set_title('Chỉ số mùa vụ theo tháng - toàn bộ cửa hàng')
plt.tight_layout(); plt.show()
print(seasonal_idx.round(3))


# %% [markdown cell 97] **Nhận xét:** Doanh số có mùa vụ rõ rệt: tháng 12 cao vượt trội (chỉ số 1.24), tháng 1-2 và 8-10 thấ
# %% [markdown cell 98] ### 3.6. Kiểm chứng thuần mô tả: nếu chỉ do lịch (mùa vụ), mức "âm" kỳ vọng là bao nhiêu?

# %% [code cell 99]
results = []
for pattern in store[store.Promo2==1]['PromoInterval'].dropna().unique():
    active_m = parse_interval(pattern)
    inactive_m = set(range(1,13)) - active_m
    active_idx = seasonal_idx[sorted(active_m)].mean()
    inactive_idx = seasonal_idx[sorted(inactive_m)].mean()
    n_stores_pattern = (store[store.Promo2==1]['PromoInterval']==pattern).sum()
    expected_gap_pct = (active_idx/inactive_idx - 1) * 100
    results.append((pattern, n_stores_pattern, active_idx, inactive_idx, expected_gap_pct))

res_df = pd.DataFrame(results, columns=['PromoInterval pattern','n CH','Seasonal idx (active)','Seasonal idx (nghỉ)','Expected gap % (chỉ do lịch)'])
print(res_df.round(3))

# weighted average theo đúng tỷ trọng n CH mỗi pattern -> con số dự đoán tổng thể
weighted_expected = (res_df['n CH'] * res_df['Expected gap % (chỉ do lịch)']).sum() / res_df['n CH'].sum()
actual_observed = grp['pct_diff'].mean()
print(f"\n>>> Gap dự đoán CHỈ từ trùng hợp lịch (mùa vụ), gộp theo tỷ trọng 3 pattern: {weighted_expected:+.2f}%")
print(f">>> Gap thực tế quan sát được ở mục 3.4 (within-store, active vs nghỉ)      : {actual_observed:+.2f}%")



# %% [code cell 100]
fig, ax = plt.subplots(figsize=(6,4))
names = ['Dự đoán chỉ từ\nmùa vụ (lịch)', 'Quan sát thực tế\n(within-store)']
vals = [weighted_expected, actual_observed]
bars = ax.bar(names, vals, color=['#70AD47','#C00000'])
ax.axhline(0, color='black', linewidth=0.8)
for b,v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v-0.3 if v<0 else v+0.1, f"{v:+.2f}%", ha='center')
ax.set_ylabel('% chênh lệch (active vs nghỉ)')
ax.set_title('Mùa vụ đơn thuần đã dự đoán gần đúng gap quan sát được')
plt.tight_layout(); plt.show()


# %% [markdown cell 101] **Đây là phát hiện quan trọng nhất của toàn bộ EDA:**
# %% [markdown cell 102] ---

```

---

## scripts/03_F01_Promo2_Analysis.py

```python
# Notebook export: 03_F01_Promo2_Analysis
# Original source: https://colab.research.google.com/drive/19K7REYOXk6DfUlXBEf50IWaSfo_a5GK6?hl=vi
# Automatically extracted code cells only. The .ipynb is authoritative.
# Notebook magics/shell commands, if any, are commented out.

# %% [markdown cell 1] ` ` `

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
# %% [markdown cell 121] ` ` `
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


```

---

## scripts/04_F02_Forecasting_xAI.py

```python
# Notebook export: 04_F02_Forecasting_xAI
# Original source: https://drive.google.com/file/d/1wA5pKE3uRvLpU2EFhmN4yBn657X1IoI6/view
# Automatically extracted code cells only. The .ipynb is authoritative.
# Notebook magics/shell commands, if any, are commented out.


# %% [code cell 1]
# JUPYTER_ONLY: %matplotlib inline
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': 0.3})
pd.set_option('display.max_columns', 60); pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 110)

# Notebook tự tìm file ở nhiều chỗ: thư mục hiện tại, và đường dẫn Kaggle.
CANDIDATES = ['.', '/mnt/user-data/uploads', '/kaggle/input/competitions/rossmann-store-sales',
              '/kaggle/input/rossmann-store-sales', '../input/rossmann-store-sales']
def find(*names):
    for d in CANDIDATES:
        for n in names:
            p = Path(d) / n
            if p.exists(): return p
    raise FileNotFoundError(f'Không tìm thấy {names} trong {CANDIDATES}')

OUT_DIR = Path('.')      # nơi lưu các file output F02_*.csv (đổi thành /kaggle/working nếu chạy trên Kaggle)

# %% [markdown cell 2] ## 0. Load dữ liệu

# %% [code cell 3]
train = pd.read_csv(find('train_ross_1.csv', 'train_ross.csv', 'train.csv'), parse_dates=['Date'], dtype={'StateHoliday': str})
test  = pd.read_csv(find('test_ross_1.csv',  'test_ross.csv',  'test.csv'),  parse_dates=['Date'], dtype={'StateHoliday': str})
store = pd.read_csv(find('store_ross_1.csv', 'store_ross.csv', 'store.csv'))
for df in (train, test):
    df['StateHoliday'] = df['StateHoliday'].astype(str).replace('0.0', '0')

print('train', train.shape, '| test', test.shape, '| store', store.shape)

# %% [markdown cell 4] ---

# %% [code cell 5]
TRAIN_START, TRAIN_END = train.Date.min(), train.Date.max()
TEST_START,  TEST_END  = test.Date.min(),  test.Date.max()
HORIZON = (TEST_END - TEST_START).days + 1          # số ngày phải dự báo
GAP     = (TEST_START - TRAIN_END).days - 1         # số ngày "trống" giữa train và test
test_stores = np.sort(test.Store.unique())

print(f'Train: {TRAIN_START.date()} -> {TRAIN_END.date()} ({(TRAIN_END-TRAIN_START).days+1} ngày)')
print(f'Test : {TEST_START.date()} -> {TEST_END.date()}  => HORIZON = {HORIZON} ngày ({HORIZON/7:.1f} tuần)')
print(f'GAP giữa cuối train và đầu test = {GAP} ngày (test bắt đầu NGAY SAU train)')
print(f'Số cửa hàng trong test: {len(test_stores)} / {store.Store.nunique()} (thiếu {store.Store.nunique()-len(test_stores)} CH)')
print(f'Số dòng test: {len(test):,} | Open=1: {(test.Open==1).sum():,} | Open=0: {(test.Open==0).sum():,} | Open thiếu: {test.Open.isna().sum()}')

# %% [markdown cell 6] **Kết luận mục 1 (nền tảng cho các luật sau):**
# %% [markdown cell 7] ---

# %% [code cell 8]
tr_cols, te_cols, st_cols = set(train.columns), set(test.columns), set(store.columns)
rows = []
for c in sorted(tr_cols | te_cols | st_cols):
    in_tr, in_te, in_st = c in tr_cols, c in te_cols, c in st_cols
    if c == 'Id':                       kind = 'Chỉ là mã dòng để nộp bài, KHÔNG phải feature'
    elif c in ('Sales', 'Customers'):   kind = 'CHỈ BIẾT QUÁ KHỨ (không có trong test) -> không dùng trực tiếp'
    elif in_te or in_st:                kind = 'BIẾT TRƯỚC (có trong test hoặc là thuộc tính cố định của cửa hàng)'
    else:                               kind = '?'
    rows.append((c, 'x' if in_tr else '', 'x' if in_te else '', 'x' if in_st else '', kind))
audit = pd.DataFrame(rows, columns=['Cột', 'train', 'test', 'store', 'Phân loại'])
audit

# %% [markdown cell 9] **Kết luận mục 2 → Luật R1:**
# %% [markdown cell 10] ---

# %% [code cell 11]
# --- (a) train: 2 loại Sales = 0
closed_rows = (train.Open == 0).sum()
open_zero   = ((train.Open == 1) & (train.Sales == 0)).sum()
print(f'train: Open=0 -> {closed_rows:,} dòng | Open=1 nhưng Sales=0 -> {open_zero} dòng')
print('Open=0 mà Sales != 0:', ((train.Open == 0) & (train.Sales != 0)).sum(), '(kỳ vọng 0)')

# --- (b) test: Open=0 rơi vào thứ mấy?
print('\ntest: Open=0 theo DayOfWeek (1=T2 ... 7=CN):')
print(test[test.Open == 0].DayOfWeek.value_counts().sort_index().to_dict())

# --- (c) 11 dòng Open thiếu
miss = test[test.Open.isna()]
print('\n11 dòng Open thiếu thuộc cửa hàng:', miss.Store.unique().tolist(), '| thứ trong tuần:', sorted(miss.DayOfWeek.unique()))
hist622 = train[train.Store == 622].groupby('DayOfWeek').Open.mean().round(3)
print('Tỷ lệ mở cửa lịch sử của Store 622 theo thứ:', hist622.to_dict())

# %% [markdown cell 12] **Nhận xét:** Open=0 trong test chủ yếu là Chủ nhật (5.803/5.984 dòng); còn lại là 165 dòng T7 (một 

# %% [code cell 13]
n_eval = int(((test.Open == 1) | test.Open.isna()).sum())    # số dòng sẽ được chấm (xấp xỉ)
n_bad  = int(test.Open.isna().sum())                        # 11 dòng nếu lỡ gán Open=0
sim = []
for base in (0.10, 0.12, 0.15, 0.22):
    new = np.sqrt(base**2 + n_bad / n_eval)
    sim.append((base, round(new, 4), round(new - base, 4)))
pd.DataFrame(sim, columns=['RMSPE của model (giả định)', 'RMSPE nếu gán nhầm Open=0 cho 11 dòng', 'Điểm bị cộng thêm'])

# %% [markdown cell 14] **Kết luận mục 3 → Luật R2, R3:**
# %% [markdown cell 15] ---

# %% [code cell 16]
print('Promo trung bình theo DayOfWeek')
print(pd.DataFrame({'train': train.groupby('DayOfWeek').Promo.mean().round(3),
                    'test':  test.groupby('DayOfWeek').Promo.mean().round(3)}).T)
print('\nPromo=1 vào T7/CN: train =', train[train.DayOfWeek >= 6].Promo.sum(), '| test =', test[test.DayOfWeek >= 6].Promo.sum())

# Lịch Promo trong test theo tuần (T2-T6)
t = test[test.DayOfWeek <= 5].copy(); t['week_start'] = t.Date - pd.to_timedelta(t.Date.dt.dayofweek, unit='D')
wk = t.groupby('week_start').Promo.agg(['mean', 'nunique'])
wk.columns = ['tỷ lệ cửa hàng có Promo', 'số giá trị Promo khác nhau trong tuần']
print('\nLịch Promo trong test theo tuần:'); print(wk)

# %% [markdown cell 17] **Nhận xét:** Promo **chỉ chạy T2-T6**, không bao giờ vào T7/CN. Trong test, lịch Promo là **luân ph
# %% [markdown cell 18] ---

# %% [code cell 19]
ALL_DATES = pd.date_range(TRAIN_START, TEST_END, name='Date')
STORES = pd.Index(np.sort(store.Store.unique()), name='Store')

def to_mat(df, col):
    return df.pivot(index='Date', columns='Store', values=col).reindex(index=ALL_DATES, columns=STORES)

S      = to_mat(train, 'Sales')
O      = to_mat(train, 'Open')
Promo  = pd.concat([to_mat(train, 'Promo').loc[:TRAIN_END], to_mat(test, 'Promo').loc[TEST_START:]])   # lịch Promo biết trước cả test
valid  = (O == 1) & (S > 0)
logS   = np.log1p(S.where(valid))            # NaN nếu đóng cửa / thiếu ngày / Sales=0
print('Ma trận ngày x cửa hàng:', logS.shape, '| tỷ lệ ô có giá trị:', round(logS.notna().mean().mean(), 3))

# %% [markdown cell 20] ### 5.3 Hàm tạo biến lịch sử (đã tuân thủ R5) + bộ biến đề xuất

# %% [code cell 21]
def first_multiple_of_7_at_least(h):
    return int(np.ceil(h / 7) * 7)

def same_dow_same_promo_mean(logS, Promo, lags):
    # Trung bình log-sales của các ngày CÙNG THỨ (lag bội của 7) và CÙNG trạng thái Promo với ngày cần dự báo.
    num, den = 0, 0
    for k in lags:
        v  = logS.shift(k)
        ok = v.notna() & (Promo.shift(k) == Promo)
        num = num + v.where(ok).fillna(0); den = den + ok.astype(float)
    return num / den.replace(0, np.nan), den

def build_history_features(logS, Promo, horizon=HORIZON):
    first = first_multiple_of_7_at_least(horizon)             # 49 khi horizon = 48
    lags  = list(range(first, first + 7 * 11, 7))             # 49, 56, ..., 119
    assert min(lags) >= horizon, 'Vi phạm luật R5: lag phải >= horizon'
    L = logS.shift(horizon)                                   # mọi cửa sổ rolling kết thúc ở t - horizon
    wk = pd.Series(ALL_DATES.dayofweek < 5, index=ALL_DATES).shift(horizon).fillna(False).values[:, None]
    f = {}
    f['dow_promo_mean'], f['dow_promo_n'] = same_dow_same_promo_mean(logS, Promo, lags)
    f['roll28'] = L.rolling(28, min_periods=7).mean()
    f['roll91'] = L.rolling(91, min_periods=20).mean()
    f['open28'] = logS.notna().astype(float).shift(horizon).rolling(28, min_periods=1).sum()
    mp = L.where(Promo.shift(horizon) == 1).rolling(91, min_periods=5).mean()
    mn = L.where((Promo.shift(horizon) == 0) & wk).rolling(91, min_periods=5).mean()
    f['promo_uplift91'] = mp - mn
    f['ly364'] = logS.shift(364)
    f['ly728'] = logS.shift(728)
    return f

HIST = build_history_features(logS, Promo)
print('Đã tạo', len(HIST), 'biến lịch sử, mỗi biến là ma trận', logS.shape)

# %% [markdown cell 22] ### 5.4 Test chống rò rỉ (quan trọng nhất của F02)

# %% [code cell 23]
def leakage_test(cutoff, features_fn, horizon=HORIZON):
    cutoff = pd.Timestamp(cutoff)
    logS_cut = logS.copy(); logS_cut.loc[logS_cut.index > cutoff] = np.nan       # "chỉ biết dữ liệu đến cutoff"
    full, cut = features_fn(logS), features_fn(logS_cut)
    inside  = slice(None, cutoff + pd.Timedelta(days=horizon))
    outside = slice(cutoff + pd.Timedelta(days=horizon + 1), TRAIN_END)
    res = []
    for k in full:
        same_inside = np.allclose(full[k].loc[inside].values, cut[k].loc[inside].values, equal_nan=True)
        diff_after  = not np.allclose(full[k].loc[outside].values, cut[k].loc[outside].values, equal_nan=True)
        res.append((k, 'ĐẠT' if same_inside else 'RÒ RỈ!', 'có' if diff_after else 'không (n/a)'))
    return pd.DataFrame(res, columns=['Biến', f'Giống hệt khi cắt tại {cutoff.date()} (t <= C+48)?', 'Có đổi khi t > C+48 (test nhạy)?'])

fn_good  = lambda ls: build_history_features(ls, Promo)
fn_leaky = lambda ls: {'lag7_RÒ_RỈ': ls.shift(7), 'lag1_RÒ_RỈ': ls.shift(1)}
display(leakage_test('2015-03-09', fn_good))
display(leakage_test('2015-03-09', fn_leaky))

# %% [markdown cell 24] **Kết luận 5.4:** toàn bộ biến theo R5 đều **ĐẠT**; hai biến cố tình rò rỉ bị bắt ngay (`RÒ RỈ!`). (

# %% [code cell 25]
cnt = train.groupby('Store').Date.count()
short_stores = cnt[cnt < cnt.max()].index
print('Số cửa hàng thiếu ngày:', len(short_stores), '|', cnt[short_stores].value_counts().to_dict())
alld = set(ALL_DATES[ALL_DATES <= TRAIN_END])
gaps = {}
for s in short_stores:
    miss_days = sorted(alld - set(train.loc[train.Store == s, 'Date']))
    gaps[s] = (miss_days[0].date(), miss_days[-1].date(), len(miss_days))
gdf = pd.DataFrame(gaps, index=['thiếu từ', 'thiếu đến', 'số ngày thiếu']).T
print(gdf.groupby(['thiếu từ', 'thiếu đến', 'số ngày thiếu']).size().rename('số cửa hàng'))
print('\nSố cửa hàng thiếu này nằm trong test:', len(set(short_stores) & set(test_stores)), '/', len(short_stores))
# Hệ quả: tỷ lệ NaN của roll28 ở 180 cửa hàng mất 184 ngày, theo tháng
gap_stores = [s for s, g in gaps.items() if g[2] == 184]
nan_m = HIST['roll28'][gap_stores].isna().mean(axis=1).groupby(ALL_DATES.to_period('M')).mean()
print('Tỷ lệ NaN của roll28 ở 180 CH này theo tháng:', {str(k): round(float(v), 2) for k, v in nan_m.loc['2014-07':'2015-04'].items()})
n_by_date = S.loc[:TRAIN_END].notna().sum(axis=1)
fig, ax = plt.subplots(figsize=(9, 3.2)); n_by_date.plot(ax=ax, color='#4472C4')
ax.set_title('Daily Active Stores'); ax.set_ylabel('Store Count'); plt.tight_layout(); plt.show()


# %% [markdown cell 26] **Phát hiện (sửa lại nhận định ở U02 mục 1.10):** **180 cửa hàng mất đúng 184 ngày liên tục (01/07/2

# %% [code cell 27]
# R8: warm-up = ngày đầu tiên mà (trung vị theo cửa hàng) có >= 4 mẫu cho dow_promo_mean và roll91 khác NaN ở >= 90% cửa hàng
n_med = HIST['dow_promo_n'].loc[:TRAIN_END].median(axis=1)
cov91 = HIST['roll91'].loc[:TRAIN_END].notna().mean(axis=1)
warm_start = ALL_DATES[ALL_DATES <= TRAIN_END][(n_med >= 4).values & (cov91 >= 0.9).values][0]
print('Ngày đầu tiên đủ lịch sử (warm-up kết thúc):', warm_start.date(), f'-> loại {(warm_start - TRAIN_START).days} ngày đầu khỏi tập train')

# Độ phủ NaN của từng biến trên các dòng TEST sẽ được dự báo (856 CH, Open=1)
tt = test[(test.Open != 0)][['Store', 'Date']]
ri, ci = ALL_DATES.get_indexer(tt.Date), STORES.get_indexer(tt.Store)     # vị trí (ngày, cửa hàng) của từng dòng test trong ma trận
cov = {k: float(np.isnan(m.values[ri, ci]).mean()) for k, m in HIST.items()}
pd.Series(cov).round(3).rename('Tỷ lệ NaN trên dòng test (Open!=0)').to_frame()

# %% [markdown cell 28] **Nhận xét:** `ly364` thiếu ở khoảng 1/5 dòng test (đúng là 180 cửa hàng bị khoảng trống 2014), còn 
# %% [markdown cell 29] ---

# %% [code cell 30]
FOLD_ORIGINS = {
    'F1 (gần test nhất)':       pd.Timestamp('2015-06-13'),   # dự báo 14/06 -> 31/07/2015
    'F2':                       pd.Timestamp('2015-04-26'),   # dự báo 27/04 -> 13/06/2015
    'F3 (có lễ Phục sinh)':     pd.Timestamp('2015-03-09'),   # dự báo 10/03 -> 26/04/2015
    'F4 (cùng mùa năm trước)':  pd.Timestamp('2014-07-31'),   # dự báo 01/08 -> 17/09/2014
}
cols = [s for s in STORES if s in set(test_stores)]            # chỉ chấm trên 856 cửa hàng có trong test

def rmspe(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); m = y != 0       # Kaggle bỏ qua dòng Sales = 0
    return float(np.sqrt(np.mean(((y[m] - yhat[m]) / y[m]) ** 2)))

rows = []
for name, T in FOLD_ORIGINS.items():
    v0, v1 = T + pd.Timedelta(days=1), T + pd.Timedelta(days=HORIZON)
    Sv, Ov = S.loc[v0:v1, cols], O.loc[v0:v1, cols]
    ok = (Ov == 1) & (Sv > 0)
    vd = train[(train.Date >= v0) & (train.Date <= v1) & train.Store.isin(cols) & (train.Open == 1)]
    rows.append({'Fold': name, 'Mốc cắt T (hết dữ liệu train)': T.date(), 'Giai đoạn validate': f'{v0.date()} -> {v1.date()}',
                 'Số ngày': (v1 - v0).days + 1, 'Số CH có Sales': int(Sv.notna().any().sum()), 'Số dòng được chấm': int(ok.values.sum()),
                 'Tỷ lệ dòng Promo': round(vd.Promo.mean(), 2), 'Số dòng ngày lễ nhà nước': int((vd.StateHoliday != '0').sum())})
folds_df = pd.DataFrame(rows)
folds_df

# %% [markdown cell 31] **Nhận xét về các fold:**

# %% [code cell 32]
def profile_pred(T, val_index, cols, n_weeks, use_promo):
    # Dự báo = trung bình log-sales của n_weeks tuần gần nhất (<= T) cùng thứ (và cùng Promo nếu use_promo). Trả về ma trận val x cols.
    LT, PT = logS.loc[:T, cols], Promo.loc[:T, cols]
    pv = Promo.loc[val_index, cols].values
    out = np.full((len(val_index), len(cols)), np.nan)
    for d in range(7):
        hist = LT[LT.index.dayofweek == d].tail(n_weeks); hp = PT.loc[hist.index]
        rows_d = np.where(val_index.dayofweek == d)[0]
        if use_promo:
            for pr in (0, 1):
                mu = hist.where(hp == pr).mean().values
                for r in rows_d: out[r, pv[r] == pr] = mu[pv[r] == pr]
        else:
            mu = hist.mean().values
            for r in rows_d: out[r, :] = mu
    return out

def eval_fold(T):
    v0, v1 = T + pd.Timedelta(days=1), T + pd.Timedelta(days=HORIZON)
    vi = ALL_DATES[(ALL_DATES >= v0) & (ALL_DATES <= v1)]
    Sv, Ov = S.loc[vi, cols], O.loc[vi, cols]
    y = Sv.where((Ov == 1) & (Sv > 0)).values
    preds = {
        'B1 cùng thứ (8 tuần)':        profile_pred(T, vi, cols, 8, False),
        'B2 cùng thứ + cùng Promo':    profile_pred(T, vi, cols, 16, True),
        'B3 dow_promo_mean (lag>=49)': HIST['dow_promo_mean'].loc[vi, cols].values,
    }
    leaky_feat, _ = same_dow_same_promo_mean(logS, Promo, range(7, 57, 7))                  # X1: lag 7..49 -> rò rỉ
    preds['X1 RÒ RỈ lag 7-49'] = leaky_feat.loc[vi, cols].values
    preds['X2 RÒ RỈ profile toàn train'] = profile_pred(TRAIN_END, vi, cols, 10_000, True)  # X2: dùng cả giai đoạn validate
    res = {}
    for k, p in preds.items():
        pe = np.expm1(p); m = ~np.isnan(y) & ~np.isnan(pe)
        res[k] = round(rmspe(y[m], pe[m]), 4)
    return res

floor = pd.DataFrame({name: eval_fold(T) for name, T in FOLD_ORIGINS.items()})
floor['Trung bình 4 fold'] = floor.mean(axis=1).round(4)
floor


# %% [code cell 33]
fig, ax = plt.subplots(figsize=(10, 3.8))
SCENARIOS_EN = {
    'B1 cùng thứ (8 tuần)': 'B1 DOW Mean (8w)',
    'B2 cùng thứ + cùng Promo': 'B2 DOW + Promo (16w)',
    'B3 dow_promo_mean (lag>=49)': 'B3 dow_promo_mean (Valid)',
    'X1 RÒ RỈ lag 7-49': 'X1 Target Leakage (lag 7-49)',
    'X2 RÒ RỈ profile toàn train': 'X2 Target Leakage (Full Train)'
}
FOLDS_EN = {
    'F1 (gần test nhất)': 'F1 (Near Test)',
    'F2': 'F2 (Spring)',
    'F3 (có lễ Phục sinh)': 'F3 (Easter)',
    'F4 (cùng mùa năm trước)': 'F4 (Prior Year)'
}
plot_df = floor.drop(columns='Trung bình 4 fold', errors='ignore').rename(index=SCENARIOS_EN).rename(columns=FOLDS_EN).T
plot_df.plot(kind='bar', ax=ax, color=['#A6A6A6', '#4472C4', '#2E75B6', '#ED7D31', '#C00000'], width=0.8)
ax.set_ylabel('RMSPE')
ax.set_title('Baseline and Leakage RMSPE Across Folds')
ax.legend(fontsize=8, loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# %% [markdown cell 34] **Nhận xét (đọc từ bảng và biểu đồ):**
# %% [markdown cell 35] ---

# %% [code cell 36]
c = store[['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']].copy()
c['comp_open'] = pd.to_datetime(dict(year=c.CompetitionOpenSinceYear, month=c.CompetitionOpenSinceMonth, day=1), errors='coerce')
facts = {
    'CompetitionDistance thiếu (cửa hàng)': c[c.CompetitionDistance.isna()].Store.tolist(),
    'Có khoảng cách nhưng THIẾU ngày đối thủ mở (CH)': int((c.CompetitionDistance.notna() & c.comp_open.isna()).sum()),
    'Năm đối thủ mở < 1990 (nghi placeholder) (CH)': int((c.CompetitionOpenSinceYear < 1990).sum()),
    'Đối thủ mở SAU ngày cuối train (từ 08/2015) (CH)': int((c.comp_open >= TEST_START).sum()),
    'Đối thủ mở trong 12 tháng cuối của train (CH)': int(((c.comp_open >= TEST_START - pd.DateOffset(years=1)) & (c.comp_open < TEST_START)).sum()),
    'CompetitionDistance lớn nhất (m)': float(c.CompetitionDistance.max()),
}
pd.Series(facts).to_frame('giá trị')


# %% [code cell 37]
def add_competition_features(df, store):
    s = store[['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']].copy()
    s.loc[s.CompetitionOpenSinceYear < 1990, ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']] = np.nan   # placeholder -> coi như chưa biết
    s['comp_open'] = pd.to_datetime(dict(year=s.CompetitionOpenSinceYear, month=s.CompetitionOpenSinceMonth, day=1), errors='coerce')
    s['CompetitionDistanceMissing'] = s.CompetitionDistance.isna().astype('int8')
    s['CompetitionDistance'] = s.CompetitionDistance.fillna(s.CompetitionDistance.max()).astype('float32')   # không có đối thủ gần -> coi như xa nhất
    out = df.merge(s[['Store', 'CompetitionDistance', 'CompetitionDistanceMissing', 'comp_open']], on='Store', how='left')
    months = (out.Date.dt.year - out.comp_open.dt.year) * 12 + (out.Date.dt.month - out.comp_open.dt.month)
    out['CompetitionSinceUnknown'] = out.comp_open.isna().astype('int8')
    out['CompetitorNotOpenYet']    = (months < 0).astype('int8')                     # đối thủ chưa mở tại thời điểm của dòng này
    out['MonthsSinceCompetitionOpened'] = months.clip(lower=0).astype('float32')     # NaN giữ nguyên khi chưa biết
    return out.drop(columns='comp_open')

tc = add_competition_features(test[['Store', 'Date']], store)
print('Tỷ lệ dòng có cờ - TEST :', tc[['CompetitionDistanceMissing', 'CompetitionSinceUnknown', 'CompetitorNotOpenYet']].mean().round(4).to_dict())
trc = add_competition_features(train[['Store', 'Date']], store)
print('Tỷ lệ dòng có cờ - TRAIN:', trc[['CompetitionDistanceMissing', 'CompetitionSinceUnknown', 'CompetitorNotOpenYet']].mean().round(4).to_dict())
tc.MonthsSinceCompetitionOpened.describe().round(1)

# %% [markdown cell 38] **Luật R10 (Competition):**

# %% [code cell 39]
MONTHS = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Sept':9,'Oct':10,'Nov':11,'Dec':12}   # lưu ý 'Sept'

def add_promo2_features(df, store):
    s = store[['Store', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']].copy()
    for m in range(1, 13):
        s[f'm{m}'] = s.PromoInterval.fillna('').apply(lambda x: m in {MONTHS[t.strip()] for t in x.split(',') if t.strip()})
    s['p2_start'] = pd.to_datetime(s.Promo2SinceYear.astype('Int64').astype(str) + '-W' + s.Promo2SinceWeek.astype('Int64').astype(str).str.zfill(2) + '-1',
                                   format='%G-W%V-%u', errors='coerce')          # tuần ISO, khớp cách làm ở U02 mục 3.3
    out = df.merge(s.drop(columns=['PromoInterval', 'Promo2SinceWeek', 'Promo2SinceYear']), on='Store', how='left')
    mm = out[[f'm{m}' for m in range(1, 13)]].values
    in_interval = mm[np.arange(len(out)), out.Date.dt.month.values - 1]
    started = (out.Date >= out.p2_start).values
    out['Promo2Started'] = (out.Promo2.eq(1).values & started).astype('int8')
    out['IsPromo2ActiveThisMonth'] = (out.Promo2Started.values.astype(bool) & in_interval).astype('int8')
    out['Promo2WeeksSinceStart'] = np.where(out.Promo2Started.values == 1, ((out.Date - out.p2_start).dt.days // 7).values, np.nan).astype('float32')
    return out.drop(columns=[f'm{m}' for m in range(1, 13)] + ['p2_start'])

# --- đối chiếu độc lập với logic U02 mục 3.3 (so sánh năm-tuần ISO): số dòng khác nhau phải bằng 0
p2_ids = store[store.Promo2 == 1].Store
d = add_promo2_features(train[train.Store.isin(p2_ids)][['Store', 'Date']], store)
iso = d.Date.dt.isocalendar()
sy = d.merge(store[['Store', 'Promo2SinceYear', 'Promo2SinceWeek']], on='Store', how='left')
u02_past = ((iso.year.values > sy.Promo2SinceYear.values) | ((iso.year.values == sy.Promo2SinceYear.values) & (iso.week.values >= sy.Promo2SinceWeek.values)))
print('Đối chiếu với U02 mục 3.3 (Promo2Started):', int((u02_past.astype(int) != d.Promo2Started.values).sum()), 'dòng khác nhau trên', f'{len(d):,}', 'dòng')

# --- test nhanh: cửa hàng chu kỳ 'Jan,Apr,Jul,Oct' chỉ được active vào các tháng 1,4,7,10 (và sau mốc bắt đầu)
chk = add_promo2_features(train[train.Store.isin(store[store.PromoInterval == 'Jan,Apr,Jul,Oct'].Store.head(30))][['Store', 'Date']], store)
act_months = sorted(chk[chk.IsPromo2ActiveThisMonth == 1].Date.dt.month.unique())
print('Các tháng có IsPromo2ActiveThisMonth=1 của nhóm Jan,Apr,Jul,Oct:', act_months, '=> ĐẠT' if act_months == [1, 4, 7, 10] else '=> LỖI')

tp = add_promo2_features(test[['Store', 'Date']], store)
p2s = store[store.Promo2 == 1].copy()
p2s['start'] = pd.to_datetime(p2s.Promo2SinceYear.astype(int).astype(str) + '-W' + p2s.Promo2SinceWeek.astype(int).astype(str).str.zfill(2) + '-1', format='%G-W%V-%u')
print('\nPromo2 bắt đầu sau 01/08/2015:', int((p2s.start >= TEST_START).sum()), '| bắt đầu trong 12 tháng cuối của train:', int(((p2s.start >= TEST_START - pd.DateOffset(years=1)) & (p2s.start < TEST_START)).sum()))
print('Test: số dòng IsPromo2ActiveThisMonth=1 theo tháng:', tp[tp.IsPromo2ActiveThisMonth == 1].Date.dt.month.value_counts().sort_index().to_dict())

# %% [markdown cell 40] **Luật R11 (Promo2):**

# %% [code cell 41]
print('Test: StateHoliday =', test.StateHoliday.value_counts().to_dict(), '| SchoolHoliday = 1:', round(test.SchoolHoliday.mean(), 3))
print(test[test.StateHoliday != '0'].groupby('Date').agg(số_dòng=('Store', 'count'), số_dòng_mở_cửa=('Open', 'sum')))

allcal = pd.concat([train[['Store', 'Date', 'StateHoliday']], test[['Store', 'Date', 'StateHoliday']]])
share = allcal.assign(h=allcal.StateHoliday != '0').groupby('Date').h.mean()
print('\nSố ngày mà StateHoliday khác nhau giữa các cửa hàng (lễ theo bang):', int(((share > 0) & (share < 1)).sum()))
nat = share[share >= 0.5].index
print('Số ngày lễ "toàn chuỗi" (>=50% cửa hàng):', len(nat), '| trong giai đoạn test:', int((nat >= TEST_START).sum()))

# Rule cho biến lịch lễ
def add_holiday_features(df, nat_dates, cap=30):
    d = df.copy(); dates = pd.DatetimeIndex(sorted(nat_dates))
    idx_prev = dates.searchsorted(d.Date.values, side='right') - 1
    last = pd.Series(dates[np.clip(idx_prev, 0, None)], index=d.index).where(idx_prev >= 0)
    d['DaysSinceLastHoliday'] = (d.Date - last).dt.days.clip(upper=cap).astype('float32')
    d['IsDayAfterHoliday']  = d.Date.isin(dates + pd.Timedelta(days=1)).astype('int8')
    d['IsDayBeforeHoliday'] = d.Date.isin(dates - pd.Timedelta(days=1)).astype('int8')
    return d
th = add_holiday_features(test[['Store', 'Date']], nat)
print('\nTest: DaysSinceLastHoliday (cap 30):', th.DaysSinceLastHoliday.describe().round(1).to_dict())

# %% [markdown cell 42] **Nhận xét & luật R12 (lịch lễ):**
# %% [markdown cell 43] ---

# %% [code cell 44]
# (1) Minh họa: sai số log ≈ sai số phần trăm khi sai số nhỏ
y = np.array([2000., 2000., 10000., 10000.]); yhat = np.array([1500., 2500., 9500., 10500.])
print(pd.DataFrame({'y': y, 'ŷ': yhat, 'sai số %': ((y-yhat)/y).round(3), 'sai số log': (np.log(y)-np.log(yhat)).round(3)}))

# (2) Hàm hậu xử lý CHUẨN để dùng cho mọi model (R3 + R13)
def postprocess(pred_log, open_flag, c=1.0):
    # pred_log: dự báo trên thang log1p | open_flag: cột Open của test | c: hệ số hiệu chỉnh (mặc định 1.0, tune ở M02)
    pred = np.expm1(np.asarray(pred_log, float)) * c
    open_flag = np.where(pd.isna(open_flag), 1, open_flag)         # R3.2: Open thiếu -> 1
    pred = np.where(open_flag == 1, pred, 0.0)                      # R3.1: đóng cửa -> 0
    return np.clip(pred, 0, None)

# (3) Unit test nhỏ
out = postprocess([np.log1p(1000), np.log1p(1000), np.log1p(1000)], [1, 0, np.nan])
assert np.allclose(out, [1000, 0, 1000]), out
print('postprocess OK:', out)

# (4) Định dạng file nộp: Id, Sales (đúng thứ tự Id của test)
sub_tmpl = pd.read_csv(find('sample_submission_ross.csv', 'sample_submission.csv'))
assert list(sub_tmpl.columns) == ['Id', 'Sales'] and set(sub_tmpl.Id) == set(test.Id)
def make_submission(ids, pred): return pd.DataFrame({'Id': ids, 'Sales': pred}).sort_values('Id').reset_index(drop=True)
print('Định dạng nộp bài hợp lệ:', sub_tmpl.shape, '| Id test khớp sample_submission: True')

# %% [markdown cell 45] **Luật R13:**

# %% [code cell 46]
CATS = {'StateHoliday': ['0', 'a', 'b', 'c'], 'StoreType': list('abcd'), 'Assortment': list('abc')}
def align_categories(df):
    df = df.copy()
    for col, levels in CATS.items():
        if col in df: df[col] = pd.Categorical(df[col], categories=levels)
    return df

print('StateHoliday mức xuất hiện - train:', sorted(train.StateHoliday.unique()), '| test:', sorted(test.StateHoliday.unique()))
a, b = align_categories(train[['StateHoliday']]), align_categories(test[['StateHoliday']])
print('Mã số của mức "a" - train:', int(a.StateHoliday.cat.codes[a.StateHoliday == 'a'].iloc[0]), '| test:', int(b.StateHoliday.cat.codes[b.StateHoliday == 'a'].iloc[0]), '(phải bằng nhau)')


# %% [code cell 47]
RULES = pd.DataFrame([
 ('R1',  'Biến',        'Chỉ dùng cột có trong test/store; Customers bị chặn (chỉ được dùng dạng lịch sử ≥48 ngày, tùy chọn); Sales chỉ làm target và nguyên liệu cho biến lịch sử', 'Mục 2: Customers và Sales không có trong test', 'F04, M02'),
 ('R2',  'Dòng train',  'Chỉ train trên Open=1 và Sales>0 (loại Open=0 và 54 dòng Open=1&Sales=0)', 'Mục 3', 'F04'),
 ('R3',  'Open',        'Open=0 -> Sales=0; 11 dòng Open thiếu (Store 622) -> Open=1', 'Mục 3: Store 622 mở 94-100% T2-T7; đoán 0 nhầm làm RMSPE tăng ~0.001-0.002', 'Hậu xử lý'),
 ('R4',  'Promo',       'Dùng nguyên lịch Promo của test (chỉ T2-T6); tạo IsPromoWeek/Promo_prev7/Promo_next7; mọi biến lịch sử ghép theo trạng thái Promo', 'Mục 4 và 6: ghép Promo giảm RMSPE baseline từ ~0.2 xuống ~0.15', 'F04'),
 ('R5',  'Lag/rolling', 'Biến lịch sử chỉ dùng Sales của ngày <= t-48 (lag >= 48, cửa sổ rolling kết thúc ở t-48); bắt buộc qua leakage_test', 'Mục 5.4: các biến R5 ĐẠT, biến lag1/lag7 bị bắt RÒ RỈ', 'F04'),
 ('R6',  'Profile CH',  'Mọi thống kê theo cửa hàng/target encoding chỉ tính từ dữ liệu <= mốc cắt của fold; cấm tính trên toàn bộ train', 'Mục 6: X2 (profile toàn train) cho điểm đẹp giả', 'F04, M02'),
 ('R7',  'Dữ liệu thiếu','180 CH mất 01/07-31/12/2014: giữ NaN + cờ has_ly364, thêm ly728 làm dự phòng; GB nhận NaN, RF phải điền (roll91 rồi trung bình StoreType x Assortment)', 'Mục 5.5: 180/181 CH thiếu đều nằm trong test (21%)', 'F04, M02'),
 ('R8',  'Warm-up',     'Loại các ngày đầu train chưa đủ lịch sử khỏi tập train', 'Mục 5.5', 'F04'),
 ('R9',  'Validation',  'Walk-forward 4 fold x 48 ngày (3 fold gần + 1 fold cùng mùa 2014); chấm trên 856 CH; RMSPE bỏ Sales=0; không K-fold ngẫu nhiên; Kaggle chỉ là kỳ thi cuối, không tune theo Kaggle', 'Mục 6', 'M02, E02'),
 ('R10', 'Competition', 'MonthsSinceCompetitionOpened tính theo ngày dòng; âm -> 0 + cờ; thiếu -> NaN + cờ; năm<1990 -> NaN; thiếu Distance -> max + cờ', 'Mục 7.1: 351 CH thiếu ngày mở, 3 CH thiếu Distance, 3 CH đối thủ mở sau train', 'F04'),
 ('R11', 'Promo2',      'IsPromo2ActiveThisMonth theo định nghĩa thống nhất ở U02 mục 3.3 (tuần ISO, ánh xạ Sept); thêm Promo2Started, Promo2WeeksSinceStart; chỉ là biến dự báo, không suy nhân quả', 'Mục 7.2: không CH nào bắt đầu Promo2 sau train', 'F04, M03'),
 ('R12', 'Lịch lễ',     'StateHoliday theo từng CH; khoảng cách lễ tính từ lịch lễ toàn chuỗi (cap 30); báo cáo thêm RMSPE bỏ ngày lễ', 'Mục 7.3: test chỉ có 1 ngày lễ cấp bang (180 CH), 0 lễ toàn chuỗi; F1, F4 không có ngày lễ', 'F04, E02'),
 ('R13', 'Target',       'Train trên log1p(Sales); Sales_hat = expm1(ŷ) x c (c tune trên fold); đóng cửa -> 0; dùng chung hàm postprocess', 'Mục 8', 'M02, E02'),
 ('R14', 'Nhất quán',   'Train và test dùng CÙNG hàm tạo biến, CÙNG danh sách category, CÙNG thứ tự cột; kiểm tra tỷ lệ NaN từng cột giữa train và test', 'Mục 9.1', 'F04'),
 ('R15', 'Đóng cửa dài', 'Từ lịch Open (biết trước ở cả test): DaysSinceReopen = số ngày kể từ lần mở lại sau đợt đóng cửa liên tục >= 10 ngày (cap 120); LastClosureLen = độ dài đợt đóng cửa đó. Chỉ nhìn ngược về quá khứ, không nhìn trước. Giữ lại hay không do kiểm tra A/B ở mục 12.2', 'U02 mục 1.7: 72 CH từng đóng liên tục >= 10 ngày (tối đa 185 ngày); doanh số sau mở lại có thể lệch mạnh', 'F04, M02'),
 ('R16', 'Mùa vụ chuỗi', 'chain_delta = chênh lệch log-sales trung bình toàn chuỗi giữa tuần cùng kỳ năm trước và cửa sổ 91 ngày dùng làm mốc (cùng kỳ năm trước); chỉ dùng Sales <= t-361. Giữ lại hay không do kiểm tra A/B ở mục 12.2', 'U02 mục 3.5: tháng 12 cao (1.24), tháng 1-2 và 8-10 thấp (~0.94-0.96)', 'F04, M02'),
], columns=['Mã', 'Nhóm', 'Luật', 'Bằng chứng (từ notebook này)', 'Áp dụng ở task'])
RULES


# %% [code cell 48]
SPEC = pd.DataFrame([
 # feature, nhóm, lúc dự báo, cách tính / luật, dtype, xử lý NaN, RF, GB
 ('Store',                        'Định danh', 'Biết trước', 'Mã cửa hàng. GB dùng như category; RF KHÔNG one-hot 1.115 mức mà dùng các biến profile bên dưới', 'category (int16)', 'không có NaN', 'Không', 'Có'),
 ('DayOfWeek',                    'Lịch', 'Biết trước', '1-7', 'int8', 'không có NaN', 'Có', 'Có'),
 ('Month, WeekOfYear(ISO), DayOfMonth, DaysToMonthEnd', 'Lịch', 'Biết trước', 'Tách từ Date. KHÔNG dùng Year hoặc Date thô (cây không ngoại suy được sang giá trị mới)', 'int8', 'không có NaN', 'Có', 'Có'),
 ('Promo',                        'Khuyến mãi', 'Biết trước', 'Nguyên từ test (R4.1), chỉ T2-T6', 'int8', 'không có NaN', 'Có', 'Có'),
 ('IsPromoWeek, Promo_prev7, Promo_next7', 'Khuyến mãi', 'Biết trước', 'Từ lịch Promo (R4.2); Promo_next7 ở cuối test = -1', 'int8', 'điền -1 ở biên', 'Có', 'Có'),
 ('StateHoliday',                 'Lịch lễ', 'Biết trước', "Mức cố định ['0','a','b','c'] -> mã 0-3 (R14)", 'category (int8)', 'không có NaN', 'Có', 'Có'),
 ('SchoolHoliday',                'Lịch lễ', 'Biết trước', 'Nguyên từ test', 'int8', 'không có NaN', 'Có', 'Có'),
 ('DaysSinceLastHoliday, IsDayBeforeHoliday, IsDayAfterHoliday', 'Lịch lễ', 'Biết trước', 'Từ lịch lễ toàn chuỗi, cap 30 ngày (R12.2)', 'float32 / int8', 'không có NaN', 'Có', 'Có'),
 ('StoreType, Assortment',        'Đặc điểm CH', 'Biết trước', 'Mức cố định a-d, a-c (R14). Nhóm b rất nhỏ (n=17 và n=9): chỉ kết luận kèm n', 'category (int8)', 'không có NaN', 'Có', 'Có'),
 ('CompetitionDistance, CompetitionDistanceMissing', 'Đối thủ', 'Biết trước', 'Thiếu -> giá trị lớn nhất + cờ (R10.5)', 'float32 / int8', 'đã điền', 'Có', 'Có'),
 ('MonthsSinceCompetitionOpened, CompetitionSinceUnknown, CompetitorNotOpenYet', 'Đối thủ', 'Biết trước', 'Tính theo ngày dòng; âm -> 0 + cờ; thiếu -> NaN + cờ (R10)', 'float32 / int8', 'NaN (RF: điền -1 + giữ cờ)', 'Có', 'Có'),
 ('Promo2, Promo2Started, IsPromo2ActiveThisMonth', 'Promo2', 'Biết trước', 'Định nghĩa U02 mục 3.3 (R11)', 'int8', 'không có NaN', 'Có', 'Có'),
 ('Promo2WeeksSinceStart',        'Promo2', 'Biết trước', 'Số tuần kể từ mốc tham gia; NaN nếu chưa/không tham gia', 'float32', 'NaN (RF: điền -1)', 'Có', 'Có'),
 ('DaysSinceReopen, LastClosureLen', 'Đóng cửa dài (R15)', 'Biết trước (từ lịch Open)', 'Số ngày kể từ lần mở lại sau đợt đóng cửa >= 10 ngày liên tục (cap 120) và độ dài đợt đó (cap 200)', 'float32', 'không có NaN (không có đợt đóng cửa -> 120 / 0)', 'Tùy (A/B)', 'Tùy (A/B)'),
 ('chain_delta',                  'Mùa vụ chuỗi (R16)', 'Quá khứ >= 361 ngày', 'Log-sales TB toàn chuỗi (panel cố định) của tuần cùng kỳ năm trước trừ cửa sổ mốc 91 ngày cùng kỳ năm trước', 'float32', 'NaN trước 05/2014 (RF: điền 0)', 'Tùy (A/B)', 'Tùy (A/B)'),
 ('dow_promo_mean, dow_promo_n',  'Lịch sử Sales', 'Quá khứ >= 49 ngày', 'Trung bình log-sales cùng thứ và cùng Promo, lag 49-119 (R5); dow_promo_n = số mẫu', 'float32 / int8', 'NaN (RF: điền roll91 -> trung bình StoreType x Assortment)', 'Có', 'Có'),
 ('roll28, roll91, open28',       'Lịch sử Sales', 'Quá khứ >= 48 ngày', 'Rolling ngày mở cửa, cửa sổ kết thúc ở t-48; open28 = số ngày mở trong 28 ngày đó', 'float32 / int8', 'NaN (RF: như trên)', 'Có', 'Có'),
 ('promo_uplift91',               'Lịch sử Sales', 'Quá khứ >= 48 ngày', 'Mức tăng (log) khi có Promo so với không Promo của riêng cửa hàng', 'float32', 'NaN (RF: điền 0)', 'Có', 'Có'),
 ('ly364, ly728, has_ly364',      'Lịch sử Sales', 'Quá khứ >= 364 ngày', 'Log-sales cùng thứ 1 và 2 năm trước; has_ly364 = ly364 có giá trị (R7)', 'float32 / int8', 'NaN + cờ (RF: điền dow_promo_mean)', 'Có', 'Có'),
 ('(tùy chọn) cust_roll28',       'Lịch sử Customers', 'Quá khứ >= 48 ngày', 'Trung bình Customers kết thúc ở t-48 (R1.2); chỉ thêm nếu cải thiện validation', 'float32', 'NaN', 'Tùy', 'Tùy'),
 ('Customers (trực tiếp)',        'BỊ CHẶN', 'KHÔNG có lúc dự báo', 'Không có trong test (R1.2)', '-', '-', 'Không', 'Không'),
 ('Sales (thô), Id',              'BỊ CHẶN', 'Target / mã nộp bài', 'Sales chỉ là target; Id chỉ để nộp bài', '-', '-', 'Không', 'Không'),
], columns=['Feature', 'Nhóm', 'Có sẵn lúc dự báo?', 'Cách tính / luật', 'Dtype đề xuất', 'Xử lý NaN', 'Dùng cho RF', 'Dùng cho GB'])
SPEC

# %% [markdown cell 49] ### 9.2 Đối chiếu: các phát hiện ở bước EDA (U02) đã được đưa vào luật chưa?

# %% [code cell 50]
EDA_MAP = pd.DataFrame([
 ('1.1',  'Missing: Open test thiếu 11 dòng; Competition thiếu; Promo2 thiếu có cấu trúc', 'R3.2, R10, R11', 'Đã có'),
 ('1.2',  'StateHoliday: 0 (số) và "0" (chuỗi) bị tách nhóm', 'Ép chuỗi lúc đọc + danh sách category cố định (R14)', 'Đã có'),
 ('1.3',  'Open=0 -> Sales=0; 54 dòng Open=1 & Sales=0', 'R2, R3.1', 'Đã có'),
 ('1.4',  'PromoInterval chỉ có 3 chu kỳ; "Sept" khác "Sep"', 'R11 (bảng ánh xạ tháng)', 'Đã có'),
 ('1.5/2.3', 'StoreType b (17 CH) và Assortment b (9 CH) quá nhỏ', 'Ghi n khi báo cáo; bảng sai số theo nhóm ở E02', 'Đã có'),
 ('1.6',  'test chỉ có 856/1.115 cửa hàng', 'R9: chỉ chấm trên 856 CH; train dùng tất cả CH', 'Đã có'),
 ('1.7',  '72 CH từng đóng cửa liên tục >= 10 ngày (tối đa 185 ngày); doanh số sau mở lại và trung bình cuốn bị méo', 'Trước đây chỉ có open28 và cờ báo cáo ở E02', 'THIẾU -> bổ sung R15 (DaysSinceReopen, LastClosureLen), kiểm tra A/B ở 12.2'),
 ('1.8',  'CompetitionOpenSinceYear có giá trị 1900, 1961', 'R10.4 (năm < 1990 -> NaN)', 'Đã có'),
 ('1.9',  'Không trùng lặp; (Store, Date) là khóa duy nhất', 'Thêm assert kiểm tra khóa duy nhất trong F04 (mục 11.1)', 'Đã bổ sung kiểm tra'),
 ('1.10', '181 CH thiếu ngày liên tục', 'R7 (đã sửa: mất 01/07-31/12/2014, không phải bắt đầu muộn)', 'Đã có (và đã sửa nhận định của U02)'),
 ('1.11', 'Promo chỉ chạy T2-T6', 'R4.1', 'Đã có'),
 ('1.12', '11 dòng Open thiếu đều là Store 622 -> Open=1', 'R3.2', 'Đã có'),
 ('1.13', '191 CH (17,1%) có đối thủ mới mở trong giai đoạn train', 'R10: MonthsSinceCompetitionOpened tính theo ngày dòng', 'Đã có'),
 ('1.14', 'Sales lệch phải; RMSPE; log và gán 0 khi đóng cửa', 'R13', 'Đã có'),
 ('1.15', '225 CH tham gia Promo2 trong giai đoạn train', 'Chỉ phục vụ phân tích Promo2, không dùng cho dự báo', 'Không áp dụng (phân tích Promo2 đã bỏ khỏi notebook)'),
 ('2.1',  'Promo +38%; nguy cơ mua dồn / dịch chuyển nhu cầu (forward-buying)', 'R4.2: Promo_prev7, Promo_next7, IsPromoWeek', 'Đã có'),
 ('2.2',  'CompetitionDistance gần như không liên quan Sales (r ~ -0,04)', 'Giữ làm biến phụ, không kỳ vọng nhiều', 'Đã có'),
 ('2.4',  'Chủ nhật hiếm khi mở nhưng khi mở thì doanh số cao; Thứ 7 thấp nhất', 'DayOfWeek + biến lịch sử cùng thứ; báo cáo riêng nhóm Chủ nhật ở E02', 'Đã có'),
 ('2.5',  'SchoolHoliday +4,4% chỉ là số gộp', 'R12.4', 'Đã có'),
 ('2.6',  'StateHoliday: nghịch lý Simpson (StoreType b tăng, a/d giảm)', 'R12; mô hình cây tự học tương tác; E02 báo cáo riêng, đây vẫn là điểm yếu', 'Đã có (còn hạn chế)'),
 ('2.7',  'Customers không có trong test', 'R1.2', 'Đã có'),
 ('3.5',  'Mùa vụ thị trường rõ (tháng 12 cao; tháng 1-2 và 8-10 thấp)', 'Trước đây chỉ có Month, WeekOfYear, ly364', 'THIẾU -> bổ sung R16 (chain_delta), kiểm tra A/B ở 12.2'),
 ('3.1-3.4, 3.6', 'Điều tra Promo2 (selection bias, within-store, mùa vụ)', 'Thuộc phân tích Promo2', 'Không áp dụng (đã bỏ khỏi notebook)'),
 ('4',    'Root cause 2: cần mô hình dự báo tập trung, walk-forward, kiểm soát leakage, đúng phạm vi 856 CH', 'R5, R6, R9', 'Đã có'),
], columns=['Mục U02', 'Phát hiện ở EDA', 'Xử lý trong pipeline', 'Trạng thái'])
EDA_MAP


# %% [code cell 51]
RULES.to_csv(OUT_DIR / 'F02_rules.csv', index=False, encoding='utf-8-sig')
SPEC.to_csv(OUT_DIR / 'F02_feature_spec.csv', index=False, encoding='utf-8-sig')
folds_df.merge(floor.drop(columns='Trung bình 4 fold').T.reset_index().rename(columns={'index': 'Fold'}), on='Fold').to_csv(OUT_DIR / 'F02_validation_folds_and_baseline_floor.csv', index=False, encoding='utf-8-sig')
EDA_MAP.to_csv(OUT_DIR / 'F02_eda_to_rules_map.csv', index=False, encoding='utf-8-sig')
print('Đã lưu: F02_rules.csv | F02_feature_spec.csv | F02_validation_folds_and_baseline_floor.csv | F02_eda_to_rules_map.csv  ->', OUT_DIR.resolve())

# %% [markdown cell 52] ---
# %% [markdown cell 53] ---

# %% [code cell 54]
QUICK = False            # True = chạy nhanh để kiểm tra code (ít cây, ít cấu hình tuning, ít cửa hàng). False = cấu hình dùng cho báo cáo.
SEED  = 42
N_JOBS = -1
rng = np.random.default_rng(SEED)

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time
print('LightGBM:', 'có' if HAS_LGB else 'KHÔNG có -> dùng HistGradientBoostingRegressor của sklearn thay thế', '| QUICK =', QUICK)

# %% [markdown cell 55] ### 11.1 Ghép train + test thành một bảng, tạo biến lịch / Promo / lễ / cửa hàng

# %% [code cell 56]
base_cols = ['Store', 'Date', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
tr_ = train[base_cols + ['Sales']].copy(); tr_['is_test'] = 0; tr_['Id'] = -1
te_ = test[base_cols + ['Id']].copy();     te_['Sales'] = np.nan; te_['is_test'] = 1
te_['Open'] = te_['Open'].fillna(1)                                  # R3.2: Open thiếu (Store 622) -> 1
TB = pd.concat([tr_, te_], ignore_index=True)
assert not TB.duplicated(['Store', 'Date']).any(), 'Vi phạm: (Store, Date) phải là khóa duy nhất (U02 mục 1.9)'
ri, ci = ALL_DATES.get_indexer(TB.Date), STORES.get_indexer(TB.Store)   # vị trí của từng dòng trong ma trận ngày x cửa hàng

# --- lịch
TB['Month']       = TB.Date.dt.month.astype('int8')
TB['WeekOfYear']  = TB.Date.dt.isocalendar().week.astype('int8')
TB['DayOfMonth']  = TB.Date.dt.day.astype('int8')
TB['DaysToMonthEnd'] = (TB.Date.dt.days_in_month - TB.Date.dt.day).astype('int8')

# --- lịch Promo biết trước (R4.2)
PM = Promo.values
week_block = pd.factorize(ALL_DATES.to_period('W-SUN'))[0]
IsPromoWeek = pd.DataFrame(PM).groupby(week_block).transform('max').values
TB['IsPromoWeek'] = np.nan_to_num(IsPromoWeek[ri, ci], nan=0).astype('int8')
TB['Promo_prev7'] = np.nan_to_num(pd.DataFrame(PM).shift(7).values[ri, ci],  nan=-1).astype('int8')
TB['Promo_next7'] = np.nan_to_num(pd.DataFrame(PM).shift(-7).values[ri, ci], nan=-1).astype('int8')

# --- lịch lễ (R12): StateHoliday mã hóa với danh sách mức cố định
TB = add_holiday_features(TB, nat)
TB['StateHoliday'] = pd.Categorical(TB.StateHoliday, categories=CATS['StateHoliday']).codes.astype('int8')

# --- đặc điểm cửa hàng cố định
st_small = store[['Store', 'StoreType', 'Assortment']].copy()
TB = TB.merge(st_small, on='Store', how='left')
TB['StoreType']  = pd.Categorical(TB.StoreType,  categories=CATS['StoreType']).codes.astype('int8')
TB['Assortment'] = pd.Categorical(TB.Assortment, categories=CATS['Assortment']).codes.astype('int8')
assert (ALL_DATES.get_indexer(TB.Date) == ri).all()          # merge không làm đổi thứ tự dòng

# --- Competition (R10) và Promo2 (R11): đúng hàm đã kiểm tra ở mục 7
cf = add_competition_features(TB[['Store', 'Date']], store)
for c_ in ['CompetitionDistance', 'CompetitionDistanceMissing', 'MonthsSinceCompetitionOpened', 'CompetitionSinceUnknown', 'CompetitorNotOpenYet']:
    TB[c_] = cf[c_].values
pf = add_promo2_features(TB[['Store', 'Date']], store)
for c_ in ['Promo2', 'Promo2Started', 'IsPromo2ActiveThisMonth', 'Promo2WeeksSinceStart']:
    TB[c_] = pf[c_].values
del cf, pf
print('Bảng sau bước 11.1:', TB.shape)

# %% [markdown cell 57] ### 11.2 Ghép biến lịch sử Sales (R5-R7) và tạo biến mục tiêu

# %% [code cell 58]
for k, m in HIST.items():
    TB[k] = m.values[ri, ci].astype('float32')
TB['has_ly364'] = TB['ly364'].notna().astype('int8')                  # R7: cờ thiếu "cùng kỳ năm trước"
TB['dow_promo_n'] = TB['dow_promo_n'].fillna(0).astype('int8')
TB['open28'] = TB['open28'].fillna(0).astype('int8')

# biến mục tiêu và cờ dòng được phép train (R2, R8, R13)
TB['y_log']    = np.log1p(TB.Sales).astype('float32')
TB['valid_y']  = ((TB.is_test == 0) & (TB.Open == 1) & (TB.Sales > 0))
TB['train_ok'] = TB.valid_y & (TB.Date >= warm_start)

# kiểu dữ liệu gọn
for c_ in ['CompetitionDistance', 'MonthsSinceCompetitionOpened', 'Promo2WeeksSinceStart', 'DaysSinceLastHoliday']:
    TB[c_] = TB[c_].astype('float32')
for c_ in ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek', 'is_test', 'Promo2']:
    TB[c_] = TB[c_].astype('int8')
TB['Store'] = TB['Store'].astype('int16')
print(f'Bảng cuối: {TB.shape[0]:,} dòng x {TB.shape[1]} cột | bộ nhớ ~{TB.memory_usage(deep=True).sum()/1e6:.0f} MB')
print(f"Dòng được phép train (R2+R8): {int(TB.train_ok.sum()):,} | dòng test: {int((TB.is_test==1).sum()):,}")

# %% [markdown cell 59] ### 11.2b Hai nhóm biến còn thiếu so với EDA (U02): đóng cửa dài ngày (R15) và mùa vụ thị trường (R1

# %% [code cell 60]
# --- R15: đóng cửa dài ngày. Ma trận Open (ngày x cửa hàng) gồm cả giai đoạn test (Open của test có sẵn).
OPEN_M = np.full((len(ALL_DATES), len(STORES)), np.nan); OPEN_M[ri, ci] = TB.Open.values.astype(float)
LONG_CLOSURE_DAYS, CAP_REOPEN = 10, 120

def closure_features(open_m, min_len=LONG_CLOSURE_DAYS, cap=CAP_REOPEN):
    nd, ns = open_m.shape; idx = np.arange(nd)
    since = np.full((nd, ns), float(cap), dtype='float32'); last_len = np.zeros((nd, ns), dtype='float32'); n_spells = 0
    for j in range(ns):
        closed = (open_m[:, j] == 0).astype(np.int8)                                  # NaN (thiếu ngày) không phải đóng cửa
        d = np.diff(np.concatenate([[0], closed, [0]])); starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0] - 1
        marker = np.full(nd, -1); lens = np.zeros(nd)
        for s_, e_ in zip(starts, ends):
            L = e_ - s_ + 1
            if L >= min_len and e_ + 1 < nd: marker[e_ + 1] = e_ + 1; lens[e_ + 1] = L; n_spells += 1
        if (marker >= 0).any():
            pos = np.maximum.accumulate(marker); ok = pos >= 0
            since[ok, j] = np.minimum(idx[ok] - pos[ok], cap); last_len[ok, j] = np.minimum(lens[pos[ok]], 200)
    return since, last_len, n_spells

SINCE_M, LEN_M, n_sp = closure_features(OPEN_M)
TB['DaysSinceReopen'] = SINCE_M[ri, ci]; TB['LastClosureLen'] = LEN_M[ri, ci]
stores_closed = int((LEN_M.max(axis=0) > 0).sum())
print(f'R15: {n_sp} đợt đóng cửa >= {LONG_CLOSURE_DAYS} ngày liên tục, {stores_closed} cửa hàng (U02 mục 1.7 ghi 72 CH >= 10 ngày trong train); đợt dài nhất = {int(LEN_M.max())} ngày')
print('Dòng test có DaysSinceReopen < 120 (vừa mở lại gần đây):', int((TB.loc[TB.is_test == 1, 'DaysSinceReopen'] < CAP_REOPEN).sum()))

# --- R16: mùa vụ thị trường
panel_stores = [s_ for s_ in STORES if (s_ not in set(gap_stores)) and store.set_index('Store').loc[s_, 'StoreType'] != 'b']
def chain_delta_series(logS_, horizon=HORIZON):
    chain = logS_[panel_stores].mean(axis=1)                                          # log-sales TB toàn chuỗi mỗi ngày (ngày mở cửa)
    A = chain.rolling(7, center=True, min_periods=4).mean().shift(364)                # tuần cùng kỳ năm trước (cửa sổ tâm, dùng tới t-361)
    B = chain.rolling(91, min_periods=60).mean().shift(364 + horizon)                 # cửa sổ mốc 91 ngày cùng kỳ năm trước
    return (A - B)
CHAIN_DELTA = chain_delta_series(logS)
TB['chain_delta'] = CHAIN_DELTA.values[ri].astype('float32')
print(f'R16: panel {len(panel_stores)} cửa hàng | chain_delta có giá trị ở {TB.chain_delta.notna().mean()*100:.1f}% dòng | test: {TB.loc[TB.is_test == 1, "chain_delta"].notna().mean()*100:.1f}%')
print('Khoảng giá trị chain_delta (test):', TB.loc[TB.is_test == 1, 'chain_delta'].describe()[['min', 'mean', 'max']].round(3).to_dict())


# %% [code cell 61]
CAT_FEATURES = ['Store', 'StoreType', 'Assortment', 'StateHoliday']           # biến phân loại (Gradient Boosting dùng natively)
FEATURES = ['Store', 'DayOfWeek', 'Month', 'WeekOfYear', 'DayOfMonth', 'DaysToMonthEnd',
            'Promo', 'IsPromoWeek', 'Promo_prev7', 'Promo_next7',
            'StateHoliday', 'SchoolHoliday', 'DaysSinceLastHoliday', 'IsDayBeforeHoliday', 'IsDayAfterHoliday',
            'StoreType', 'Assortment',
            'CompetitionDistance', 'CompetitionDistanceMissing', 'MonthsSinceCompetitionOpened', 'CompetitionSinceUnknown', 'CompetitorNotOpenYet',
            'Promo2', 'Promo2Started', 'IsPromo2ActiveThisMonth', 'Promo2WeeksSinceStart',
            'dow_promo_mean', 'dow_promo_n', 'roll28', 'roll91', 'open28', 'promo_uplift91', 'ly364', 'ly728', 'has_ly364']
FORBIDDEN = {'Customers', 'Sales', 'Id', 'y_log', 'Date'}
assert not (set(FEATURES) & FORBIDDEN), 'Có biến bị cấm trong FEATURES (R1)'
assert all(f in TB.columns for f in FEATURES)
print(len(FEATURES), 'biến đưa vào model; không có Customers / Sales thô / Id (R1).')

# %% [markdown cell 62] ### 11.3 Kiểm tra nhất quán train - test (R14) và kiểm tra rò rỉ cuối cùng

# %% [code cell 63]
trn = TB[TB.train_ok]; tst = TB[(TB.is_test == 1) & (TB.Open == 1)]
cmp_nan = pd.DataFrame({'NaN% train': trn[FEATURES].isna().mean(), 'NaN% test': tst[FEATURES].isna().mean()})
cmp_nan['chênh lệch'] = (cmp_nan['NaN% test'] - cmp_nan['NaN% train']).round(3)
display(cmp_nan[(cmp_nan[['NaN% train', 'NaN% test']].max(axis=1) > 0)].round(3))

# phân phối trung bình các biến số chính: train vs test (kiểm tra lệch phân phối)
chk = ['dow_promo_mean', 'roll91', 'CompetitionDistance', 'MonthsSinceCompetitionOpened', 'Promo']
display(pd.DataFrame({'train (trung bình)': trn[chk].mean(), 'test (trung bình)': tst[chk].mean()}).round(3))

# kiểm tra rò rỉ lần cuối trên chính bảng cuối: không có biến nào mượn Sales của ngày > t-48 (đã test ở 5.4); không có Customers
print('Customers trong bảng cuối?', 'Customers' in TB.columns, '| Sales thô nằm trong FEATURES?', 'Sales' in FEATURES)

# %% [markdown cell 64] **Kiểm tra rò rỉ ở ĐÚNG 4 mốc cắt của 4 fold** (không chỉ mốc 09/03/2015 ở mục 5.4): với mỗi mốc T, 

# %% [code cell 65]
def leak_all_origins(fn):
    out = {}
    for fname, T0 in FOLD_ORIGINS.items():
        res = leakage_test(T0, fn); out[fname] = bool((res.iloc[:, 1] == 'ĐẠT').all())
    return out
fn_all = lambda ls: {**build_history_features(ls, Promo), 'chain_delta': chain_delta_series(ls).to_frame('chain_delta')}
LEAK_RES = leak_all_origins(fn_all)
print('Kết quả test rò rỉ tại 4 mốc T của 4 fold (True = ĐẠT, mọi biến giống hệt):'); display(pd.Series(LEAK_RES).to_frame('ĐẠT?'))
assert all(LEAK_RES.values()), 'Có biến rò rỉ dữ liệu sau mốc T - dừng lại và sửa trước khi huấn luyện'

# %% [markdown cell 66] **Nhận xét F04 (đọc kết quả trên):**
# %% [markdown cell 67] ### 11.3 Kiểm tra Tương quan (Correlation Check) & Biện luận Đa cộng tuyến

# %% [code cell 68]
# 11.3 Biểu đồ Ma trận Tương quan (Correlation Heatmap) giữa các biến chính
key_feats = [
    "dow_promo_mean", "roll28", "roll91", "ly364", "ly728",
    "Promo", "IsPromoWeek", "Promo_prev7", "Promo_next7",
    "DayOfWeek", "DaysToMonthEnd", "DayOfMonth", "WeekOfYear",
    "chain_delta", "CompetitionDistance", "Store"
]
corr_sub = TB.loc[TB.train_ok, key_feats].sample(min(10000, len(TB)), random_state=SEED).corr()

fig, ax = plt.subplots(figsize=(11, 9))
cax = ax.matshow(corr_sub, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(cax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(key_feats))); ax.set_yticks(range(len(key_feats)))
ax.set_xticklabels(key_feats, rotation=45, ha="left", fontsize=9)
ax.set_yticklabels(key_feats, fontsize=9)
for i in range(len(key_feats)):
    for j in range(len(key_feats)):
        val = corr_sub.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.55 else "black", fontsize=8)
ax.set_title("Feature Correlation Matrix (Key Drivers)", fontsize=12, fontweight="bold", pad=25)
plt.tight_layout(); plt.show()

# In các cặp tương quan mạnh nhất
corr_pairs = []
for i in range(len(key_feats)):
    for j in range(i+1, len(key_feats)):
        r = corr_sub.iloc[i, j]
        if abs(r) >= 0.70:
            corr_pairs.append((key_feats[i], key_feats[j], r))
print("Các cặp biến tương quan mạnh nhất (|r| >= 0.70):")
for f1, f2, r in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"  - {f1:<22} vs {f2:<22}: r = {r:+.2f}")


# %% [markdown cell 69] ---

# %% [code cell 70]
FOLD_NAMES = list(FOLD_ORIGINS)
def fold_masks(T0):
    T1 = T0 + pd.Timedelta(days=HORIZON)
    m_tr = (TB.train_ok & (TB.Date <= T0)).values
    m_va = ((TB.is_test == 0) & (TB.Date > T0) & (TB.Date <= T1) & TB.Store.isin(test_stores)).values
    return m_tr, m_va

meta_cols = ['Store', 'Date', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'StoreType', 'Assortment', 'Promo2',
             'IsPromo2ActiveThisMonth', 'has_ly364', 'MonthsSinceCompetitionOpened', 'CompetitionSinceUnknown', 'Sales']
VAL = []
for name, T0 in FOLD_ORIGINS.items():
    _, m_va = fold_masks(T0)
    v = TB.loc[m_va, meta_cols].copy(); v['fold'] = name; v['h'] = (v.Date - T0).dt.days     # h = ngày dự báo thứ mấy (1..48)
    VAL.append(v)
VAL = pd.concat(VAL, ignore_index=True)
ROWS = {f: np.where(VAL.fold.values == f)[0] for f in FOLD_NAMES}
valid_val = ((VAL.Open == 1) & (VAL.Sales > 0)).values

summary = VAL.assign(chấm_điểm=valid_val).groupby('fold').agg(số_dòng=('Store', 'size'), số_dòng_chấm=('chấm_điểm', 'sum'), số_CH=('Store', 'nunique'))
print('Tập validation (đã loại khỏi quá trình học của chính fold đó):'); display(summary.loc[FOLD_NAMES])

PRED = {}            # PRED[model] = vector dự báo trên thang log, cùng thứ tự với VAL
TIMES = {}
def score_table(models, c_map=None, mask=None, decimals=4):
    c_map = c_map or {}; out = {}
    for m_ in models:
        ps = postprocess(PRED[m_], VAL.Open.values, c_map.get(m_, 1.0)); res = {}
        for f in FOLD_NAMES:
            mk = (VAL.fold.values == f) & (mask if mask is not None else True)
            res[f] = rmspe(VAL.Sales.values[mk], ps[mk])
        vals = [res[f] for f in FOLD_NAMES]; res['Trung bình'] = float(np.mean(vals)); res['Độ lệch chuẩn'] = float(np.std(vals))
        out[m_] = res
    return pd.DataFrame(out).T.round(decimals)

# %% [markdown cell 71] ### 12.2 Kiểm tra A/B: hai nhóm biến bổ sung từ EDA (R15, R16) có thật sự giúp không?

# %% [code cell 72]
DEFAULT_GB = dict(n_estimators=(60 if QUICK else 300), learning_rate=0.10, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0)

class GB:
    # Bọc LightGBM (ưu tiên) hoặc HistGradientBoosting (dự phòng) để dùng chung một giao diện.
    def __init__(self, params, use_store=True, seed=SEED, feats=None):
        self.p, self.seed = dict(params), seed
        base = list(feats) if feats is not None else list(FEATURES)
        self.feats = base if (use_store and HAS_LGB) else [f for f in base if f != 'Store']
        self.cats = [c for c in CAT_FEATURES if c in self.feats]
    def fit(self, df, y):
        if HAS_LGB:
            self.m = lgb.LGBMRegressor(**self.p, n_jobs=N_JOBS, verbose=-1, random_state=self.seed)
            self.m.fit(df[self.feats], y, categorical_feature=self.cats)
        else:
            p = self.p; mask = [c in self.cats for c in self.feats]
            self.m = HistGradientBoostingRegressor(max_iter=p['n_estimators'], learning_rate=p['learning_rate'], max_leaf_nodes=p['num_leaves'],
                                                   min_samples_leaf=p['min_child_samples'], l2_regularization=p.get('reg_lambda', 0.0),
                                                   categorical_features=mask, random_state=self.seed)
            self.m.fit(df[self.feats], y)
        return self
    def predict(self, df): return self.m.predict(df[self.feats])

def gb_fold_scores(feats, seed=SEED, params=None, train_stores=None, use_store=True):
    # RMSPE (thang gốc, c = 1) của GB trên 4 fold với bộ biến `feats`; train_stores = chỉ huấn luyện trên các cửa hàng này (nếu có)
    params = params or DEFAULT_GB; res = {}
    for name, T0 in FOLD_ORIGINS.items():
        m_tr, m_va = fold_masks(T0)
        if train_stores is not None: m_tr = m_tr & TB.Store.isin(train_stores).values
        tr, va = TB.loc[m_tr], TB.loc[m_va]
        model = GB(params, use_store=use_store, seed=seed, feats=feats).fit(tr, tr.y_log.values)
        res[name] = rmspe(va.Sales.values, postprocess(model.predict(va), va.Open.values))
    return res

FEATURES_V0 = list(FEATURES)
FEATURES_V1 = FEATURES_V0 + ['DaysSinceReopen', 'LastClosureLen']
FEATURES_V2 = FEATURES_V1 + ['chain_delta']
VARIANTS = {'V0 bộ biến gốc (35)': FEATURES_V0, 'V1 + đóng cửa dài (R15)': FEATURES_V1, 'V2 + mùa vụ chuỗi (R16)': FEATURES_V2}
t0 = time.time()
AB = {vn: gb_fold_scores(fs) for vn, fs in VARIANTS.items()}
V0_NAME = 'V0 bộ biến gốc (35)'
seed_runs = {SEED: AB[V0_NAME]}
for sd in ([SEED + 1] if QUICK else [SEED + 1, SEED + 2]): seed_runs[sd] = gb_fold_scores(FEATURES_V0, seed=sd)
ab_df = pd.DataFrame(AB).T[FOLD_NAMES]; ab_df['Trung bình'] = ab_df.mean(axis=1)
seed_df = pd.DataFrame(seed_runs).T[FOLD_NAMES]; seed_df['Trung bình'] = seed_df.mean(axis=1); seed_df.index = [f'V0, seed {s}' for s in seed_df.index]
NOISE_SD = float(seed_df['Trung bình'].std(ddof=1))
best_mean = ab_df['Trung bình'].min()
CHOSEN = next(vn for vn in VARIANTS if ab_df.loc[vn, 'Trung bình'] <= best_mean + NOISE_SD)
print(f'A/B xong trong {time.time()-t0:.0f}s'); print('RMSPE theo bộ biến (GB mặc định, seed ' + str(SEED) + '):'); display(ab_df.round(4))
print('Mức nhiễu do hạt giống (V0 chạy nhiều seed):'); display(seed_df.round(4))
print(f'Độ lệch chuẩn nhiễu của RMSPE trung bình = {NOISE_SD:.4f}')
print(f'=> Bộ biến được chọn: {CHOSEN}  (bộ tốt nhất {ab_df["Trung bình"].idxmin()} = {best_mean:.4f}; bộ đơn giản nhất nằm trong 1 độ lệch chuẩn nhiễu của bộ tốt nhất)')
FEATURES = list(VARIANTS[CHOSEN]); FEATURES_FINAL = FEATURES
print(len(FEATURES), 'biến được dùng cho mọi mô hình ở các mục sau.')


# %% [code cell 73]
# Lưu mẫu cho Orange (P06) với bộ biến đã chốt
orange_stores = stratified_store_sample()
orange_df = TB[TB.Store.isin(orange_stores) & (TB.is_test == 0)][['Store', 'Date'] + FEATURES[1:] + ['Open', 'Sales', 'y_log']]
orange_df.to_csv(OUT_DIR / 'F04_orange_sample.csv', index=False)
print(f'Mẫu cho Orange: {len(orange_stores)} cửa hàng, {len(orange_df):,} dòng x {orange_df.shape[1]} cột -> F04_orange_sample.csv')
print('Phân bố StoreType trong mẫu:', store[store.Store.isin(orange_stores)].StoreType.value_counts().to_dict())

# %% [markdown cell 74] **Nhận xét A/B (R15, R16):**
# %% [markdown cell 75] ### 12.3 Baseline (mức sàn): B1, B2, B3

# %% [code cell 76]
def chain(*arrs, default):
    out = np.array(arrs[0], float).copy()
    for a in arrs[1:]: out = np.where(np.isnan(out), np.asarray(a, float), out)
    return np.where(np.isnan(out), default, out)
GLOBAL_MED = float(TB.loc[TB.train_ok, 'y_log'].median())
cols_idx = pd.Index(cols)
fallback_share = {}
for name, T0 in FOLD_ORIGINS.items():
    _, m_va = fold_masks(T0); va = TB.loc[m_va]; r = ROWS[name]
    vi = ALL_DATES[(ALL_DATES > T0) & (ALL_DATES <= T0 + pd.Timedelta(days=HORIZON))]
    rp, cp = vi.get_indexer(va.Date), cols_idx.get_indexer(va.Store)
    b1 = profile_pred(T0, vi, cols, 8,  False)[rp, cp]
    b2 = profile_pred(T0, vi, cols, 16, True)[rp, cp]
    b3 = va.dow_promo_mean.values
    for k, v in {'B1 cùng thứ': chain(b1, b3, va.roll91.values, default=GLOBAL_MED),
                 'B2 cùng thứ + Promo': chain(b2, b1, b3, va.roll91.values, default=GLOBAL_MED),
                 'B3 dow_promo_mean (lag>=49)': chain(b3, b2, b1, va.roll91.values, default=GLOBAL_MED)}.items():
        PRED.setdefault(k, np.full(len(VAL), np.nan))[r] = v
    mk = (va.Open.values == 1) & (va.Sales.values > 0)
    fallback_share[name] = round(float(np.isnan(b2[mk]).mean()), 3)
print('Tỷ lệ dòng B2 phải lùi về phương án dự phòng, theo fold:', fallback_share)
BASELINES = ['B1 cùng thứ', 'B2 cùng thứ + Promo', 'B3 dow_promo_mean (lag>=49)']
score_table(BASELINES)

# %% [markdown cell 77] ### 12.4 Mô hình ML với tham số mặc định: Linear (L1), Random Forest (M1), Gradient Boosting (M2)

# %% [code cell 78]
NOSTORE = [f for f in FEATURES if f != 'Store']
def rf_fill(df):
    d = df[NOSTORE].copy()
    d['dow_promo_mean'] = d.dow_promo_mean.fillna(d.roll91).fillna(d.roll28).fillna(GLOBAL_MED)
    d['roll91'] = d.roll91.fillna(d.dow_promo_mean); d['roll28'] = d.roll28.fillna(d.roll91)
    d['ly364']  = d.ly364.fillna(d.dow_promo_mean);  d['ly728']  = d.ly728.fillna(d.dow_promo_mean)
    return d.fillna(-1).astype('float32')

NUM_L1 = [f for f in NOSTORE if f not in ('DayOfWeek', 'Month', 'StateHoliday', 'StoreType', 'Assortment')]
CAT_L1 = ['DayOfWeek', 'Month', 'StateHoliday', 'StoreType', 'Assortment']
def make_ridge(alpha=1.0):
    try: oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError: oh = OneHotEncoder(handle_unknown='ignore', sparse=False)
    pre = ColumnTransformer([('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), NUM_L1), ('cat', oh, CAT_L1)])
    return make_pipeline(pre, Ridge(alpha=alpha))

RF_PARAMS = dict(n_estimators=(20 if QUICK else 60), max_samples=0.10, min_samples_leaf=10, max_features=0.5)
def run_models(spec, store_filter=None, verbose=True):
    # spec: {tên_model: hàm tạo model có fit/predict và hàm chuẩn bị X}; trả về thời gian
    for name, T0 in FOLD_ORIGINS.items():
        m_tr, m_va = fold_masks(T0)
        if store_filter is not None:
            m_tr = m_tr & TB.Store.isin(store_filter).values; m_va = m_va & TB.Store.isin(store_filter).values
        tr, va = TB.loc[m_tr], TB.loc[m_va]; r = ROWS[name]
        va_rows = np.where(m_va)[0]                                       # vị trí trong TB
        for mname, (mk_model, prep) in spec.items():
            t0 = time.time(); model = mk_model(); model.fit(prep(tr), tr.y_log.values)
            pred = model.predict(prep(va))
            full = PRED.setdefault(mname, np.full(len(VAL), np.nan))
            if store_filter is None: full[r] = pred
            TIMES[(mname, name)] = round(time.time() - t0, 1)
            if verbose: print(f'  {mname:<22} {name:<26} {TIMES[(mname, name)]:>6.1f}s')

spec_default = {
    'L1 Ridge (log)':        (lambda: make_ridge(1.0), lambda d: d[NOSTORE]),
    'M1 Random Forest':      (lambda: RandomForestRegressor(**RF_PARAMS, n_jobs=N_JOBS, random_state=SEED), rf_fill),
    'M2 Gradient Boosting':  (lambda: GB(DEFAULT_GB, use_store=True), lambda d: d),
    'M2b GB không dùng Store': (lambda: GB(DEFAULT_GB, use_store=False), lambda d: d),
}
run_models(spec_default)
MODELS_DEFAULT = ['L1 Ridge (log)', 'M1 Random Forest', 'M2 Gradient Boosting', 'M2b GB không dùng Store']
score_table(BASELINES + MODELS_DEFAULT)

# %% [markdown cell 79] ### 12.5 Tuning (chỉnh tham số)

# %% [code cell 80]
TUNE_STORES = stratified_store_sample(n_total=(60 if QUICK else 200), seed=SEED + 1)
print('Mẫu tuning:', len(TUNE_STORES), 'cửa hàng | phân bố StoreType:', store[store.Store.isin(TUNE_STORES)].StoreType.value_counts().to_dict())

SPACE = {'num_leaves': [31, 63, 127, 255], 'learning_rate': [0.05, 0.08, 0.10, 0.15], 'min_child_samples': [20, 50, 100, 200],
         'colsample_bytree': [0.6, 0.8, 1.0], 'subsample': [0.6, 0.8, 1.0], 'reg_lambda': [0.0, 1.0, 5.0, 10.0], 'n_estimators': [200, 300, 500]}
N_ITER = 3 if QUICK else 6
r_ = np.random.default_rng(SEED)
CONFIGS = [('default', dict(DEFAULT_GB))]
for i in range(N_ITER):
    cfg = {k: v[r_.integers(len(v))] for k, v in SPACE.items()}; cfg['subsample_freq'] = 1; CONFIGS.append((f'rs{i+1}', cfg))
if QUICK: [c[1].update(n_estimators=min(c[1]['n_estimators'], 80)) for c in CONFIGS]

def eval_on_sample(cfg, store_sample):
    # huấn luyện trên mẫu cửa hàng, chấm RMSPE trung bình 4 fold trên chính mẫu đó (mọi thứ nằm trong mốc thời gian của từng fold)
    scores = []
    for name, T0 in FOLD_ORIGINS.items():
        m_tr, m_va = fold_masks(T0); sm = TB.Store.isin(store_sample).values
        tr, va = TB.loc[m_tr & sm], TB.loc[m_va & sm]
        model = GB(cfg).fit(tr, tr.y_log.values)
        p = postprocess(model.predict(va), va.Open.values)
        scores.append(rmspe(va.Sales.values, p))
    return float(np.mean(scores)), [round(s, 4) for s in scores]

tune_rows = []
t0 = time.time()
for nm, cfg in CONFIGS:
    s, per = eval_on_sample(cfg, TUNE_STORES)
    tune_rows.append({'cấu hình': nm, 'RMSPE TB 4 fold (mẫu)': round(s, 4), 'từng fold': per, **{k: cfg[k] for k in SPACE}})
tune_df = pd.DataFrame(tune_rows).sort_values('RMSPE TB 4 fold (mẫu)').reset_index(drop=True)
print(f'Tuning xong trong {time.time()-t0:.0f}s')
display(tune_df)
BEST_NAME = tune_df.loc[0, 'cấu hình']; BEST_GB = dict(dict(CONFIGS)[BEST_NAME])
print('Cấu hình GB tốt nhất trên mẫu:', BEST_NAME, {k: BEST_GB[k] for k in SPACE})


# %% [code cell 81]
# Tuning nhỏ cho Random Forest (3 cấu hình) trên cùng mẫu cửa hàng
RF_CONFIGS = [dict(RF_PARAMS), dict(RF_PARAMS, min_samples_leaf=5), dict(RF_PARAMS, min_samples_leaf=20, max_features=0.7)]
if QUICK: RF_CONFIGS = RF_CONFIGS[:2]
rf_rows = []
for i, cfg in enumerate(RF_CONFIGS):
    sc = []
    for name, T0 in FOLD_ORIGINS.items():
        m_tr, m_va = fold_masks(T0); sm = TB.Store.isin(TUNE_STORES).values
        tr, va = TB.loc[m_tr & sm], TB.loc[m_va & sm]
        rf = RandomForestRegressor(**cfg, n_jobs=N_JOBS, random_state=SEED).fit(rf_fill(tr), tr.y_log.values)
        sc.append(rmspe(va.Sales.values, postprocess(rf.predict(rf_fill(va)), va.Open.values)))
    rf_rows.append({'cấu hình RF': i, 'min_samples_leaf': cfg['min_samples_leaf'], 'max_features': cfg['max_features'], 'RMSPE TB 4 fold (mẫu)': round(float(np.mean(sc)), 4)})
rf_df = pd.DataFrame(rf_rows).sort_values('RMSPE TB 4 fold (mẫu)').reset_index(drop=True); display(rf_df)
BEST_RF = RF_CONFIGS[int(rf_df.loc[0, 'cấu hình RF'])]

# %% [markdown cell 82] ### 12.6 Huấn luyện lại bản tốt nhất trên toàn bộ cửa hàng + hệ số hiệu chỉnh `c` (R13.2) + bản kết 

# %% [code cell 83]
spec_tuned = {
    'M1 Random Forest (tuned)':     (lambda: RandomForestRegressor(**BEST_RF, n_jobs=N_JOBS, random_state=SEED), rf_fill),
    'M2 Gradient Boosting (tuned)': (lambda: GB(BEST_GB, use_store=True), lambda d: d),
}
run_models(spec_tuned)
PRED['M3 Kết hợp RF + GB (tuned)'] = 0.5 * PRED['M1 Random Forest (tuned)'] + 0.5 * PRED['M2 Gradient Boosting (tuned)']   # trung bình trên thang log

# hệ số hiệu chỉnh c: 1 tham số duy nhất, chọn theo RMSPE trung bình 4 fold
grid = np.round(np.arange(0.96, 1.0401, 0.005), 3)
def mean_rmspe_for_c(m_, c_): return score_table([m_], {m_: c_}).loc[m_, 'Trung bình']
C_BEST = {}
for m_ in ['M1 Random Forest (tuned)', 'M2 Gradient Boosting (tuned)', 'M3 Kết hợp RF + GB (tuned)']:
    vals = [mean_rmspe_for_c(m_, c_) for c_ in grid]; C_BEST[m_] = float(grid[int(np.argmin(vals))])
    print(f'{m_:<32} c tốt nhất = {C_BEST[m_]:.3f} | RMSPE TB: c=1.000 -> {mean_rmspe_for_c(m_, 1.0):.4f}  ->  c={C_BEST[m_]:.3f}: {min(vals):.4f}')

# %% [markdown cell 84] ### 12.7 Bảng so sánh tổng hợp (trên validation)

# %% [code cell 85]
ALL_MODELS = BASELINES + MODELS_DEFAULT + ['M1 Random Forest (tuned)', 'M2 Gradient Boosting (tuned)', 'M3 Kết hợp RF + GB (tuned)']
final = score_table(ALL_MODELS, C_BEST)
b2 = final.loc['B2 cùng thứ + Promo', FOLD_NAMES]
final['Cải thiện so với B2 (TB)'] = ((final['Trung bình'] / final.loc['B2 cùng thứ + Promo', 'Trung bình'] - 1) * 100).round(1).astype(str) + '%'
final['Số fold thắng B2'] = [(final.loc[m_, FOLD_NAMES] < b2).sum() for m_ in final.index]
display(final)

MODEL_MAP_EN = {
    'B1 cùng thứ': 'B1 DOW Mean',
    'B2 cùng thứ + Promo': 'B2 DOW + Promo',
    'B3 dow_promo_mean (lag>=49)': 'B3 Historical Profile',
    'L1 Ridge (log)': 'L1 Ridge Regression',
    'M1 Random Forest': 'M1 Random Forest (default)',
    'M2 Gradient Boosting': 'M2 Gradient Boosting (default)',
    'M2b GB không dùng Store': 'M2b GB without Store ID',
    'M1 Random Forest (tuned)': 'M1 Random Forest (tuned)',
    'M2 Gradient Boosting (tuned)': 'M2 Gradient Boosting (tuned)',
    'M3 Kết hợp RF + GB (tuned)': 'M3 Ensemble RF+GB (tuned)'
}
FOLDS_EN = {
    'F1 (gần test nhất)': 'F1',
    'F2': 'F2',
    'F3 (có lễ Phục sinh)': 'F3',
    'F4 (cùng mùa năm trước)': 'F4'
}
plot_final = final[FOLD_NAMES].rename(index=MODEL_MAP_EN, columns=FOLDS_EN)
fig, ax = plt.subplots(figsize=(11, 4.2))
plot_final.plot(kind='bar', ax=ax, width=0.85)
ax.set_ylabel('Validation RMSPE')
ax.set_title('Model Performance Across Folds')
ax.legend(title='Validation Fold', fontsize=8)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()


# %% [markdown cell 86] **Nhận xét M02 (đọc bảng ở trên):**
# %% [markdown cell 87] ---

# %% [code cell 88]
CANDIDATES_FINAL = ['M1 Random Forest (tuned)', 'M2 Gradient Boosting (tuned)', 'M3 Kết hợp RF + GB (tuned)']
BEST_MODEL = final.loc[CANDIDATES_FINAL, 'Trung bình'].astype(float).idxmin()
BEST_C = C_BEST[BEST_MODEL]
print(f'Mô hình tốt nhất theo RMSPE trung bình 4 fold: {BEST_MODEL} (hệ số c = {BEST_C})')

VAL['valid'] = valid_val
VAL['pred_best'] = postprocess(PRED[BEST_MODEL], VAL.Open.values, BEST_C)
VAL['pred_B2']   = postprocess(PRED['B2 cùng thứ + Promo'], VAL.Open.values, 1.0)
VAL['StoreTypeL'] = VAL.StoreType.map(dict(enumerate(CATS['StoreType']))); VAL['AssortmentL'] = VAL.Assortment.map(dict(enumerate(CATS['Assortment'])))
VAL['holiday'] = (VAL.StateHoliday > 0).astype(int)

show = ['B2 cùng thứ + Promo', 'M2 Gradient Boosting (tuned)', BEST_MODEL]
tab = score_table(list(dict.fromkeys(show)), C_BEST)
print('RMSPE trên validation (tất cả dòng):'); display(tab)
print('RMSPE trên validation, loại các dòng ngày lễ nhà nước (để giống test hơn - R12.3):')
display(score_table(list(dict.fromkeys(show)), C_BEST, mask=(VAL.StateHoliday.values == 0)))

# %% [markdown cell 89] ### 13.2 Sai số theo từng nhóm: model yếu ở đâu?

# %% [code cell 90]
# cờ bổ sung: cửa hàng thiếu dữ liệu 2014, đối thủ mới mở (<=6 tháng), cửa hàng vừa đóng cửa dài (>=7 ngày không phải Chủ nhật trong 28 ngày trước mốc T)
closed_ns = ((O == 0) & (ALL_DATES.dayofweek != 6)[:, None]).astype(float)
recent_closure = {}
for name, T0 in FOLD_ORIGINS.items():
    w = closed_ns.loc[T0 - pd.Timedelta(days=27):T0].sum()
    recent_closure[name] = set(w[w >= 7].index)
VAL['gap_store'] = VAL.Store.isin(gap_stores).astype(int)
VAL['new_competitor'] = ((VAL.MonthsSinceCompetitionOpened <= 6) & (VAL.CompetitionSinceUnknown == 0)).astype(int)
VAL['recent_closure'] = [int(s in recent_closure[f]) for s, f in zip(VAL.Store.values, VAL.fold.values)]
VAL['h_bucket'] = pd.cut(VAL.h, [0, 7, 14, 21, 28, 35, 42, 48], labels=['1-7', '8-14', '15-21', '22-28', '29-35', '36-42', '43-48'])

def _rm(g, col): 
    y = g.Sales.values; p = g[col].values; return np.sqrt(np.mean(((y - p) / y) ** 2))
def by_group(col, label=None):
    g = VAL[VAL.valid].groupby(col, observed=True)
    out = pd.DataFrame({'n': g.size(), 'CH': g.Store.nunique(), 'best': g.apply(lambda x: _rm(x, 'pred_best')), 'B2': g.apply(lambda x: _rm(x, 'pred_B2'))})
    out['best/B2'] = (out.best / out.B2).round(2); out.index.name = label or col
    return out.round(4)
for col, lab in [('StoreTypeL', 'StoreType'), ('AssortmentL', 'Assortment'), ('DayOfWeek', 'Thứ (1=T2..7=CN)'), ('Promo', 'Promo'),
                 ('IsPromo2ActiveThisMonth', 'Promo2 đang active'), ('holiday', 'Ngày lễ nhà nước'), ('h_bucket', 'Ngày dự báo thứ h'),
                 ('gap_store', 'CH thiếu dữ liệu 2014'), ('has_ly364', 'Có ly364 (cùng kỳ năm trước)'), ('new_competitor', 'Đối thủ mới mở <= 6 tháng'),
                 ('recent_closure', 'Vừa đóng cửa dài (>=7 ngày) trước mốc T')]:
    print(f'--- Theo {lab}'); display(by_group(col, lab))

# %% [markdown cell 91] ### 13.3 Đánh giá sai số tích lũy theo chiều dài khoảng dự báo (Cumulative Forecast Horizon Analysis

# %% [code cell 92]
# Đánh giá sai số RMSPE tích lũy theo từng mốc Horizon (Cumulative Forecast Horizon Analysis)
# Tính toán trực tiếp từ kết quả dự báo của mô hình tốt nhất (pred_best) trên tập validation qua 4 fold

horizons = {
    '7 Days (1 Week)': 7,
    '14 Days (2 Weeks)': 14,
    '28 Days (4 Weeks)': 28,
    '48 Days (Full Cycle)': 48
}

horizon_fold_scores = {}
horizon_scores = {}

for label, max_h in horizons.items():
    scores_per_fold = {}
    for f in FOLD_NAMES:
        mask = VAL.valid & (VAL.fold == f) & (VAL.h <= max_h)
        score = rmspe(VAL.loc[mask, 'Sales'].values, VAL.loc[mask, 'pred_best'])
        scores_per_fold[f] = score
    horizon_fold_scores[label] = scores_per_fold
    horizon_scores[label] = float(np.mean(list(scores_per_fold.values())))

# 1. Vẽ biểu đồ so sánh RMSPE theo Horizon
fig, ax = plt.subplots(figsize=(8.5, 4.4))
min_val = min(horizon_scores.values())
colors = ['#55A868' if v == min_val else '#4C72B0' for v in horizon_scores.values()]
bars = ax.bar(list(horizon_scores.keys()), list(horizon_scores.values()), width=0.52, color=colors)

ax.set_ylabel('Validation RMSPE (Mean 4 Folds)')
ax.set_title('Forecast Performance Across Cumulative Horizons', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(horizon_scores.values()) * 1.25)
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# 2. Hiển thị bảng chi tiết điểm từng fold và điểm trung bình
df_horizon = pd.DataFrame(horizon_fold_scores).T[FOLD_NAMES]
df_horizon['Mean RMSPE'] = list(horizon_scores.values())
print('Bảng chi tiết sai số RMSPE theo Horizon qua 4 fold:')
display(df_horizon.round(4))


# %% [markdown cell 93] ### 13.4 Bias (lệch hệ thống) và mức độ tập trung của sai số

# %% [code cell 94]
v = VAL[VAL.valid].copy(); v['pe'] = (v.pred_best - v.Sales) / v.Sales; v['se'] = v.pe ** 2
bias = v.groupby('fold').apply(lambda x: pd.Series({'Bias tổng (dự báo/thực tế - 1)': x.pred_best.sum() / x.Sales.sum() - 1,
                                                    'Trung vị sai số %': x.pe.median(), 'Trung vị |sai số|%': x.pe.abs().median()})).loc[FOLD_NAMES]
display((bias * 100).round(2).astype(str) + '%')
v_sorted = v.sort_values('se', ascending=False)
top_n = max(int(len(v) * 0.005), 1)
print(f'0,5% dòng sai nhiều nhất ({top_n} dòng) chiếm {v_sorted.se.head(top_n).sum() / v.se.sum() * 100:.1f}% tổng bình phương sai số.')
print(f'RMSPE nếu bỏ 0,5% dòng sai nhiều nhất: {np.sqrt(v_sorted.se.iloc[top_n:].mean()):.4f} (so với {np.sqrt(v.se.mean()):.4f} khi giữ nguyên)')

cols_show = ['fold', 'Store', 'Date', 'DayOfWeek', 'Promo', 'Sales', 'pred_best', 'pe', 'has_ly364', 'recent_closure', 'gap_store']
print('\n15 dòng sai nhiều nhất:'); display(v_sorted[cols_show].head(15).assign(pe=lambda d: (d.pe * 100).round(0).astype(int).astype(str) + '%').round(0))

st_err = v.groupby('Store').agg(n=('se', 'size'), RMSPE=('se', lambda s: np.sqrt(s.mean())), doanh_số_TB=('Sales', 'mean'), loại=('StoreTypeL', 'first'),
                                CH_thiếu_2014=('gap_store', 'max'), vừa_đóng_cửa=('recent_closure', 'max'))
print('\n15 cửa hàng sai nhiều nhất (>=100 dòng chấm):'); display(st_err[st_err.n >= 100].sort_values('RMSPE', ascending=False).head(15).round(3))

# %% [markdown cell 95] ### 13.5 Quy sang đơn vị kinh doanh (phục vụ D02)

# %% [code cell 96]
within = pd.DataFrame({'±10%': (v.pe.abs() <= 0.10).groupby(v.fold).mean(), '±20%': (v.pe.abs() <= 0.20).groupby(v.fold).mean()}).loc[FOLD_NAMES]
within.loc['Gộp 4 fold'] = [(v.pe.abs() <= 0.10).mean(), (v.pe.abs() <= 0.20).mean()]
print('Tỷ lệ dòng (ngày-cửa hàng) có sai số tuyệt đối trong ngưỡng:'); display((within * 100).round(1).astype(str) + '%')

v['week'] = v.Date.dt.to_period('W-SUN')
wk = v.groupby(['fold', 'Store', 'week']).agg(act=('Sales', 'sum'), pred=('pred_best', 'sum'), nd=('Sales', 'size')).reset_index()
wk = wk[wk.nd >= 5]                                        # chỉ tuần có >= 5 ngày mở cửa được chấm
wk['ape'] = (wk.pred - wk.act).abs() / wk.act
print(f"Gộp theo tuần (tuần có >=5 ngày mở cửa, {len(wk):,} tuần-cửa hàng): sai số % trung vị = {wk.ape.median()*100:.1f}% | trong ±10%: {(wk.ape <= 0.10).mean()*100:.1f}% | trong ±20%: {(wk.ape <= 0.20).mean()*100:.1f}%")

# %% [markdown cell 97] ### 13.6 Biểu đồ thực tế vs dự báo

# %% [code cell 98]
f1 = v[v.fold == FOLD_NAMES[0]]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].hexbin(np.log1p(f1.Sales), np.log1p(f1.pred_best), gridsize=60, bins='log', cmap='Blues')
lim = [np.log1p(f1.Sales).min(), np.log1p(f1.Sales).max()]
axs[0].plot(lim, lim, 'r--', lw=1)
axs[0].set_xlabel('Actual log(1+Sales)')
axs[0].set_ylabel('Predicted log(1+Sales)')
axs[0].set_title('Actual vs Predicted Sales')

axs[1].hist((v.pe * 100).clip(-60, 60), bins=80, color='#4472C4')
axs[1].axvline(0, color='r', ls='--')
axs[1].set_title('Percentage Error Distribution')
axs[1].set_xlabel('Percentage Error (%)')

g_ = by_group('StoreTypeL', 'StoreType')[['best', 'B2']].rename(columns={'best': 'Best Model', 'B2': 'B2 Baseline'})
g_.plot(kind='bar', ax=axs[2], rot=0)
axs[2].set_title('RMSPE by StoreType')
axs[2].set_ylabel('RMSPE')
axs[2].legend(fontsize=8)
plt.tight_layout()
plt.show()


# %% [markdown cell 99] ### 13.7 Checklist "model có hợp lý không" (tự động, theo mục 10.2)

# %% [code cell 100]
b2_tab, best_tab = score_table(['B2 cùng thứ + Promo'], C_BEST).loc['B2 cùng thứ + Promo'], score_table([BEST_MODEL], C_BEST).loc[BEST_MODEL]
gain = best_tab['Trung bình'] / b2_tab['Trung bình'] - 1
h_ratio = hh['best'].loc[25:].mean() / hh['best'].loc[:24].mean()
checks = [
 ('1. Thắng baseline B2 ở CẢ 4 fold',                       all(best_tab[f] < b2_tab[f] for f in FOLD_NAMES), {f: f'{best_tab[f]:.4f} vs {b2_tab[f]:.4f}' for f in FOLD_NAMES}),
 ('2. Không "tốt đến đáng ngờ" (cải thiện < 50% so với B2)',  gain > -0.5,                                      f'cải thiện {abs(gain)*100:.1f}%'),
 ('3. Sai số không tăng theo ngày dự báo (nửa sau / nửa đầu < 1.15)', h_ratio < 1.15,                           f'tỷ lệ = {h_ratio:.3f}'),
 ('4. Trung vị sai số % mỗi fold nằm trong ±5% (không lệch hệ thống)', bool((bias.iloc[:, 1].abs() < 0.05).all()), bias.iloc[:, 1].round(4).to_dict()),
 ('5. Không có dự báo âm hoặc NaN',                           bool((VAL.pred_best >= 0).all() and VAL.pred_best.notna().all()), 'ok'),
 ('6. Không có biến bị cấm (Customers/Sales thô/Id) trong FEATURES', not (set(FEATURES) & FORBIDDEN),         'ok'),
 ('7. Độ lệch giữa các fold hợp lý (std/mean < 0.25)',        best_tab['Độ lệch chuẩn'] / best_tab['Trung bình'] < 0.25, f"{best_tab['Độ lệch chuẩn']/best_tab['Trung bình']:.3f}"),
]
for name_, ok_, detail in checks: print(('ĐẠT  ' if ok_ else 'CHƯA ĐẠT ') + name_ + '  ->  ' + str(detail))

# %% [markdown cell 101] ### 13.8 Kiểm tra logic và độ tin cậy trước khi tin kết quả

# %% [code cell 102]
# (1) căn chỉnh dòng giữa VAL, PRED và TB
for f_, T0_ in FOLD_ORIGINS.items():
    _, m_ = fold_masks(T0_); rows_ = np.where(m_)[0]
    assert np.array_equal(TB.Sales.values[rows_], VAL.Sales.values[ROWS[f_]], equal_nan=True) and np.array_equal(TB.Store.values[rows_], VAL.Store.values[ROWS[f_]]), f'Lệch dòng ở {f_}'
print('(1) Căn chỉnh dòng: ĐẠT (VAL khớp TB theo từng fold)')

# (2) và (3): ablation
HIST_FEATS = ['dow_promo_mean', 'dow_promo_n', 'roll28', 'roll91', 'open28', 'promo_uplift91', 'ly364', 'ly728', 'has_ly364', 'chain_delta']
FEAT_NOHIST = [f for f in FEATURES if f not in HIST_FEATS]
t0 = time.time()
ABL = {'Bộ biến đầy đủ (GB mặc định)': score_table(['M2 Gradient Boosting']).loc['M2 Gradient Boosting', FOLD_NAMES].to_dict(),
       'Bỏ toàn bộ biến lịch sử Sales': gb_fold_scores(FEAT_NOHIST),
       'Chỉ huấn luyện trên 856 CH của test': gb_fold_scores(FEATURES, train_stores=test_stores)}
abl_df = pd.DataFrame(ABL).T[FOLD_NAMES]; abl_df['Trung bình'] = abl_df.mean(axis=1)
abl_df['so với đầy đủ'] = ((abl_df['Trung bình'] / abl_df.loc['Bộ biến đầy đủ (GB mặc định)', 'Trung bình'] - 1) * 100).round(1).astype(str) + '%'
print(f'(2)(3) Ablation xong trong {time.time()-t0:.0f}s'); display(abl_df.round(4))


# %% [code cell 103]
# (4) sai số lệch hệ thống theo từng cửa hàng có lặp lại giữa các fold không?
vv = VAL[VAL.valid].copy(); vv['le'] = np.log(vv.pred_best / vv.Sales)                 # sai số log: > 0 = dự báo cao hơn thực tế
sb = vv.groupby(['Store', 'fold']).le.agg(['mean', 'size']).reset_index(); sb = sb[sb['size'] >= 20]
piv = sb.pivot(index='Store', columns='fold', values='mean')[FOLD_NAMES]
print('Độ lệch chuẩn của sai số log trung bình giữa các cửa hàng, theo fold:', piv.std().round(3).to_dict())
cm = piv.corr().round(2); cm.index = cm.columns = ['F1', 'F2', 'F3', 'F4']
print('Tương quan giữa sai số lệch hệ thống của cùng một cửa hàng ở các fold (gần 0 = không lặp lại; gần 1 = lặp lại):'); display(cm)
full4 = piv.dropna()
persist_pos = int(((full4 > 0.05).all(axis=1)).sum()); persist_neg = int(((full4 < -0.05).all(axis=1)).sum())
print(f'Cửa hàng có đủ 4 fold: {len(full4)} | dự báo CAO hơn thực tế >5% ở cả 4 fold: {persist_pos} | THẤP hơn >5% ở cả 4 fold: {persist_neg}')

# (5) dự báo so với mức bán thông thường của chính cửa hàng (trung vị lịch sử các ngày mở cửa trước mốc T)
ratios = []
for f_, T0_ in FOLD_ORIGINS.items():
    med_ = np.expm1(logS.loc[:T0_].median())                                          # trung vị theo cửa hàng (ngày mở cửa)
    d_ = vv[vv.fold == f_]; ratios.append((d_.pred_best / d_.Store.map(med_)).dropna())
ratios = pd.concat(ratios)
print(f'(5) Dự báo / trung vị lịch sử của chính cửa hàng: P1 = {ratios.quantile(.01):.2f}, trung vị = {ratios.median():.2f}, P99 = {ratios.quantile(.99):.2f} | ngoài khoảng [0.3, 3]: {((ratios < 0.3) | (ratios > 3)).mean()*100:.2f}% dòng')

# %% [markdown cell 104] **Nhận xét E02 (đọc kết quả ở trên):**
# %% [markdown cell 105] ---

# %% [code cell 106]
N_SEEDS = 1 if QUICK else 3
m_tr_all = TB.train_ok.values; m_te = (TB.is_test == 1).values
tr_all, te_all = TB.loc[m_tr_all], TB.loc[m_te]
t0 = time.time(); parts = []
if 'RF' in BEST_MODEL or 'Kết hợp' in BEST_MODEL:
    rf_pred = np.mean([RandomForestRegressor(**BEST_RF, n_jobs=N_JOBS, random_state=SEED + s).fit(rf_fill(tr_all), tr_all.y_log.values).predict(rf_fill(te_all)) for s in range(N_SEEDS)], axis=0)
    parts.append(('rf', rf_pred))
if 'Gradient' in BEST_MODEL or 'Kết hợp' in BEST_MODEL:
    gb_pred = np.mean([GB(BEST_GB, use_store=True, seed=SEED + s).fit(tr_all, tr_all.y_log.values).predict(te_all) for s in range(N_SEEDS)], axis=0)
    parts.append(('gb', gb_pred))
pred_log_test = np.mean([p for _, p in parts], axis=0)
pred_test = postprocess(pred_log_test, te_all.Open.values, BEST_C)
SUB = make_submission(te_all.Id.values, pred_test)
SUB.to_csv(OUT_DIR / 'submission_F02_best_model.csv', index=False)
print(f'Đã huấn luyện lại ({time.time()-t0:.0f}s, {N_SEEDS} seed) và ghi submission_F02_best_model.csv: {SUB.shape}')

# kiểm tra trước khi nộp
assert list(SUB.columns) == ['Id', 'Sales'] and set(SUB.Id) == set(test.Id) and SUB.Sales.notna().all() and (SUB.Sales >= 0).all()
tt = te_all.assign(pred=pred_test)
print('Dòng Open=0 được gán Sales=0:', bool((tt[tt.Open == 0].pred == 0).all()), '| số dòng Open=0:', int((tt.Open == 0).sum()))
last48 = TB[(TB.is_test == 0) & (TB.Date > TRAIN_END - pd.Timedelta(days=HORIZON)) & (TB.Open == 1) & (TB.Sales > 0) & TB.Store.isin(test_stores)]
print(f'Doanh số TB mỗi ngày-cửa hàng (dòng mở cửa): dự báo test = {tt[tt.Open == 1].pred.mean():,.0f} | 48 ngày cuối của train = {last48.Sales.mean():,.0f}')

# %% [markdown cell 107] **Khi đã nộp Kaggle**, ghi điểm vào biến bên dưới để so sánh với ước lượng trên validation:

# %% [code cell 108]
KAGGLE_RMSPE = None          # <- điền điểm RMSPE Kaggle trả về (ví dụ 0.123). Giữ None nếu chưa nộp.
lo, hi = best_tab[FOLD_NAMES].min(), best_tab[FOLD_NAMES].max()
print(f'Ước lượng trên validation: trung bình 4 fold = {best_tab["Trung bình"]:.4f}, khoảng {lo:.4f} - {hi:.4f} (F1 {best_tab[FOLD_NAMES[0]]:.4f}, F4 {best_tab[FOLD_NAMES[3]]:.4f})')
if KAGGLE_RMSPE is None: print('Chưa có điểm Kaggle -> chưa kết luận được model có tổng quát hóa tốt trên tệp test hay không.')
else:
    pos = 'TRONG' if lo <= KAGGLE_RMSPE <= hi else 'NGOÀI'
    print(f'Điểm Kaggle = {KAGGLE_RMSPE:.4f} -> {pos} khoảng validation; chênh so với trung bình validation: {KAGGLE_RMSPE - best_tab["Trung bình"]:+.4f}')

# %% [markdown cell 109] ---

# %% [code cell 110]
# JUPYTER_ONLY: %matplotlib inline
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from scipy import stats
plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': 0.3})
assert HAS_LGB, 'Phần xAI dùng TreeSHAP có sẵn của LightGBM (pred_contrib) nên cần cài lightgbm'

XAI_QUICK = False                         # True = chạy nhanh để kiểm tra code (mẫu nhỏ, ít cây)
N_SHAP = 4000 if XAI_QUICK else 8000     # số dòng validation dùng để tính SHAP
N_PERM = 4000 if XAI_QUICK else 8000     # số dòng dùng cho permutation importance
N_REP  = 1    if XAI_QUICK else 2         # số lần xáo trộn cho mỗi biến / nhóm biến
FOLD_XAI = FOLD_NAMES[0]                  # F1: fold gần test nhất
XAI_PARAMS = dict(BEST_GB)
if XAI_QUICK: XAI_PARAMS['n_estimators'] = 60

FEATURE_GROUPS = {
 'Lịch sử Sales (mức nền, cùng kỳ năm trước)': ['dow_promo_mean', 'dow_promo_n', 'roll28', 'roll91', 'open28', 'promo_uplift91', 'ly364', 'ly728', 'has_ly364'],
 'Mùa vụ toàn chuỗi':       ['chain_delta'],
 'Định danh cửa hàng':      ['Store'],
 'Đặc điểm cửa hàng':       ['StoreType', 'Assortment'],
 'Promo ngắn hạn':          ['Promo', 'IsPromoWeek', 'Promo_prev7', 'Promo_next7'],
 'Lịch (thứ, tháng, tuần...)': ['DayOfWeek', 'Month', 'WeekOfYear', 'DayOfMonth', 'DaysToMonthEnd'],
 'Ngày lễ':                 ['StateHoliday', 'SchoolHoliday', 'DaysSinceLastHoliday', 'IsDayBeforeHoliday', 'IsDayAfterHoliday'],
 'Đối thủ':                 ['CompetitionDistance', 'CompetitionDistanceMissing', 'MonthsSinceCompetitionOpened', 'CompetitionSinceUnknown', 'CompetitorNotOpenYet'],
 'Promo2':                  ['Promo2', 'Promo2Started', 'IsPromo2ActiveThisMonth', 'Promo2WeeksSinceStart'],
 'Đóng cửa dài ngày':       ['DaysSinceReopen', 'LastClosureLen'],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES), 'FEATURE_GROUPS phải phủ đúng toàn bộ FEATURES'
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
DOW_NAME = {1: 'T2', 2: 'T3', 3: 'T4', 4: 'T5', 5: 'T6', 6: 'T7', 7: 'CN'}

def fit_gb_df(tr, feats, params=None, seed=SEED):
    return GB(params or XAI_PARAMS, use_store=('Store' in feats), seed=seed, feats=feats).fit(tr, tr.y_log.values)

def fold_data(fold):
    # train (<= T) và validation (48 ngày sau T, cửa hàng của test, chỉ dòng được chấm: Open=1 và Sales>0)
    T0 = FOLD_ORIGINS[fold]; m_tr, m_va = fold_masks(T0); tr, va = TB.loc[m_tr], TB.loc[m_va]
    return tr, va[(va.Open == 1) & (va.Sales > 0)]

def score_pred(pred_log, df): return rmspe(df.Sales.values, postprocess(pred_log, df.Open.values))     # RMSPE thang gốc, c = 1

def perm_imp(predict, X, meta, units, n_rep=N_REP, seed=0):
    # Permutation importance: xáo trộn (đồng thời) các cột của từng "đơn vị" (biến hoặc nhóm biến) rồi đo RMSPE tăng bao nhiêu.
    r = np.random.default_rng(seed); base = score_pred(predict(X), meta); out = {}
    for u, cs in units.items():
        ds = []
        for _ in range(n_rep):
            Xp = X.copy(); idx = r.permutation(len(X))
            for c in cs: Xp[c] = X[c].values[idx]
            ds.append(score_pred(predict(Xp), meta) - base)
        out[u] = float(np.mean(ds))
    return base, pd.Series(out)

def shap_vals(model, X):
    # TreeSHAP có sẵn của LightGBM: mỗi dòng ra 1 giá trị cho từng biến (cộng lại + base = dự báo trên thang log)
    c = model.m.predict(X[model.feats], pred_contrib=True)
    return pd.DataFrame(c[:, :-1], columns=model.feats, index=X.index), c[:, -1]

t0 = time.time()
tr1, va1 = fold_data(FOLD_XAI)
M1 = fit_gb_df(tr1, FEATURES)
va1s = va1.sample(min(N_SHAP, len(va1)), random_state=SEED)
SV1, BASE1 = shap_vals(M1, va1s)
vperm = va1.sample(min(N_PERM, len(va1)), random_state=SEED + 1)
pred_fn = lambda d: M1.predict(d)
print(f'Đã huấn luyện GB trên fold {FOLD_XAI} ({len(tr1):,} dòng) và tính SHAP cho {len(va1s):,} dòng validation trong {time.time()-t0:.0f}s')
print(f'RMSPE của mô hình trên validation {FOLD_XAI} (c = 1): {score_pred(M1.predict(va1), va1):.4f}')
chk = SV1.sum(axis=1).values + BASE1; print('Kiểm tra tính cộng của SHAP (tổng SHAP + base = dự báo log): sai lệch tối đa =', float(np.abs(chk - M1.predict(va1s)).max()))

# %% [markdown cell 111] ### 15.1 Chọn phương pháp xAI: so sánh các ứng viên & tác động khi Test không có Sales thật

# %% [code cell 112]
t0 = time.time()
Xp1 = vperm[FEATURES]
base_perm, perm_feat = perm_imp(pred_fn, Xp1, vperm, {f: [f] for f in FEATURES})
rank_gain = pd.Series(M1.m.booster_.feature_importance('gain'), index=M1.feats).reindex(FEATURES)
rank_shap = SV1.abs().mean().reindex(FEATURES)
RANKS = {'gain (cây)': rank_gain, 'SHAP (mean |SHAP|)': rank_shap, 'permutation (ΔRMSPE)': perm_feat}
top10 = pd.DataFrame({k: v.sort_values(ascending=False).head(10).index.tolist() for k, v in RANKS.items()}); top10.index = range(1, 11)
print(f'Permutation theo từng biến xong trong {time.time()-t0:.0f}s | RMSPE nền trên mẫu = {base_perm:.4f}')
print('Top 10 biến theo từng cách xếp hạng:'); display(top10)
sp = pd.DataFrame({a: {b: stats.spearmanr(RANKS[a], RANKS[b])[0] for b in RANKS} for a in RANKS}).round(2)
print('Tương quan thứ hạng (Spearman) giữa các cách xếp hạng trên 38 biến:'); display(sp)

# %% [markdown cell 113] **Nhận xét 15.1 (So sánh mức độ đồng thuận giữa các phương pháp):**
# %% [markdown cell 114] ### 15.3 Mức độ quan trọng theo NHÓM biến

# %% [code cell 115]
base_g, perm_grp = perm_imp(pred_fn, Xp1, vperm, FEATURE_GROUPS)
GROUP_TAB = pd.DataFrame({'ΔRMSPE': perm_grp, '% of Baseline RMSPE': perm_grp / base_g * 100,
                          'Feature Count': {g: len(fs) for g, fs in FEATURE_GROUPS.items()}}).sort_values('ΔRMSPE', ascending=False)
display(GROUP_TAB.round(4))

GROUP_EN = {
    'Lịch sử Sales (mức nền, cùng kỳ năm trước)': 'Sales History',
    'Lịch (thứ, tháng, tuần...)': 'Calendar',
    'Định danh cửa hàng': 'Store Identity',
    'Promo ngắn hạn': 'Short-term Promo',
    'Ngày lễ': 'Holidays',
    'Mùa vụ toàn chuỗi': 'Chain Trend',
    'Promo2': 'Promo2',
    'Đối thủ': 'Competition',
    'Đóng cửa dài ngày': 'Store Closures',
    'Đặc điểm cửa hàng': 'Store Characteristics'
}
s_perm = GROUP_TAB['ΔRMSPE'].rename(index=GROUP_EN).iloc[::-1]

fig, ax = plt.subplots(figsize=(8.5, 4.4))
s_perm.plot(kind='barh', ax=ax, color='#4472C4')
ax.set_title('Permutation Feature Importance (Level 1)')
ax.set_xlabel('ΔRMSPE when permuting feature group')
ax.set_xlim(0, s_perm.max() * 1.18)
for p in ax.patches:
    w = p.get_width()
    ax.annotate(f'+{w:.4f}', (w, p.get_y() + p.get_height() / 2), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8.5, fontweight='bold')
plt.tight_layout()
plt.show()


# %% [markdown cell 116] ### 15.4 Biểu đồ SHAP tổng quát (beeswarm) - biến nào đẩy dự báo lên hay xuống?

# %% [code cell 117]
# Lọc bỏ nhóm Lịch sử Sales để tập trung hoàn toàn vào các biến Đòn bẩy Điều hành (Actionable Levers)
sales_hist = FEATURE_GROUPS['Lịch sử Sales (mức nền, cùng kỳ năm trước)']
actionable_feats = [f for f in rank_shap.sort_values(ascending=False).index if f not in sales_hist][:12]

# 1. Bar Chart: Mean Absolute SHAP Value
fig, ax = plt.subplots(figsize=(8.5, 4.6))
s_top = rank_shap.loc[actionable_feats].iloc[::-1]
s_top.plot(kind='barh', ax=ax, color='#ED7D31')
ax.set_title('Mean Absolute SHAP Value')
ax.set_xlabel('mean |SHAP| (log scale)')
ax.set_xlim(0, s_top.max() * 1.18)
for p in ax.patches:
    w = p.get_width()
    ax.annotate(f'{w:.4f}', (w, p.get_y() + p.get_height() / 2), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8.5, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Beeswarm Plot: SHAP Beeswarm Summary
fig, ax = plt.subplots(figsize=(10.5, 6))
rj = np.random.default_rng(1)
top_f = actionable_feats[::-1]
for i, f in enumerate(top_f):
    v = va1s[f].astype(float); pct = v.rank(pct=True).fillna(0.5).values
    ax.scatter(SV1[f].values, i + rj.uniform(-0.28, 0.28, len(v)), c=pct, cmap='coolwarm', s=2.5, alpha=0.55)
ax.set_yticks(range(len(top_f)))
ax.set_yticklabels(top_f, fontsize=9.5)
ax.axvline(0, color='k', lw=0.9)
ax.set_xlabel('SHAP Value', fontsize=10)
ax.set_title('SHAP Beeswarm Summary', fontsize=11, fontweight='bold')
ax.set_xlim(-0.23, 0.27)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=ax, fraction=0.03, pad=0.02)
cbar.set_ticks([0.05, 0.95])
cbar.set_ticklabels(['Low', 'High'])
cbar.set_label('Feature Value')
plt.tight_layout()
plt.show()


# %% [markdown cell 118] ### 15.5 Chiều và độ lớn của các tác động chính

# %% [code cell 119]
X1s = va1s[FEATURES]
promo_cols = FEATURE_GROUPS['Promo ngắn hạn']
g_promo = SV1[promo_cols].sum(axis=1)
eff_p = g_promo.groupby(X1s.Promo.values).mean()
print(f'Promo: nhóm biến Promo đẩy dự báo của ngày có Promo cao hơn ngày không Promo: {np.expm1(eff_p[1] - eff_p[0]) * 100:+.1f}% (SHAP nhóm; mẫu {len(X1s):,} dòng)')
print(f"      riêng biến Promo: {np.expm1(SV1['Promo'].groupby(X1s.Promo.values).mean().diff().iloc[-1]) * 100:+.1f}%")
byT = pd.DataFrame({'SHAP nhóm Promo': g_promo.values, 'Promo': X1s.Promo.values, 'StoreType': va1s.StoreType.map(dict(enumerate(CATS['StoreType']))).values})
t_ = byT.groupby(['StoreType', 'Promo'])['SHAP nhóm Promo'].mean().unstack(); t_['% chênh (Promo=1 so với 0)'] = (np.expm1(t_[1] - t_[0]) * 100).round(1); t_['n dòng'] = byT.groupby('StoreType').size()
print('\nTheo StoreType (nhóm n nhỏ chỉ nêu như trường hợp cá biệt):'); display(t_.round(3))
dow = SV1['DayOfWeek'].groupby(X1s.DayOfWeek.values).mean(); dow_n = X1s.groupby('DayOfWeek').size()
dowdf = pd.DataFrame({'thứ': [DOW_NAME[k] for k in dow.index], 'SHAP trung bình (log)': dow.values, '% so với TB các thứ': (np.expm1(dow.values - np.average(dow.values, weights=dow_n.loc[dow.index].values)) * 100), 'n dòng': dow_n.loc[dow.index].values})
print('\nTác động của thứ trong tuần (biến DayOfWeek):'); display(dowdf.round(3))
for f_, lab in [('IsPromo2ActiveThisMonth', 'Promo2 đang active'), ('Promo2', 'cửa hàng có Promo2')]:
    e = SV1[f_].groupby(X1s[f_].values).mean()
    if len(e) > 1: print(f'{lab}: SHAP của biến {f_} khác nhau giữa 1 và 0 là {np.expm1(e.get(1, np.nan) - e.get(0, np.nan)) * 100:+.2f}% (rất nhỏ nghĩa là model gần như không dùng)')
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, f_ in zip(axs, ['dow_promo_mean', 'chain_delta', 'MonthsSinceCompetitionOpened']):
    ax.scatter(va1s[f_], SV1[f_], s=3, alpha=0.3); ax.set_xlabel(f_); ax.set_ylabel('SHAP Value (log)'); ax.set_title(f'{f_} SHAP Dependence')
plt.tight_layout(); plt.show()


# %% [markdown cell 120] ### 15.6 PDP / ICE: nếu chỉ đổi một biến thì dự báo đổi thế nào?

# %% [code cell 121]
def pdp_ice(model, X, feat, grid, n_ice=80, seed=0):
    r = np.random.default_rng(seed); preds = []
    for g_ in grid:
        Xg = X.copy(); Xg[feat] = g_; preds.append(model.predict(Xg))
    P = np.array(preds).T                                  # dòng x lưới, thang log
    rel = np.expm1(P - P[:, [0]]) * 100                    # % thay đổi so với điểm đầu lưới
    return rel.mean(axis=0), rel[r.choice(len(rel), min(n_ice, len(rel)), replace=False)]
Xpd = va1.sample(min(2000, len(va1)), random_state=3)[FEATURES]
specs = [('Promo', [0, 1]), ('DayOfWeek', [1, 2, 3, 4, 5, 6]), ('MonthsSinceCompetitionOpened', list(np.round(np.nanpercentile(TB.MonthsSinceCompetitionOpened, np.linspace(5, 95, 12)), 1))),
         ('CompetitionDistance', list(np.round(np.nanpercentile(TB.CompetitionDistance, np.linspace(5, 95, 12)), 0))), ('chain_delta', list(np.round(np.nanpercentile(TB.chain_delta, np.linspace(5, 95, 12)), 3))),
         ('DaysSinceReopen', [0, 7, 14, 30, 60, 90, 120])]
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5)); PDP = {}
for ax, (f_, grid) in zip(axs.ravel(), specs):
    pdp, ice = pdp_ice(M1, Xpd, f_, grid); PDP[f_] = pd.Series(pdp, index=grid)
    for line in ice: ax.plot(grid, line, color='#4472C4', alpha=0.12, lw=0.8)
    ax.plot(grid, pdp, color='#C00000', lw=2.5, marker='o', ms=5); ax.axhline(0, color='k', lw=0.6); ax.set_title(f_); ax.set_ylabel('% Prediction Change')
    # Gắn Data Labels cho các điểm quan trọng
    if f_ == 'Promo':
        for x_val, y_val in zip(grid, pdp):
            ax.annotate(f'{y_val:+.1f}%', (x_val, y_val), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='#C00000')
    elif f_ == 'DayOfWeek':
        for x_val, y_val in zip(grid, pdp):
            ax.annotate(f'{y_val:+.1f}%', (x_val, y_val), textcoords='offset points', xytext=(0, 8 if y_val >= 0 else -14), ha='center', fontsize=8.5, fontweight='bold', color='#C00000')
    elif f_ == 'DaysSinceReopen':
        for x_val, y_val in zip(grid[:4], pdp[:4]):
            ax.annotate(f'{y_val:.1f}%', (x_val, y_val), textcoords='offset points', xytext=(0, 8 if y_val >= -20 else -14), ha='center', fontsize=8.5, fontweight='bold', color='#C00000')
plt.tight_layout(); plt.show()
print('PDP (% thay đổi dự báo so với giá trị đầu lưới):'); 
for f_, s_ in PDP.items(): print(f'  {f_:<30}', {k: round(float(v), 1) for k, v in s_.items()})


# %% [markdown cell 122] ### 15.7 Giải thích từng dự báo cụ thể

# %% [code cell 123]
pe1 = (np.expm1(M1.predict(va1)) - va1.Sales.values) / va1.Sales.values
rows_sel = {'Điển hình (sai số gần trung vị)': va1.index[np.argmin(np.abs(np.abs(pe1) - np.median(np.abs(pe1))))], 'Sai nhiều nhất': va1.index[np.argmax(np.abs(pe1))]}
def explain_row(model, df, idx, title):
    X = df.loc[[idx], FEATURES]; sv, base = shap_vals(model, X); pred_log = float(model.predict(X)[0])
    s = sv.iloc[0].sort_values(key=np.abs, ascending=False).head(8)
    print(f'--- {title}: Store {int(df.loc[idx, "Store"])}, {df.loc[idx, "Date"].date()}, thứ {DOW_NAME[int(df.loc[idx, "DayOfWeek"])]}, Promo={int(df.loc[idx, "Promo"])}')
    print(f'    Thực tế {df.loc[idx, "Sales"]:,.0f} | dự báo {np.expm1(pred_log):,.0f} | mức nền (trung bình của model) {np.expm1(base[0]):,.0f}')
    display(pd.DataFrame({'giá trị biến': df.loc[idx, s.index].values, 'đóng góp (log)': s.values, '% so với mức hiện tại': (np.expm1(s.values) * 100)}, index=s.index).round(3))
for t_, i_ in rows_sel.items(): explain_row(M1, va1, i_, t_)

# %% [markdown cell 124] ### 15.8 Góc nhìn thứ hai: xem "động lực" khi đã trừ mức nền của cửa hàng

# %% [code cell 125]
t0 = time.time()
DRV_FEATS = [f for f in FEATURES if GROUP_OF[f] not in ('Lịch sử Sales (mức nền, cùng kỳ năm trước)', 'Định danh cửa hàng')]
def offset(df): return df.dow_promo_mean.fillna(df.roll91).fillna(df.roll28).fillna(GLOBAL_MED).values
trd = tr1.copy(); trd['y_res'] = trd.y_log.values - offset(trd)
drv = GB(XAI_PARAMS, use_store=False, seed=SEED, feats=DRV_FEATS).fit(trd.assign(y_log=trd['y_res']), trd['y_res'].values)
pred_drv = offset(va1) + drv.predict(va1)
print(f'Mô hình phụ (offset + phần chênh) trên validation {FOLD_XAI}: RMSPE = {score_pred(pred_drv, va1):.4f} | mô hình chính: {score_pred(M1.predict(va1), va1):.4f} | chỉ offset (B3): {score_pred(offset(va1), va1):.4f}  ({time.time()-t0:.0f}s)')
SVd, _ = shap_vals(drv, va1s)
drv_rank = SVd.abs().mean().sort_values(ascending=False)
drv_grp = pd.DataFrame({g: SVd[[f for f in fs if f in DRV_FEATS]].sum(axis=1) for g, fs in FEATURE_GROUPS.items() if any(f in DRV_FEATS for f in fs)}).abs().mean().sort_values(ascending=False)
print('Biến "động lực" quan trọng nhất (mean |SHAP|, log):'); display(drv_rank.head(10).round(4).to_frame('mean |SHAP|'))
print('Theo nhóm:'); display(drv_grp.round(4).to_frame('mean |SHAP nhóm|'))

# %% [markdown cell 126] **Nhận xét M03 (kết quả giải thích):**
# %% [markdown cell 127] ---

# %% [code cell 128]
t0 = time.time(); FOLD_MODELS = {FOLD_XAI: (M1, va1, SV1, va1s, vperm)}
for fold in FOLD_NAMES[1:]:
    tr_f, va_f = fold_data(fold); m_f = fit_gb_df(tr_f, FEATURES)
    vs_f = va_f.sample(min(N_SHAP, len(va_f)), random_state=SEED); vp_f = va_f.sample(min(N_PERM, len(va_f)), random_state=SEED + 1)
    FOLD_MODELS[fold] = (m_f, va_f, shap_vals(m_f, vs_f)[0], vs_f, vp_f)
print(f'Đã huấn luyện và giải thích {len(FOLD_MODELS)} mô hình (mỗi fold một mô hình) trong {time.time()-t0:.0f}s')

fold_shap = pd.DataFrame({f: v[2].abs().mean() for f, v in FOLD_MODELS.items()}).reindex(FEATURES)
fold_perm_grp = pd.DataFrame({f: perm_imp(lambda d, m=v[0]: m.predict(d), v[4][FEATURES], v[4], FEATURE_GROUPS)[1] for f, v in FOLD_MODELS.items()})
print('mean |SHAP| theo biến - top 10 của từng fold:'); display(pd.DataFrame({f: fold_shap[f].sort_values(ascending=False).head(10).index.tolist() for f in FOLD_NAMES}, index=range(1, 11)))
print('Permutation theo nhóm (ΔRMSPE) theo fold:'); display(fold_perm_grp.round(4))
sp_f = pd.DataFrame({a: {b: stats.spearmanr(fold_shap[a], fold_shap[b])[0] for b in FOLD_NAMES} for a in FOLD_NAMES}).round(2)
sp_f.index = sp_f.columns = ['F1', 'F2', 'F3', 'F4']; print('Spearman thứ hạng mean |SHAP| (38 biến) giữa các fold:'); display(sp_f)
spg = pd.DataFrame({a: {b: stats.spearmanr(fold_perm_grp[a], fold_perm_grp[b])[0] for b in FOLD_NAMES} for a in FOLD_NAMES}).round(2)
spg.index = spg.columns = ['F1', 'F2', 'F3', 'F4']; print('Spearman thứ hạng NHÓM (permutation) giữa các fold:'); display(spg)
top5 = {f: set(fold_shap[f].sort_values(ascending=False).head(5).index) for f in FOLD_NAMES}
overlap = {f: len(top5[f] & top5[FOLD_XAI]) for f in FOLD_NAMES}; print('Số biến trùng trong top-5 so với F1:', overlap)
promo_eff = {}
for f, (m_, vf, sv_, vs_, vp_) in FOLD_MODELS.items():
    gp = sv_[promo_cols].sum(axis=1); e = gp.groupby(vs_.Promo.values).mean(); promo_eff[f] = float(np.expm1(e[1] - e[0]) * 100)
print('Tác động Promo mà model dùng (SHAP nhóm Promo, Promo=1 so với 0) theo fold (%):', {k: round(v, 1) for k, v in promo_eff.items()})
dow_eff = pd.DataFrame({f: v[2]['DayOfWeek'].groupby(v[3].DayOfWeek.values).mean() for f, v in FOLD_MODELS.items()})
dow_corr = {f: round(float(dow_eff[[FOLD_XAI, f]].dropna().corr().iloc[0, 1]), 2) for f in FOLD_NAMES[1:]}
print('Tương quan mẫu hình "thứ trong tuần" (SHAP của DayOfWeek) giữa các fold so với F1:', dow_corr)

# %% [markdown cell 129] ### 16.2 Ổn định giữa các hạt giống và giữa hai loại mô hình

# %% [code cell 130]
t0 = time.time(); seed_shap = {SEED: rank_shap}
for sd in ([SEED + 1] if XAI_QUICK else [SEED + 1, SEED + 2]):
    ms = fit_gb_df(tr1, FEATURES, seed=sd); seed_shap[sd] = shap_vals(ms, va1s)[0].abs().mean().reindex(FEATURES)
print(f'Huấn luyện thêm {len(seed_shap)-1} hạt giống trong {time.time()-t0:.0f}s')
sp_seed = {sd: round(float(stats.spearmanr(seed_shap[SEED], seed_shap[sd])[0]), 3) for sd in seed_shap if sd != SEED}
ov_seed = {sd: len(set(seed_shap[SEED].sort_values(ascending=False).head(5).index) & set(seed_shap[sd].sort_values(ascending=False).head(5).index)) for sd in seed_shap if sd != SEED}
print('Spearman mean |SHAP| so với seed gốc:', sp_seed, '| số biến trùng top-5:', ov_seed)

# Random Forest (không có Store, điền NaN theo rf_fill) so với Gradient Boosting: so sánh theo NHÓM (không gồm 'Định danh cửa hàng')
t0 = time.time(); grp_nostore = {g: [f for f in fs if f != 'Store'] for g, fs in FEATURE_GROUPS.items() if any(f != 'Store' for f in fs)}
cmp_models = {}
for fold in [FOLD_NAMES[0], FOLD_NAMES[3]]:
    tr_f, va_f = fold_data(fold); rf = RandomForestRegressor(**BEST_RF, n_jobs=N_JOBS, random_state=SEED).fit(rf_fill(tr_f), tr_f.y_log.values)
    vp_f = FOLD_MODELS[fold][4]; gb_f = FOLD_MODELS[fold][0]
    pg = perm_imp(lambda d: gb_f.predict(d), vp_f[FEATURES], vp_f, grp_nostore)[1]
    pr = perm_imp(lambda d: rf.predict(rf_fill(d)), vp_f[NOSTORE], vp_f, grp_nostore)[1]
    cmp_models[fold] = pd.DataFrame({'GB': pg, 'RF': pr})
    print(f'Fold {fold}: Spearman thứ hạng nhóm GB vs RF = {stats.spearmanr(pg, pr)[0]:.2f}')
print(f'(RF xong trong {time.time()-t0:.0f}s)'); display(pd.concat(cmp_models, axis=1).round(4))

# %% [markdown cell 131] ### 16.3 Cảm biến nhiễu và đối chiếu với dữ liệu thô

# %% [code cell 132]
t0 = time.time(); rn = np.random.default_rng(99)
trn, van = tr1.copy(), va1.copy(); trn['noise'] = rn.normal(size=len(trn)); van['noise'] = rn.normal(size=len(van))
FN = FEATURES + ['noise']; mn = fit_gb_df(trn, FN)
vsn = van.loc[va1s.index]; svn, _ = shap_vals(mn, vsn); shap_rank_n = svn.abs().mean().sort_values(ascending=False)
vpn = van.loc[vperm.index]; _, perm_n = perm_imp(lambda d: mn.predict(d), vpn[FN], vpn, {f: [f] for f in FN}); perm_rank_n = perm_n.sort_values(ascending=False)
print(f'Biến nhiễu: hạng SHAP = {list(shap_rank_n.index).index("noise") + 1}/{len(FN)} (mean |SHAP| = {shap_rank_n["noise"]:.5f}); hạng permutation = {list(perm_rank_n.index).index("noise") + 1}/{len(FN)} (ΔRMSPE = {perm_n["noise"]:+.5f})  ({time.time()-t0:.0f}s)')

# đối chiếu Promo trong dữ liệu thô (validation F1): trong từng cửa hàng và thứ T2-T6, chênh lệch log-doanh số giữa ngày có và không Promo
d_ = va1[va1.DayOfWeek <= 5].assign(ly=lambda x: np.log(x.Sales))
g_ = d_.groupby(['Store', 'DayOfWeek', 'Promo']).ly.mean().unstack('Promo').dropna()
emp = float(np.expm1((g_[1] - g_[0]).mean()) * 100)
print(f'Dữ liệu thô (validation {FOLD_XAI}): trong cùng cửa hàng và cùng thứ, ngày có Promo cao hơn ngày không Promo trung bình {emp:+.1f}% (từ {len(g_):,} cặp cửa hàng-thứ)')
print(f'Model dùng (SHAP nhóm Promo, fold {FOLD_XAI}): {promo_eff[FOLD_XAI]:+.1f}%')

# %% [markdown cell 133] ### 16.4 Bảng kết quả kiểm tra độ tin cậy (tự động chấm theo tiêu chí ở đầu mục)

# %% [code cell 134]
chk_rows = []
others = [f for f in FOLD_NAMES if f != FOLD_XAI]
chk_rows.append(('1a. Top-5 biến trùng ≥ 4/5 với F1 (cả 3 fold còn lại)', all(overlap[f] >= 4 for f in others), overlap))
chk_rows.append(('1b. Spearman thứ hạng biến (SHAP) giữa fold ≥ 0,8', bool((sp_f.values[np.triu_indices(4, 1)] >= 0.8).all()), float(sp_f.values[np.triu_indices(4, 1)].min())))
chk_rows.append(('1c. Spearman thứ hạng NHÓM (permutation) giữa fold ≥ 0,8', bool((spg.values[np.triu_indices(4, 1)] >= 0.8).all()), float(spg.values[np.triu_indices(4, 1)].min())))
chk_rows.append(('1d. Chiều tác động Promo không đổi dấu giữa các fold', len({np.sign(v) for v in promo_eff.values()}) == 1, {k: round(v, 1) for k, v in promo_eff.items()}))
chk_rows.append(('2.  Hạt giống: Spearman ≥ 0,9', all(v >= 0.9 for v in sp_seed.values()), sp_seed))
rf_sp = {f: float(stats.spearmanr(df_['GB'], df_['RF'])[0]) for f, df_ in cmp_models.items()}
chk_rows.append(('3.  GB vs RF: Spearman thứ hạng nhóm ≥ 0,7', all(v >= 0.7 for v in rf_sp.values()), {k: round(v, 2) for k, v in rf_sp.items()}))
rk_s, rk_p = list(shap_rank_n.index).index('noise') + 1, list(perm_rank_n.index).index('noise') + 1
chk_rows.append(('4.  Biến nhiễu nằm ở nửa dưới bảng xếp hạng (cả SHAP và permutation)', rk_s > len(FN) / 2 and rk_p > len(FN) / 2, {'hạng SHAP': rk_s, 'hạng permutation': rk_p, 'tổng biến': len(FN)}))
chk_rows.append(('5.  Promo: model và dữ liệu thô cùng chiều', np.sign(emp) == np.sign(promo_eff[FOLD_XAI]), {'dữ liệu thô %': round(emp, 1), 'model %': round(promo_eff[FOLD_XAI], 1)}))
leak_ok = not (set(FEATURES) & FORBIDDEN)
chk_rows.append(('6.  Không có biến bị cấm (Customers/Sales thô/Id) trong danh sách biến được giải thích', leak_ok, 'ok' if leak_ok else 'CÓ LỖI'))
E03 = pd.DataFrame(chk_rows, columns=['Kiểm tra', 'Đạt?', 'Chi tiết']); E03['Đạt?'] = E03['Đạt?'].map({True: 'ĐẠT', False: 'CHƯA ĐẠT'})
display(E03)

# %% [markdown cell 135] **Nhận xét E03 (độ tin cậy của giải thích):**
# %% [markdown cell 136] ### 16.5 Kiểm tra bổ sung: nghịch lý Simpson của ngày lễ (`StateHoliday` theo `StoreType`)

# %% [code cell 137]
hol = []
for fold in [FOLD_NAMES[1], FOLD_NAMES[2]]:
    m_, vf, _, _, _ = FOLD_MODELS[fold]; vh = vf[vf.StateHoliday > 0]
    if len(vh) == 0: continue
    sv_, _ = shap_vals(m_, vh); hol.append(pd.DataFrame({'fold': fold, 'StoreType': vh.StoreType.map(dict(enumerate(CATS['StoreType']))).values, 'SHAP StateHoliday (log)': sv_['StateHoliday'].values}))
hol = pd.concat(hol); tab_h = hol.groupby('StoreType')['SHAP StateHoliday (log)'].agg(['mean', 'size']); tab_h['% đẩy dự báo'] = (np.expm1(tab_h['mean']) * 100).round(1)
print('SHAP của biến StateHoliday trên các dòng ngày lễ (F2 + F3), theo StoreType:'); display(tab_h.round(3))

# %% [markdown cell 138] **Kết luận xAI (tổng hợp M03 + E03):**

```
