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
