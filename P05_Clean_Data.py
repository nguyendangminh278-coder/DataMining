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
