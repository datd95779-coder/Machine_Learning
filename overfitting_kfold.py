# ============================================================
# 1. IMPORT THƯ VIỆN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# train_test_split:
# Dùng để chia dữ liệu thành 2 phần:
# - Tập TRAIN: dùng để cho mô hình học
# - Tập TEST: dùng để kiểm tra mô hình sau khi học
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# PolynomialFeatures:
# Dùng để biến dữ liệu bình thường thành dữ liệu đa thức.
#
# Ví dụ:
# Chiều cao = x
#
# Bậc 2 sẽ tạo ra:
# x, x²
#
# Bậc 3 sẽ tạo ra:
# x, x², x³
#
# Bậc càng cao -> mô hình càng phức tạp
from sklearn.preprocessing import PolynomialFeatures

# LinearRegression:
# Đây là mô hình hồi quy tuyến tính.
# Nó tìm ra mối quan hệ giữa chiều cao và cân nặng.
from sklearn.linear_model import LinearRegression

# make_pipeline:
# Dùng để ghép nhiều bước xử lý thành một mô hình.
#
# Trong bài này:
# PolynomialFeatures -> LinearRegression
from sklearn.pipeline import make_pipeline

# r2_score:
# Dùng để đánh giá mô hình bằng chỉ số R².
#
# R² càng gần 1 -> mô hình giải thích dữ liệu càng tốt.
# R² thấp hoặc âm -> mô hình dự đoán kém.
from sklearn.metrics import r2_score


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

# Đọc file data.csv
df = pd.read_csv("data.csv")

# In ra toàn bộ dữ liệu để kiểm tra
print("DỮ LIỆU:")
print(df)


# ============================================================
# 3. CHỌN X VÀ y
# ============================================================

# X = dữ liệu đầu vào (Feature)
# Trong bài này, chúng ta dùng CHIỀU CAO để dự đoán cân nặng.
#
# X phải viết [["ChieuCao"]] với 2 dấu ngoặc vuông
# vì Scikit-learn cần X ở dạng bảng 2 chiều.
X = df[["ChieuCao"]]

# y = giá trị mà chúng ta muốn dự đoán (Target)
# Ở đây là CÂN NẶNG.
y = df["CanNang"]


# ============================================================
# 4. CHIA DỮ LIỆU TRAIN / TEST
# ============================================================

# Chia dữ liệu:
# - 80% dùng để TRAIN
# - 20% dùng để TEST
#
# random_state=42:
# Cố định cách chia dữ liệu.
# Nếu chạy lại chương trình thì kết quả chia vẫn giống nhau.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# In số lượng dữ liệu của mỗi tập
print()
print("SỐ LƯỢNG DỮ LIỆU:")
print("Số mẫu TRAIN:", len(X_train))
print("Số mẫu TEST :", len(X_test))
# ============================================================
# 5. TẠO MÔ HÌNH POLYNOMIAL BẬC CAO
# ============================================================

# Polynomial Regression là hồi quy đa thức.
#
# degree=10 nghĩa là mô hình có thể tạo ra các thành phần:
#
# x, x², x³, x⁴, ..., x¹⁰
#
# Khi degree càng lớn, mô hình càng phức tạp.
#
# Dataset của chúng ta chỉ có 15 mẫu,
# nhưng mô hình lại có độ phức tạp rất cao.
# Đây là cách chúng ta CỐ TÌNH tạo ra nguy cơ Overfitting.
model_overfit = make_pipeline(
    PolynomialFeatures(degree=10),
    LinearRegression()
)


# ============================================================
# 6. CHO MÔ HÌNH HỌC DỮ LIỆU TRAIN
# ============================================================

# fit() = cho mô hình học từ dữ liệu.
#
# Mô hình chỉ được nhìn thấy X_train và y_train.
model_overfit.fit(X_train, y_train)


# ============================================================
# 7. DỰ ĐOÁN TRÊN TRAIN VÀ TEST
# ============================================================

# Dự đoán lại dữ liệu TRAIN
y_train_pred = model_overfit.predict(X_train)

# Dự đoán dữ liệu TEST
# Đây là dữ liệu mà mô hình chưa được học trực tiếp.
y_test_pred = model_overfit.predict(X_test)


# ============================================================
# 8. ĐÁNH GIÁ MÔ HÌNH
# ============================================================

# Tính R² trên tập TRAIN
r2_train = r2_score(y_train, y_train_pred)

# Tính R² trên tập TEST
r2_test = r2_score(y_test, y_test_pred)


# In kết quả
print()
print("KẾT QUẢ MÔ HÌNH POLYNOMIAL BẬC 10")
print("-----------------------------------")
print("R² trên TRAIN:", r2_train)
print("R² trên TEST :", r2_test)
# ============================================================
# 9. K-FOLD CROSS VALIDATION
# ============================================================

# KFold dùng để chia dữ liệu thành K phần.
#
# Ở đây chúng ta sử dụng 5-Fold:
#
#        DỮ LIỆU
#    ┌────┬────┬────┬────┬────┐
#    │ F1 │ F2 │ F3 │ F4 │ F5 │
#    └────┴────┴────┴────┴────┘
#
# Mỗi lần:
# - 4 phần dùng để TRAIN
# - 1 phần dùng để VALIDATION
#
# Sau đó thay đổi phần Validation.
# Làm tổng cộng 5 lần.
#
# shuffle=True:
# Trộn dữ liệu trước khi chia.
#
# random_state=42:
# Giúp mỗi lần chạy chương trình có cách chia giống nhau.
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 10. THỬ NHIỀU ĐỘ PHỨC TẠP CỦA MÔ HÌNH
# ============================================================

# Danh sách các bậc Polynomial mà chúng ta muốn thử.
#
# Ví dụ:
# degree=1  -> mô hình đơn giản
# degree=2  -> phức tạp hơn
# degree=3  -> phức tạp hơn nữa
# ...
# degree=10 -> rất phức tạp
degrees = range(1, 11)


# Dùng dictionary để lưu kết quả Cross Validation.
cv_results = {}


# ============================================================
# 11. CHẠY K-FOLD CHO TỪNG DEGREE
# ============================================================

for degree in degrees:

    # Tạo mô hình Polynomial với degree hiện tại.
    #
    # Ví dụ vòng lặp đầu tiên:
    # degree = 1
    #
    # Sau đó:
    # degree = 2
    # degree = 3
    # ...
    # degree = 10
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    # cross_val_score sẽ:
    # 1. Chia dữ liệu thành 5 phần
    # 2. Train trên 4 phần
    # 3. Kiểm tra trên 1 phần
    # 4. Lặp lại 5 lần
    #
    # scoring="r2":
    # Sử dụng R² để đánh giá mô hình.
    scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring="r2"
    )

    # Tính điểm R² trung bình của 5 Fold.
    mean_score = scores.mean()

    # Lưu kết quả.
    cv_results[degree] = mean_score


# ============================================================
# 12. IN KẾT QUẢ K-FOLD
# ============================================================

print()
print("KẾT QUẢ K-FOLD CROSS VALIDATION")
print("-----------------------------------")

for degree, score in cv_results.items():
    print(f"Degree {degree}: R² trung bình = {score:.4f}")


# ============================================================
# 13. TÌM DEGREE CÓ ĐIỂM K-FOLD CAO NHẤT
# ============================================================

# Tìm degree có R² trung bình cao nhất.
best_degree = max(
    cv_results,
    key=cv_results.get
)

# Lấy điểm R² tương ứng.
best_score = cv_results[best_degree]


# In kết quả.
print()
print("MÔ HÌNH ĐƯỢC CHỌN SAU K-FOLD")
print("-----------------------------------")
print("Degree tốt nhất:", best_degree)
print("R² trung bình :", round(best_score, 4))
# ============================================================
# 14. TẠO LẠI MÔ HÌNH TỐT NHẤT
# ============================================================

# K-Fold đã cho chúng ta biết degree phù hợp nhất.
# Trong dữ liệu hiện tại, kết quả cho thấy degree = 1.
#
# Bây giờ chúng ta tạo mô hình với degree được chọn.
final_model = make_pipeline(
    PolynomialFeatures(degree=best_degree),
    LinearRegression()
)


# ============================================================
# 15. HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG
# ============================================================

# Cho mô hình học toàn bộ 12 dữ liệu TRAIN.
final_model.fit(X_train, y_train)


# ============================================================
# 16. KIỂM TRA TRÊN 3 DỮ LIỆU TEST
# ============================================================

# Mô hình dự đoán 3 dữ liệu mà nó chưa được học.
y_test_final_pred = final_model.predict(X_test)


# Tính R² trên tập TEST.
final_r2_test = r2_score(
    y_test,
    y_test_final_pred
)


# ============================================================
# 17. IN KẾT QUẢ CUỐI CÙNG
# ============================================================

print()
print("KẾT QUẢ MÔ HÌNH SAU KHI DÙNG K-FOLD")
print("-----------------------------------")
print("Degree được chọn:", best_degree)
print("R² TEST cuối cùng:", final_r2_test)

print()
print("SO SÁNH GIÁ TRỊ THỰC TẾ VÀ DỰ ĐOÁN")
print("-----------------------------------")

for actual, predicted in zip(y_test, y_test_final_pred):
    print(
        f"Thực tế: {actual} kg"
        f" | Dự đoán: {predicted:.2f} kg"
    )
# ============================================================
# 18. VẼ BIỂU ĐỒ
# ============================================================

# Tạo một vùng để vẽ biểu đồ
plt.figure(figsize=(10, 6))


# ------------------------------------------------------------
# Vẽ các điểm dữ liệu thực tế
# ------------------------------------------------------------

# Scatter plot = biểu đồ các điểm.
#
# Trục X: Chiều cao
# Trục Y: Cân nặng
#
# Đây là dữ liệu thật trong file data.csv.
plt.scatter(
    X["ChieuCao"],
    y,
    label="Dữ liệu thực tế"
)


# ------------------------------------------------------------
# Tạo dữ liệu chiều cao liên tục để vẽ đường dự đoán
# ------------------------------------------------------------

# Tạo 200 giá trị chiều cao từ 147 đến 183 cm.
#
# Tại sao không dùng trực tiếp 15 chiều cao?
# Vì muốn đường dự đoán nhìn mượt hơn.
X_plot = np.linspace(
    X["ChieuCao"].min(),
    X["ChieuCao"].max(),
    200
).reshape(-1, 1)


# ------------------------------------------------------------
# Dự đoán bằng mô hình Overfitting
# ------------------------------------------------------------

# Mô hình Polynomial bậc 10 dự đoán cân nặng
# tương ứng với 200 chiều cao ở trên.
y_overfit_plot = model_overfit.predict(X_plot)


# Vẽ đường của mô hình Overfitting
plt.plot(
    X_plot,
    y_overfit_plot,
    label="Polynomial Degree 10"
)


# ------------------------------------------------------------
# Dự đoán bằng mô hình sau K-Fold
# ------------------------------------------------------------

# Mô hình cuối cùng sử dụng degree mà K-Fold lựa chọn.
y_final_plot = final_model.predict(X_plot)


# Vẽ đường mô hình sau K-Fold
plt.plot(
    X_plot,
    y_final_plot,
    label=f"Model sau K-Fold (Degree {best_degree})"
)


# ------------------------------------------------------------
# Trang trí biểu đồ
# ------------------------------------------------------------

plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")

plt.title(
    "So sánh mô hình Overfitting và mô hình sau K-Fold"
)

plt.legend()

plt.grid(True)

plt.show()