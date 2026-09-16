# Overfitting and K-Fold Cross Validation

Đây là bài tập Machine Learning đầu tiên của tôi.

## Mục tiêu

Bài tập nhằm minh họa hiện tượng Overfitting và cách sử dụng K-Fold Cross Validation để lựa chọn mô hình có độ phức tạp phù hợp.

## Dataset

Dataset gồm 15 mẫu dữ liệu về:

- Chiều cao (cm)
- Cân nặng (kg)

Mục tiêu là dự đoán cân nặng dựa trên chiều cao.

## Phương pháp

Quy trình thực hiện:

1. Đọc dữ liệu bằng Pandas.
2. Chia dữ liệu thành tập Train và Test.
3. Sử dụng Polynomial Regression bậc cao để tạo mô hình có dấu hiệu Overfitting.
4. Đánh giá mô hình bằng R² trên Train và Test.
5. Sử dụng 5-Fold Cross Validation.
6. Thử các Polynomial Degree từ 1 đến 10.
7. Chọn Degree có điểm Cross Validation tốt nhất.
8. Huấn luyện lại mô hình được chọn.
9. Đánh giá trên tập Test.

## Kết quả

Mô hình Polynomial Degree 10:

- R² Train: 0.9955
- R² Test: 0.6080

Kết quả K-Fold:

- Degree 4 có R² trung bình cao nhất: 0.8822

Do đó Degree 4 được lựa chọn làm mô hình sau Cross Validation.

## Cấu trúc project

```text
Overfitting-KFold-ML/
│
├── data.csv
├── TestData.py
├── overfitting_kfold.py
├── README.md
└── .gitignore
