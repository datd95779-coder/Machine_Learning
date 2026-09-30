"""
ID3 - Cây quyết định cho bài toán "Rủi ro tín dụng"
Chạy: python id3_credit_risk.py
Chỉ dùng thư viện chuẩn của Python (math, collections).
"""
import math
from collections import Counter

# ------------------------------------------------------------
# 1. DỮ LIỆU (chép từ đề bài)
# ------------------------------------------------------------
# (ID, Tuổi, Hôn nhân, BĐS, Thu nhập, Rủi ro)
RAW = [
    (1, 25, "Độc thân", "Ở cùng bố mẹ", 7000000, 0),
    (2, 40, "Đã kết hôn", "Nhà sở hữu", 18000000, 0),
    (3, 35, "Từng ly hôn", "Nhà thuê", 12000000, 1),
    (4, 27, "Đã kết hôn", "Ở cùng bố mẹ", 9000000, 1),
    (5, 31, "Độc thân", "Nhà thuê", 6000000, 1),
    (6, 36, "Đã kết hôn", "Nhà sở hữu", 8000000, 1),
    (7, 48, "Độc thân", "Nhà thuê", 7000000, 0),
    (8, 26, "Độc thân", "Nhà thuê", 8000000, 1),
    (9, 33, "Từng ly hôn", "Ở cùng bố mẹ", 5000000, 1),
    (10, 29, "Độc thân", "Nhà thuê", 10000000, 0),
    (11, 38, "Đã kết hôn", "Nhà sở hữu", 15000000, 0),
    (12, 44, "Độc thân", "Nhà sở hữu", 14000000, 1),
    (13, 42, "Đã kết hôn", "Nhà sở hữu", 10000000, 0),
    (14, 28, "Độc thân", "Nhà thuê", 7000000, 1),
    (15, 30, "Đã kết hôn", "Ở cùng bố mẹ", 6000000, 1),
]

TARGET = "Rủi ro"
FEATURES = ["Tuổi", "Hôn nhân", "BĐS", "Thu nhập"]


# ------------------------------------------------------------
# 2. RỜI RẠC HÓA (ID3 chỉ làm việc với dữ liệu dạng nhóm)
# ------------------------------------------------------------
def bin_age(age):
    if age < 30:
        return "Trẻ (<30)"
    if age < 40:
        return "Trung (30-39)"
    return "Lớn (>=40)"


def bin_income(income):
    if income < 8_000_000:
        return "Thấp (<8tr)"
    if income < 12_000_000:
        return "TB (8-<12tr)"
    return "Cao (>=12tr)"


def build_dataset():
    rows = []
    for id_, age, marry, prop, income, risk in RAW:
        rows.append({
            "ID": id_,
            "Tuổi": bin_age(age),
            "Hôn nhân": marry,
            "BĐS": prop,
            "Thu nhập": bin_income(income),
            TARGET: risk,
        })
    return rows


# ------------------------------------------------------------
# 3. ENTROPY & INFORMATION GAIN
# ------------------------------------------------------------
def entropy(rows):
    """Độ 'lộn xộn' của một nhóm."""
    total = len(rows)
    counts = Counter(r[TARGET] for r in rows)
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def split_by(rows, feature):
    groups = {}
    for r in rows:
        groups.setdefault(r[feature], []).append(r)
    return groups


def info_gain(rows, feature):
    """IG = Entropy(trước chia) - Entropy(sau chia, có trọng số)."""
    total = len(rows)
    after = 0.0
    for group in split_by(rows, feature).values():
        after += len(group) / total * entropy(group)
    return entropy(rows) - after


# ------------------------------------------------------------
# 4. XÂY CÂY ID3 (đệ quy)
# ------------------------------------------------------------
def majority(rows):
    return Counter(r[TARGET] for r in rows).most_common(1)[0][0]


def id3(rows, features, depth=0, verbose=True):
    labels = {r[TARGET] for r in rows}

    # Điều kiện dừng 1: nhóm thuần nhất
    if len(labels) == 1:
        return labels.pop()
    # Điều kiện dừng 2: hết thuộc tính -> lấy đa số
    if not features:
        return majority(rows)

    # Chọn thuộc tính có IG lớn nhất
    gains = {f: info_gain(rows, f) for f in features}
    best = max(gains, key=gains.get)

    if verbose:
        pad = "  " * depth
        ids = [r["ID"] for r in rows]
        print(f"{pad}Nút có ID {ids}, Entropy = {entropy(rows):.3f}")
        for f, g in gains.items():
            mark = "  <-- chọn" if f == best else ""
            print(f"{pad}  IG({f}) = {g:.3f}{mark}")

    remaining = [f for f in features if f != best]
    tree = {best: {}}
    for value, group in split_by(rows, best).items():
        tree[best][value] = id3(group, remaining, depth + 1, verbose)
    return tree


# ------------------------------------------------------------
# 5. IN CÂY & DỰ ĐOÁN
# ------------------------------------------------------------
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"{indent}=> {tree}")
        return
    feature = next(iter(tree))
    print(f"{indent}[{feature}]")
    for value, sub in tree[feature].items():
        if isinstance(sub, dict):
            print(f"{indent}  ├─ {value}:")
            print_tree(sub, indent + "  │   ")
        else:
            print(f"{indent}  ├─ {value} => {sub}")


def predict(tree, sample, default=1):
    """sample: dict đã rời rạc hóa, ví dụ {'Tuổi': 'Trẻ (<30)', ...}"""
    while isinstance(tree, dict):
        feature = next(iter(tree))
        value = sample.get(feature)
        if value not in tree[feature]:
            return default  # giá trị chưa từng thấy
        tree = tree[feature][value]
    return tree


def predict_raw(tree, age, marry, prop, income):
    sample = {
        "Tuổi": bin_age(age),
        "Hôn nhân": marry,
        "BĐS": prop,
        "Thu nhập": bin_income(income),
    }
    return predict(tree, sample)


# ------------------------------------------------------------
# 6. CHẠY
# ------------------------------------------------------------
if __name__ == "__main__":
    data = build_dataset()

    print("=" * 60)
    print(f"Entropy toàn bộ dữ liệu: {entropy(data):.3f}")
    print("=" * 60)
    print("\nIG của từng thuộc tính ở gốc:")
    for f in FEATURES:
        print(f"  IG({f:9s}) = {info_gain(data, f):.3f}")

    print("\n--- QUÁ TRÌNH XÂY CÂY ---")
    tree = id3(data, FEATURES)

    print("\n--- CÂY QUYẾT ĐỊNH ---")
    print_tree(tree)

    # Kiểm tra độ chính xác trên chính tập huấn luyện
    correct = sum(predict(tree, r) == r[TARGET] for r in data)
    print(f"\nĐộ chính xác trên tập huấn luyện: {correct}/{len(data)}")

    # Thử dự đoán khách hàng mới
    print("\n--- DỰ ĐOÁN KHÁCH MỚI ---")
    kq = predict_raw(tree, age=35, marry="Đã kết hôn",
                     prop="Nhà sở hữu", income=15_000_000)
    print("35 tuổi, Đã kết hôn, Nhà sở hữu, 15tr =>",
          "Có rủi ro (1)" if kq == 1 else "Không rủi ro (0)")
