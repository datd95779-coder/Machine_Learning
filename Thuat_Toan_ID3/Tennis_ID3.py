"""
ID3 - Bài toán "Có đi chơi không?" (dữ liệu thời tiết)
Chạy: python id3_play_tennis.py
Chỉ dùng thư viện chuẩn. Dữ liệu đã là dạng nhóm nên KHÔNG cần rời rạc hóa.
"""
import math
from collections import Counter

FEATURES = ["outlook", "temperature", "humidity", "wind"]
TARGET = "play"

# (id, outlook, temperature, humidity, wind, play)
RAW = [
    (1, "sunny", "hot", "high", "weak", "no"),
    (2, "sunny", "hot", "high", "strong", "no"),
    (3, "overcast", "hot", "high", "weak", "yes"),
    (4, "rainy", "mild", "high", "weak", "yes"),
    (5, "rainy", "cool", "normal", "weak", "yes"),
    (6, "rainy", "cool", "normal", "strong", "no"),
    (7, "overcast", "cool", "normal", "strong", "yes"),
    (8, "sunny", "mild", "high", "weak", "no"),
    (9, "sunny", "cool", "normal", "weak", "yes"),
    (10, "rainy", "mild", "normal", "weak", "yes"),
    (11, "sunny", "mild", "normal", "strong", "yes"),
    (12, "overcast", "mild", "high", "strong", "yes"),
    (13, "overcast", "hot", "normal", "weak", "yes"),
    (14, "rainy", "mild", "high", "strong", "no"),
]
DATA = [dict(zip(["id"] + FEATURES + [TARGET], row)) for row in RAW]


def entropy(rows):
    """Độ lộn xộn của một nhóm."""
    total = len(rows)
    h = 0.0
    for c in Counter(r[TARGET] for r in rows).values():
        p = c / total
        h -= p * math.log2(p)
    return h


def split_by(rows, feature):
    groups = {}
    for r in rows:
        groups.setdefault(r[feature], []).append(r)
    return groups


def info_gain(rows, feature):
    """IG = Entropy(trước) - Entropy(sau, có trọng số)."""
    total = len(rows)
    after = sum(len(g) / total * entropy(g)
                for g in split_by(rows, feature).values())
    return entropy(rows) - after


def majority(rows):
    return Counter(r[TARGET] for r in rows).most_common(1)[0][0]


def id3(rows, features, depth=0):
    labels = {r[TARGET] for r in rows}
    if len(labels) == 1:          # dừng: nhóm thuần nhất
        return labels.pop()
    if not features:              # dừng: hết thuộc tính
        return majority(rows)

    gains = {f: info_gain(rows, f) for f in features}
    best = max(gains, key=gains.get)

    pad = "  " * depth
    yes = sum(r[TARGET] == "yes" for r in rows)
    print(f"{pad}Nút ID {[r['id'] for r in rows]}  "
          f"(yes={yes}, no={len(rows) - yes}, H={entropy(rows):.3f})")
    for f, g in gains.items():
        print(f"{pad}  IG({f}) = {g:.3f}" + ("  <-- chọn" if f == best else ""))

    rest = [f for f in features if f != best]
    return {best: {v: id3(g, rest, depth + 1)
                   for v, g in split_by(rows, best).items()}}


def print_tree(tree, indent=""):
    feature = next(iter(tree))
    print(f"{indent}[{feature}]")
    for value, sub in tree[feature].items():
        if isinstance(sub, dict):
            print(f"{indent}  ├─ {value}:")
            print_tree(sub, indent + "  │   ")
        else:
            print(f"{indent}  ├─ {value} => {sub}")


def predict(tree, sample, default="yes"):
    while isinstance(tree, dict):
        feature = next(iter(tree))
        value = sample.get(feature)
        if value not in tree[feature]:
            return default
        tree = tree[feature][value]
    return tree


if __name__ == "__main__":
    print(f"Entropy toàn bộ: {entropy(DATA):.3f}\n")
    print("--- XÂY CÂY ---")
    tree = id3(DATA, FEATURES)

    print("\n--- CÂY QUYẾT ĐỊNH ---")
    print_tree(tree)

    ok = sum(predict(tree, r) == r[TARGET] for r in DATA)
    print(f"\nĐộ chính xác trên tập huấn luyện: {ok}/{len(DATA)}")

    khach = {"outlook": "sunny", "temperature": "cool",
             "humidity": "high", "wind": "strong"}
    print("Dự đoán (sunny, cool, high, strong):", predict(tree, khach))
