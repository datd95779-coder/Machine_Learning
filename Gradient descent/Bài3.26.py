# Hàm f(x)
def cost(x):
    return x**2 - 4*x + 5

# Đạo hàm f'(x)
def grad(x):
    return 2*x - 4

# Gradient Descent
def myGD1(x0, eta):
    x = [x0]

    # Thực hiện 4 bước
    for it in range(4):
        # Công thức: x_new = x - eta * gradient
        x_new = x[-1] - eta * grad(x[-1])

        # Lưu giá trị x mới
        x.append(x_new)

    return x


# x ban đầu = 5, learning rate = 0.2
x = myGD1(5, 0.2)

# In kết quả từng bước
for i in range(len(x)):
    print("Buoc", i)
    print("x =", x[i])
    print("f(x) =", cost(x[i]))
    print()