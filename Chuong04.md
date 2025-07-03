Sau khi đã chuẩn bị và khám phá dữ liệu, giờ là lúc chúng ta bước vào một trong những nhiệm vụ cốt lõi và thú vị nhất của khai phá dữ liệu: **Phân lớp**.

-----

### **Chương 4: Phân lớp Dữ liệu (Classification)**

Ở chương này, chúng ta sẽ "dạy" cho máy tính cách phân biệt các loài hoa Iris dựa trên các đặc điểm của chúng. Đây là một nhiệm vụ **học có giám sát (supervised learning)**, nghĩa là chúng ta có sẵn dữ liệu đã được gán nhãn (chúng ta biết bông hoa nào thuộc loài nào) để làm "mẫu" cho máy học.

Mục tiêu là xây dựng một **mô hình (model)** có khả năng dự đoán đúng nhãn (loài hoa) cho một bông hoa mới mà nó chưa từng thấy trước đây.

-----

### **Phần 1: Lý Thuyết (Theory)**

#### **1. Phân lớp là gì?**

Phân lớp là quá trình xây dựng một mô hình ánh xạ các thuộc tính đầu vào (input features) vào một nhãn lớp rời rạc (discrete class label) đã được định nghĩa trước.

  * **Đầu vào:** Một tập hợp các thuộc tính (ví dụ: 4 số đo của hoa Iris).
  * **Đầu ra:** Một nhãn lớp (ví dụ: 'setosa', 'versicolor', hoặc 'virginica').

Quy trình phân lớp thường bao gồm hai giai đoạn chính:

1.  **Giai đoạn Huấn luyện (Training):** Mô hình sẽ "học" từ một tập dữ liệu được gọi là **tập huấn luyện (training set)**. Trong tập này, các nhãn lớp là đã biết trước. Mô hình cố gắng tìm ra các quy luật, các mối quan hệ giữa các thuộc tính và nhãn lớp.
2.  **Giai đoạn Kiểm thử (Testing):** Sau khi được huấn luyện, mô hình sẽ được đánh giá trên một **tập kiểm thử (testing set)** - một phần dữ liệu mà mô hình chưa từng thấy. Độ chính xác của mô hình trên tập này cho thấy khả năng **tổng quát hóa (generalization)** của nó, tức là khả năng dự đoán đúng trên dữ liệu thực tế.

#### **2. Các Thuật toán Phân lớp Phổ biến**

**A. Cây Quyết định (Decision Tree)** 🌳

Đây là một trong những mô hình trực quan và dễ hiểu nhất. Nó xây dựng một cấu trúc giống như một biểu đồ luồng (flowchart).

  * Mỗi **nút trong (internal node)** đại diện cho một câu hỏi hay một "phép thử" trên một thuộc tính (ví dụ: `petal length < 2.45 cm?`).
  * Mỗi **nhánh (branch)** đại diện cho kết quả của phép thử (ví dụ: Đúng hoặc Sai).
  * Mỗi **nút lá (leaf node)** đại diện cho một quyết định cuối cùng, tức là một nhãn lớp (ví dụ: `setosa`).

**Ưu điểm:** Rất dễ diễn giải ("white-box" model), chúng ta có thể thấy chính xác cách mô hình đưa ra quyết định.
**Nhược điểm:** Dễ bị **học vẹt (overfitting)**, tức là mô hình học quá tốt trên dữ liệu huấn luyện nhưng lại dự đoán kém trên dữ liệu mới.

**B. Naive Bayes**

Đây là một bộ phân lớp xác suất dựa trên **Định lý Bayes**. Nó tính toán xác suất để một mẫu dữ liệu thuộc về một lớp cụ thể.
Nó được gọi là "ngây thơ" (naive) vì nó đưa ra một giả định đơn giản hóa nhưng mạnh mẽ: **tất cả các thuộc tính đều độc lập với nhau**. Mặc dù giả định này hiếm khi đúng trong thực tế, Naive Bayes vẫn hoạt động hiệu quả một cách đáng ngạc nhiên.

**Ưu điểm:** Nhanh, yêu cầu ít dữ liệu huấn luyện, hoạt động tốt với dữ liệu có số chiều lớn.

**C. Bộ phân lớp dựa trên Luật (Rule-Based Classifier)**

Mô hình này sử dụng một tập hợp các quy tắc `IF-THEN` để thực hiện phân lớp.

  * Ví dụ: `IF petal_length = 'low' AND petal_width = 'low' THEN species = 'setosa'`.
    Các luật này rất dễ hiểu đối với con người và có thể được trích xuất từ một cây quyết định.

-----

### **Phần 2: Thực hành Python với Iris**

Chúng ta sẽ xây dựng và so sánh hai mô hình: **Cây Quyết định** và **Naive Bayes**. Chúng ta sẽ sử dụng dữ liệu đã được chuẩn hóa ở Chương 2 để các thuật toán hoạt động tốt nhất.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# 1. Tải và Chuẩn bị dữ liệu
iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target_names[iris_dataset.target]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Chia dữ liệu thành tập Huấn luyện và Kiểm thử (80% train, 20% test)
# 'stratify=y' đảm bảo tỷ lệ các lớp trong tập train và test là như nhau
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Kích thước tập kiểm thử: {X_test.shape[0]} mẫu")
```

#### **Mô hình 1: Cây Quyết định (Decision Tree)**

```python
# 3. Huấn luyện mô hình Cây Quyết định
# Khởi tạo mô hình
dt_classifier = DecisionTreeClassifier(random_state=42)

# Huấn luyện mô hình trên tập train
dt_classifier.fit(X_train, y_train)

# 4. Dự đoán trên tập kiểm thử
y_pred_dt = dt_classifier.predict(X_test)

# 5. Đánh giá mô hình
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"\n--- Cây Quyết định ---")
print(f"Độ chính xác (Accuracy): {accuracy_dt:.2f}")

# In báo cáo phân loại chi tiết
print("\nBáo cáo Phân loại (Classification Report):")
print(classification_report(y_test, y_pred_dt))

# Trực quan hóa cây quyết định đã học
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_classifier,
               feature_names=iris_dataset.feature_names,
               class_names=iris_dataset.target_names,
               filled=True,
               rounded=True)
plt.title("Cây Quyết định cho dữ liệu Iris")
plt.show()
```

**Phân tích kết quả của Cây Quyết định:**

  * **Độ chính xác (Accuracy)** thường khá cao, cho thấy mô hình dự đoán tốt.
  * **Classification Report** cung cấp cái nhìn sâu hơn:
      * **Precision:** Trong số những lần mô hình dự đoán là một loài, có bao nhiêu lần là đúng?
      * **Recall:** Trong tất cả các mẫu thực tế của một loài, mô hình đã nhận diện được bao nhiêu?
      * **F1-score:** Trung bình hài hòa của Precision và Recall.
  * Biểu đồ cây cho thấy chính xác các quy tắc mà mô hình đã học. Ví dụ, bạn có thể thấy nó chỉ cần một câu hỏi (`petal length <= ...`) để tách hoàn toàn loài `setosa`.

#### **Mô hình 2: Naive Bayes**

```python
# 6. Huấn luyện mô hình Naive Bayes
# Sử dụng GaussianNB vì dữ liệu của chúng ta là các số thực (phân phối Gaussian)
nb_classifier = GaussianNB()

# Huấn luyện mô hình
nb_classifier.fit(X_train, y_train)

# 7. Dự đoán và Đánh giá
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"\n--- Naive Bayes ---")
print(f"Độ chính xác (Accuracy): {accuracy_nb:.2f}")
print("\nBáo cáo Phân loại (Classification Report):")
print(classification_report(y_test, y_pred_nb))
```

**Phân tích kết quả của Naive Bayes:**

  * Naive Bayes cũng cho kết quả rất tốt trên tập dữ liệu Iris, đôi khi độ chính xác có thể tương đương hoặc cao hơn Cây Quyết định.
  * Điều này cho thấy ngay cả với giả định "ngây thơ" về tính độc lập của các thuộc tính, mô hình vẫn có thể nắm bắt được cấu trúc cơ bản của dữ liệu.

Qua chương này, bạn đã học được cách xây dựng, huấn luyện và đánh giá các mô hình phân lớp, một trong những kỹ năng nền tảng và được ứng dụng nhiều nhất trong lĩnh vực Khoa học Dữ liệu.
