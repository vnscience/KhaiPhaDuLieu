Bây giờ chúng ta sẽ đi sâu vào **Chương 2**, một trong những bước quan trọng và tốn nhiều công sức nhất trong bất kỳ dự án khoa học dữ liệu nào.

-----

### **Chương 2: Tiền xử lý dữ liệu (Data Preprocessing)**

Nếu "Khai phá dữ liệu" là quá trình đi tìm vàng, thì "Tiền xử lý" chính là công đoạn sàng lọc đất đá, rửa sạch quặng thô trước khi đưa vào lò luyện. Chất lượng của tri thức khám phá được phụ thuộc trực tiếp vào chất lượng của dữ liệu đầu vào. Nguyên tắc vàng ở đây là: **"Rác vào, Rác ra" (Garbage In, Garbage Out)**.

-----

### **Phần 1: Lý Thuyết (Theory)**

Dữ liệu trong thế giới thực thường rất "bẩn". Nó có thể:

  * **Thiếu (Missing):** Bị thiếu giá trị ở một vài thuộc tính (ví dụ: không ghi lại tuổi của khách hàng).
  * **Nhiễu (Noisy):** Chứa lỗi hoặc các giá trị ngoại lai (outliers) (ví dụ: tuổi = -1, hoặc lương = 2 tỷ USD/tháng cho một nhân viên bình thường).
  * **Không nhất quán (Inconsistent):** Có sự khác biệt trong mã hóa hoặc tên gọi (ví dụ: trong cùng một cột `thanh_pho`, dữ liệu được ghi là "TP.HCM", "HCMC", và "Hồ Chí Minh").

Vì vậy, tiền xử lý là quá trình biến dữ liệu thô, "bẩn" thành dữ liệu sạch, sẵn sàng cho việc phân tích.

#### **1. Các Nhiệm vụ Tiền xử lý Chính**

**A. Làm sạch dữ liệu (Data Cleaning)**

Đây là quá trình xử lý các vấn đề về dữ liệu thiếu và nhiễu.

  * **Xử lý giá trị thiếu (Handling Missing Values):**

      * **Bỏ qua bản ghi:** Cách đơn giản nhất là xóa dòng dữ liệu đó. Tuy nhiên, cách này không nên dùng nếu dữ liệu thiếu chiếm tỷ lệ lớn vì sẽ làm mất thông tin quý giá.
      * **Điền thủ công:** Tốn thời gian và không khả thi với tập dữ liệu lớn.
      * **Điền bằng một giá trị toàn cục:** Dùng một hằng số như "Unknown" hoặc "N/A".
      * **Điền bằng giá trị trung bình/trung vị:** Đây là cách phổ biến. Dùng giá trị trung bình (mean) cho dữ liệu số có phân phối đối xứng và giá trị trung vị (median) cho dữ liệu có phân phối lệch (bị ảnh hưởng bởi outliers).
      * **Điền bằng giá trị có khả năng cao nhất:** Dùng các thuật toán (như hồi quy, cây quyết định) để dự đoán giá trị bị thiếu.

  * **Làm mịn dữ liệu nhiễu (Smoothing Noisy Data):**

      * **Binning (Phân giỏ):** Sắp xếp dữ liệu và chia thành các "giỏ" có kích thước bằng nhau, sau đó làm mịn bằng giá trị trung bình hoặc biên của giỏ.
      * **Hồi quy (Regression):** Dùng một hàm hồi quy để làm khớp và làm mịn dữ liệu.
      * **Phát hiện và xử lý Outliers:** Có thể loại bỏ hoặc điều chỉnh các giá trị ngoại lai.

**B. Biến đổi dữ liệu (Data Transformation)**

Biến đổi dữ liệu để chúng ở định dạng phù hợp hơn cho các thuật toán khai phá.

  * **Chuẩn hóa (Normalization/Standardization):** Rất quan trọng\! Nhiều thuật toán (như K-Means, SVM) rất nhạy cảm với thang đo của dữ liệu. Nếu một thuộc tính có thang đo từ 0-1,000,000 (ví dụ: lương) và một thuộc tính khác có thang đo 0-100 (ví dụ: tuổi), thuộc tính lương sẽ "lấn át" và ảnh hưởng đến kết quả nhiều hơn. Chuẩn hóa sẽ đưa tất cả các thuộc tính về một thang đo chung.
      * **Min-Max Normalization:** Ánh xạ dữ liệu vào một khoảng giá trị xác định, thường là `[0, 1]`.
        $$v' = \frac{v - \min_A}{\max_A - \min_A}$$
        Trong đó $v$ là giá trị cũ, $v'$ là giá trị mới, $\\min\_A$ và $\\max\_A$ là giá trị nhỏ nhất và lớn nhất của thuộc tính A.
      * **Z-score Standardization:** Biến đổi dữ liệu sao cho nó có giá trị trung bình (mean) là **0** và độ lệch chuẩn (standard deviation) là **1**. Đây là phương pháp phổ biến và được ưa chuộng nhất.
        $$v' = \frac{v - \mu_A}{\sigma_A}$$
        Trong đó $\\mu\_A$ là trung bình và $\\sigma\_A$ là độ lệch chuẩn của thuộc tính A.

-----

### **Phần 2: Thực hành Python với Iris**

Bây giờ, hãy áp dụng các kỹ thuật này vào bộ dữ liệu Iris của chúng ta. Chúng ta sẽ tiếp tục với `DataFrame` đã tạo ở Chương 1.

```python
# Import các thư viện cần thiết
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Tải lại dữ liệu để đảm bảo tính toàn vẹn
iris_dataset = load_iris()
df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
df['species'] = iris_dataset.target_names[iris_dataset.target]
```

#### **1. Khám phá và Làm sạch Dữ liệu**

Đầu tiên, hãy kiểm tra xem dữ liệu Iris có "sạch" không.

```python
# 1. Kiểm tra giá trị thiếu (Missing Values)
print("--- Kiểm tra dữ liệu thiếu ---")
# .isnull() trả về True/False cho mỗi ô, .sum() đếm số lượng True trên mỗi cột
print(df.isnull().sum())
```

**Kết quả:**

```
--- Kiểm tra dữ liệu thiếu ---
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
species              0
dtype: int64
```

**Nhận xét:** Thật tuyệt vời\! Bộ dữ liệu Iris là một bộ dữ liệu mẫu rất sạch và không có bất kỳ giá trị thiếu nào. Trong thực tế, bạn sẽ cần áp dụng các kỹ thuật đã học ở phần lý thuyết để xử lý chúng.

Tiếp theo, hãy dùng biểu đồ hộp (boxplot) để kiểm tra các giá trị nhiễu và ngoại lai (outliers).

```python
# 2. Kiểm tra giá trị ngoại lai (Outliers) bằng Boxplot
plt.figure(figsize=(12, 6))
# Bỏ cột species vì nó không phải là số
sns.boxplot(data=df.drop('species', axis=1))
plt.title('Biểu đồ hộp cho các thuộc tính của Iris')
plt.show()
```

**Nhận xét:** Biểu đồ hộp cho thấy:

  * Phân phối của các thuộc tính khá rõ ràng.
  * Chỉ có một vài điểm dữ liệu nằm ngoài "râu" (whiskers) của biểu đồ `sepal width (cm)`, cho thấy chúng có thể là các outliers nhẹ. Tuy nhiên, chúng không quá cực đoan, nên chúng ta có thể giữ lại chúng.

#### **2. Biến đổi Dữ liệu: Chuẩn hóa Z-score**

Đây là bước thực hành cốt lõi của chương này. Chúng ta sẽ chuẩn hóa các thuộc tính số về cùng một thang đo.

```python
# 1. Tách các cột thuộc tính (features) ra khỏi cột mục tiêu (target)
# X chứa các cột số cần được chuẩn hóa
X = df.drop('species', axis=1)
# y chứa cột nhãn mà chúng ta muốn dự đoán sau này
y = df['species']

# 2. Khởi tạo đối tượng StandardScaler từ scikit-learn
scaler = StandardScaler()

# 3. Fit và transform dữ liệu
# .fit(X) để scaler "học" các tham số (trung bình μ và độ lệch chuẩn σ) từ dữ liệu X
# .transform(X) để áp dụng công thức chuẩn hóa Z-score vào dữ liệu X
X_scaled = scaler.fit_transform(X)

# Chuyển mảng kết quả về lại DataFrame để dễ quan sát
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 4. So sánh dữ liệu gốc và dữ liệu đã chuẩn hóa
print("\n--- 5 dòng dữ liệu GỐC ---")
print(X.head())

print("\n--- 5 dòng dữ liệu ĐÃ CHUẨN HÓA ---")
print(X_scaled_df.head())

# 5. Kiểm tra lại kết quả chuẩn hóa
print("\n--- Trung bình và Độ lệch chuẩn sau khi chuẩn hóa ---")
# mean() sẽ rất gần 0, và std() sẽ rất gần 1
print(X_scaled_df.describe().loc[['mean', 'std']])
```

Để trực quan hóa sự thay đổi, hãy vẽ biểu đồ phân phối của thuộc tính `petal length (cm)` trước và sau khi chuẩn hóa.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Biểu đồ trước khi chuẩn hóa
sns.kdeplot(data=df, x='petal length (cm)', ax=axes[0], fill=True)
axes[0].set_title('Trước khi chuẩn hóa (Original)')

# Biểu đồ sau khi chuẩn hóa
sns.kdeplot(data=X_scaled_df, x='petal length (cm)', ax=axes[1], fill=True, color='red')
axes[1].set_title('Sau khi chuẩn hóa (Standardized)')

plt.suptitle('So sánh phân phối của Petal Length', fontsize=16)
plt.show()
```

**Phân tích kết quả:**

  * So sánh 5 dòng đầu tiên của dữ liệu gốc và dữ liệu đã chuẩn hóa, bạn có thể thấy các giá trị đã được biến đổi hoàn toàn.
  * Bảng `describe` cho thấy `mean` của các cột sau khi chuẩn hóa đều là những số rất gần 0 (ví dụ: `-1.468455e-15`) và `std` (độ lệch chuẩn) rất gần 1. Điều này chứng tỏ quá trình chuẩn hóa đã thành công.
  * Biểu đồ phân phối cho thấy **hình dạng (shape)** của dữ liệu không thay đổi, nhưng **thang đo trên trục x (scale)** đã được điều chỉnh về trung tâm là 0.

Sau khi hoàn thành chương này, dữ liệu của chúng ta đã "sạch" và "chuẩn", sẵn sàng để đưa vào các mô hình khai phá ở các chương tiếp theo\!
