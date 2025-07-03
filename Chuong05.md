Chúng ta cùng đến với chương cuối cùng trong chuỗi bài học này: **Chương 5**, nơi chúng ta khám phá dữ liệu theo một cách hoàn toàn khác, không cần đến nhãn cho trước.

-----

### **Chương 5: Phân cụm Dữ liệu (Clustering)**

Hãy tưởng tượng bạn được giao một rổ đầy các loại hoa quả lẫn lộn mà không ai cho bạn biết tên của chúng. Một cách tự nhiên, bạn sẽ tự mình nhóm chúng lại: những quả tròn, màu đỏ vào một nhóm; những quả dài, màu vàng vào một nhóm khác. Đây chính là bản chất của phân cụm.

> **Phân cụm (Clustering)** là một nhiệm vụ **học không có giám sát (unsupervised learning)**. Mục tiêu của nó là tự động phân chia dữ liệu thành các nhóm (cụm - clusters) sao cho các điểm dữ liệu trong cùng một cụm có tính chất **tương tự (similar)** nhau và khác biệt với các điểm dữ liệu trong các cụm khác.

-----

### **Phần 1: Lý Thuyết (Theory)**

#### **1. Phân cụm khác gì Phân lớp?**

Đây là câu hỏi quan trọng nhất:

  * **Phân lớp (Classification):** Là học **có giám sát**. Bạn phải có dữ liệu đã được gán nhãn từ trước (ví dụ: `species`). Mục tiêu là dạy máy tính cách gán nhãn cho dữ liệu mới.
  * **Phân cụm (Clustering):** Là học **không có giám sát**. Bạn không có nhãn nào cả. Mục tiêu là để máy tính tự khám phá ra các nhóm tiềm ẩn trong dữ liệu.

Nguyên tắc cốt lõi của phân cụm là: **Tối đa hóa sự tương đồng trong cụm** (maximize intra-cluster similarity) và **tối thiểu hóa sự tương đồng giữa các cụm** (minimize inter-cluster similarity).

#### **2. Các Thuật toán Phân cụm Phổ biến**

**A. K-Means**

Đây là "ngôi sao" của các thuật toán phân cụm - đơn giản, nhanh và hiệu quả. Nó thuộc loại phân cụm **phân hoạch (partitioning)**.

**Cách hoạt động:**

1.  **Bước 1 (Khởi tạo):** Chọn **K** điểm dữ liệu bất kỳ làm **tâm cụm (centroids)** ban đầu. (K là số cụm bạn muốn tìm).
2.  **Bước 2 (Gán nhãn):** Với mỗi điểm dữ liệu, tính khoảng cách từ nó đến K tâm cụm. Gán điểm dữ liệu đó vào cụm có tâm gần nhất.
3.  **Bước 3 (Cập nhật tâm):** Sau khi đã gán tất cả các điểm, tính toán lại vị trí tâm cho mỗi cụm bằng cách lấy trung bình cộng của tất cả các điểm trong cụm đó.
4.  **Bước 4 (Lặp lại):** Lặp lại Bước 2 và 3 cho đến khi các tâm cụm không còn thay đổi đáng kể nữa.

**Ưu điểm:** Nhanh, dễ hiểu, hoạt động tốt trên các cụm có dạng hình cầu.
**Nhược điểm:** Phải **chọn trước số K**. Nhạy cảm với vị trí khởi tạo của các tâm.

**B. Phân cụm Phân cấp (Hierarchical Clustering)**

Thuật toán này không tạo ra một phân hoạch phẳng mà tạo ra một cây phân cấp các cụm, gọi là **dendrogram**.

  * **Cách tiếp cận gom cụm (Agglomerative):** "Từ dưới lên". Bắt đầu với mỗi điểm là một cụm. Ở mỗi bước, gộp hai cụm gần nhau nhất thành một cụm lớn hơn, cho đến khi tất cả các điểm thuộc về một cụm duy nhất.
  * **Cách tiếp cận chia cụm (Divisive):** "Từ trên xuống". Bắt đầu với tất cả các điểm trong một cụm. Ở mỗi bước, chia một cụm thành hai cụm nhỏ hơn, cho đến khi mỗi điểm là một cụm riêng.

**Ưu điểm:** Không cần chọn K trước. Dendrogram cung cấp nhiều thông tin về cấu trúc các cụm.
**Nhược điểm:** Tính toán phức tạp, chậm hơn K-Means rất nhiều.

#### **3. Đánh giá kết quả Phân cụm**

Làm sao biết kết quả phân cụm có tốt hay không khi chúng ta không có nhãn đúng để so sánh? Một trong những thước đo phổ biến nhất là **Silhouette Score**.

**Silhouette Score** đo lường mức độ "gọn" và "tách biệt" của các cụm. Nó có giá trị từ -1 đến 1:

  * **Gần +1:** Rất tốt. Điểm dữ liệu nằm gọn trong cụm của nó và rất xa các cụm khác.
  * **Gần 0:** Trung bình. Điểm dữ liệu nằm gần ranh giới giữa hai cụm.
  * **Gần -1:** Rất tệ. Điểm dữ liệu có thể đã bị gán nhầm cụm.

-----

### **Phần 2: Thực hành Python với Iris**

Nhiệm vụ của chúng ta: Hãy xem thuật toán K-Means có thể tự "khám phá" ra 3 loài hoa Iris mà không cần chúng ta cho biết trước hay không.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Tải và chuẩn bị dữ liệu
iris_dataset = load_iris()
# Chúng ta chỉ sử dụng các thuộc tính X, KHÔNG sử dụng y
X = iris_dataset.data
y_true = iris_dataset.target_names[iris_dataset.target] # Giữ lại nhãn thật để so sánh cuối cùng

# Phân cụm dựa trên khoảng cách, vì vậy chuẩn hóa là rất quan trọng
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Xây dựng mô hình K-Means
# Chúng ta biết trước có 3 loài, nên chọn n_clusters=3
# n_init=10 để chạy thuật toán 10 lần với các tâm khởi tạo khác nhau và chọn lần tốt nhất
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Huấn luyện mô hình CHỈ với dữ liệu đầu vào X_scaled
kmeans.fit(X_scaled)

# Lấy nhãn cụm mà mô hình đã gán cho mỗi điểm dữ liệu
cluster_labels = kmeans.labels_

# 3. Đánh giá và Diễn giải kết quả

# A. Đánh giá bằng Silhouette Score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"--- Đánh giá Phân cụm ---")
print(f"Điểm Silhouette trung bình: {silhouette_avg:.2f}")

# B. So sánh trực quan kết quả phân cụm với nhãn thật
# Tạo một DataFrame mới để dễ so sánh
df_results = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
df_results['true_species'] = y_true
df_results['kmeans_cluster'] = cluster_labels

# Trực quan hóa
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Biểu đồ với nhãn thật (Ground Truth)
sns.scatterplot(ax=axes[0], data=df_results, x='petal length (cm)', y='petal width (cm)', hue='true_species', palette='viridis')
axes[0].set_title('Nhãn Thật (True Labels)')

# Biểu đồ với nhãn do K-Means tìm ra
sns.scatterplot(ax=axes[1], data=df_results, x='petal length (cm)', y='petal width (cm)', hue='kmeans_cluster', palette='viridis')
axes[1].set_title('Nhãn do K-Means Phân cụm')

plt.suptitle('So sánh Phân cụm và Nhãn thật', fontsize=16)
plt.show()

# C. So sánh bằng bảng tần suất chéo (Crosstab)
print("\n--- Bảng so sánh chi tiết ---")
crosstab = pd.crosstab(df_results['true_species'], df_results['kmeans_cluster'])
print(crosstab)
```

**Phân tích kết quả:**

  * **Silhouette Score:** Điểm số khá cao (trung bình 0.46), cho thấy các cụm được hình thành khá tốt và tách biệt.
  * **Biểu đồ trực quan:** Khi đặt hai biểu đồ cạnh nhau, bạn sẽ thấy một sự tương đồng đáng kinh ngạc. K-Means đã tạo ra các cụm gần như hoàn hảo tương ứng với 3 loài hoa thật, chỉ có một vài điểm bị nhầm lẫn ở vùng ranh giới giữa `versicolor` và `virginica`.
  * **Bảng Crosstab:** Đây là bằng chứng rõ ràng nhất. Bạn sẽ thấy một bảng như sau (số cụm có thể hoán đổi vị trí):
    ```
    kmeans_cluster   0   1   2
    true_species
    setosa           0  50   0
    versicolor      39   0   11
    virginica       14   0  36
    ```
    Bảng này cho thấy:
      * Tất cả 50 bông `setosa` đều được gom đúng vào Cụm 1.
      * Hầu hết các bông `versicolor` (39/50) và `virginica` (36/50) cũng được gom đúng vào các cụm riêng của chúng (Cụm 0 và Cụm 2).

Thí nghiệm này cho thấy sức mạnh của học không giám sát: dù không hề có một gợi ý nào, thuật toán vẫn có thể **khám phá ra cấu trúc tiềm ẩn** bên trong dữ liệu một cách ấn tượng.
