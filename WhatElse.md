Sau khi đã đi qua các chương cốt lõi, chúng ta có thể tổng hợp lại tất cả kiến thức và xem xét các bước nâng cao hơn để hoàn thiện một dự án khai phá dữ liệu. Đây là những gì một chuyên gia sẽ làm trong thực tế.

-----

### **Tổng hợp và Quy trình Hoàn chỉnh (Putting It All Together)**

Trong thực tế, bạn sẽ không thực hiện các bước một cách rời rạc. Thay vào đó, bạn sẽ kết hợp chúng thành một quy trình liền mạch để giải quyết một bài toán cụ thể.

**Bài toán:** Xây dựng mô hình **tốt nhất có thể** để phân loại hoa Iris.

Để làm điều này, chúng ta cần một phương pháp đánh giá mô hình đáng tin cậy hơn là chỉ chia `train_test_split` một lần. Chúng ta sẽ sử dụng **Đánh giá chéo (Cross-Validation)**.

**Đánh giá chéo (K-Fold Cross-Validation)** là kỹ thuật chia dữ liệu thành K phần (ví dụ, K=5). Sau đó, mô hình sẽ được huấn luyện K lần. Mỗi lần, nó sẽ lấy 1 phần làm tập kiểm thử (test set) và K-1 phần còn lại làm tập huấn luyện (train set). Độ chính xác cuối cùng là trung bình của cả K lần chạy. Điều này cho kết quả đánh giá ổn định và đáng tin cậy hơn nhiều.

```python
# Thực hành: Đánh giá mô hình bằng Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Chúng ta vẫn dùng dữ liệu đã chuẩn hóa X_scaled và nhãn y
# Khởi tạo các mô hình
dt_classifier = DecisionTreeClassifier(random_state=42)
nb_classifier = GaussianNB()

# Thực hiện đánh giá chéo 5-fold cho Cây Quyết định
scores_dt = cross_val_score(dt_classifier, X_scaled, y, cv=5)

# Thực hiện đánh giá chéo 5-fold cho Naive Bayes
scores_nb = cross_val_score(nb_classifier, X_scaled, y, cv=5)

print("--- Đánh giá chéo 5-Fold ---")
print(f"Các điểm Accuracy của Cây Quyết định: {np.round(scores_dt, 2)}")
print(f"Accuracy trung bình của Cây Quyết định: {scores_dt.mean():.2f}\n")

print(f"Các điểm Accuracy của Naive Bayes: {np.round(scores_nb, 2)}")
print(f"Accuracy trung bình của Naive Bayes: {scores_nb.mean():.2f}")
```

**Nhận xét:** Đánh giá chéo cho chúng ta một cái nhìn tổng thể và khách quan hơn về hiệu suất của mô hình. Cả hai mô hình đều có độ chính xác trung bình khoảng 95-96%, rất tốt.

-----

### **Các Chủ đề Nâng cao (What's Next?)**

Thế giới khai phá dữ liệu còn rất rộng lớn. Dưới đây là những khái niệm nâng cao bạn sẽ gặp tiếp theo:

#### **1. Tinh chỉnh Siêu tham số (Hyperparameter Tuning)**

Hầu hết các mô hình đều có các "nút vặn" gọi là **siêu tham số (hyperparameters)** mà chúng ta có thể điều chỉnh để cải thiện hiệu suất. Ví dụ, với Cây Quyết định, một siêu tham số quan trọng là `max_depth` (độ sâu tối đa của cây).

  * Nếu `max_depth` quá nhỏ, cây quá đơn giản và không học được hết các mẫu (underfitting).
  * Nếu `max_depth` quá lớn, cây quá phức tạp và dễ bị học vẹt (overfitting).

Các kỹ thuật như **Grid Search** hoặc **Random Search** sẽ tự động thử nghiệm nhiều bộ siêu tham số khác nhau để tìm ra bộ tốt nhất.

#### **2. Các Phương pháp Ensemble (Ensemble Methods)** 🧠+🧠=💪

Nguyên tắc của Ensemble là **"Nhiều cái đầu luôn tốt hơn một"**. Thay vì chỉ dùng một mô hình, chúng ta sẽ kết hợp dự đoán của nhiều mô hình để đưa ra quyết định cuối cùng chính xác và ổn định hơn.

  * **Random Forest:** Là một tập hợp của nhiều Cây Quyết định. Đây là một trong những thuật toán mạnh mẽ và phổ biến nhất hiện nay.
  * **Gradient Boosting (ví dụ: XGBoost, LightGBM):** Xây dựng các mô hình một cách tuần tự, trong đó mô hình sau sẽ cố gắng sửa lỗi của mô hình trước.

#### **3. Giảm chiều Dữ liệu (Dimensionality Reduction)**

Khi làm việc với dữ liệu có hàng trăm hoặc hàng nghìn thuộc tính, việc huấn luyện mô hình sẽ rất chậm và dễ bị overfitting. Các kỹ thuật giảm chiều dữ liệu giúp chúng ta nén bộ thuộc tính đó xuống còn một vài thuộc tính mới nhưng vẫn giữ lại được phần lớn thông tin quan trọng.

  * **PCA (Principal Component Analysis):** Là kỹ thuật phổ biến nhất. Nó tìm ra các "thành phần chính" (principal components) - là các hướng mà dữ liệu có phương sai lớn nhất - và chiếu dữ liệu lên các hướng đó.

Hãy xem PCA có thể làm gì với dữ liệu Iris 4 chiều của chúng ta:

```python
# Thực hành: Giảm chiều dữ liệu Iris từ 4D xuống 2D bằng PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Khởi tạo PCA để giữ lại 2 thành phần chính
pca = PCA(n_components=2)

# Fit và transform dữ liệu đã chuẩn hóa
X_pca = pca.fit_transform(X_scaled)

# Trực quan hóa kết quả
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=70)
plt.title('Dữ liệu Iris sau khi giảm chiều từ 4D xuống 2D bằng PCA')
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.legend()
plt.grid()
plt.show()

print(f"Tỷ lệ phương sai được giải thích bởi 2 thành phần: {pca.explained_variance_ratio_.sum():.2f}")
```

**Nhận xét:** Thật đáng kinh ngạc\! Chúng ta đã nén dữ liệu từ 4 chiều xuống chỉ còn 2 chiều mà vẫn giữ được **96%** thông tin (phương sai) của dữ liệu gốc. Quan trọng hơn, trên biểu đồ 2D, bạn có thể thấy ba loài hoa vẫn được phân tách rất rõ ràng. Điều này cho thấy PCA đã làm rất tốt việc tóm tắt dữ liệu.

Tóm lại, những gì bạn đã học là nền tảng vững chắc. Các bước tiếp theo sẽ là khám phá các thuật toán mạnh mẽ hơn, học cách tinh chỉnh chúng, và áp dụng quy trình này vào các bộ dữ liệu lớn và phức tạp hơn trong thực tế.
