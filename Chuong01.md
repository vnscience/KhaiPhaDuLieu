### **Chương 1: Giới thiệu về Khai phá dữ liệu (Introduction to Data Mining)**

Chương này đặt nền móng cho toàn bộ môn học, giúp bạn hiểu rõ "Khai phá dữ liệu" là gì, tại sao nó quan trọng, và quy trình tổng thể để biến dữ liệu thô thành tri thức hữu ích.

----------

### **Phần 1: Lý Thuyết (Theory)**

#### **1. Khai phá dữ liệu là gì? (What is Data Mining?)**

Hãy tưởng tượng bạn có một ngọn núi khổng lồ chứa đầy đất đá, và bạn biết rằng bên trong nó ẩn chứa những mỏ vàng quý giá. Bạn không thể lấy cả ngọn núi, mà bạn cần những công cụ và kỹ thuật đặc biệt để _khai phá_, sàng lọc, và tìm ra những thỏi vàng đó.

Trong thế giới kỹ thuật số, "ngọn núi" chính là các **tập dữ liệu khổng lồ (large datasets)**, và "vàng" là những **tri thức (knowledge)**, **mẫu tiềm ẩn (hidden patterns)**, và **thông tin có giá trị (valuable information)**.

> **Định nghĩa:** **Khai phá dữ liệu (Data Mining)** là quá trình khám phá các mẫu, các mối quan hệ và các insight hữu ích từ các tập dữ liệu lớn bằng cách sử dụng các kỹ thuật từ thống kê, máy học và hệ quản trị cơ sở dữ liệu. Nó không chỉ đơn thuần là truy xuất dữ liệu, mà là việc _khám phá_ những điều chúng ta chưa biết trước đó.

#### **2. Quy trình Khám phá Tri thức (The KDD Process)**

Khai phá dữ liệu thực chất chỉ là một bước quan trọng trong một quy trình lớn hơn gọi là **Khám phá Tri thức trong Cơ sở dữ liệu (Knowledge Discovery in Databases - KDD)**. Quy trình này bao gồm các bước tuần tự để đảm bảo rằng tri thức cuối cùng là đáng tin cậy và hữu ích.

1.  **Lựa chọn dữ liệu (Data Selection):** Xác định và lựa chọn dữ liệu mục tiêu từ các nguồn khác nhau (cơ sở dữ liệu, file, web,...).
    
2.  **Tiền xử lý (Preprocessing):** "Làm sạch" và chuẩn bị dữ liệu. Đây là bước quan trọng nhất, thường chiếm nhiều thời gian nhất, bao gồm xử lý giá trị thiếu, loại bỏ nhiễu, làm mịn dữ liệu.
    
3.  **Biến đổi (Transformation):** Chuyển đổi dữ liệu sang định dạng phù hợp cho việc khai phá. Ví dụ: chuẩn hóa dữ liệu (normalization) để các thuộc tính có cùng một thang đo.
    
4.  **Khai phá dữ liệu (Data Mining):** 🎯 **Đây là trái tim của quy trình.** Áp dụng các thuật toán thông minh để trích xuất các mẫu dữ liệu.
    
5.  **Đánh giá mẫu (Pattern Evaluation):** Xác định xem các mẫu được tìm thấy có thực sự thú vị, hữu ích và mới mẻ hay không dựa trên các thước đo nhất định.
    
6.  **Trình bày tri thức (Knowledge Presentation):** Trực quan hóa và trình bày tri thức đã được khám phá cho người dùng cuối (ví dụ: qua biểu đồ, báo cáo).
    

#### **3. Các Nhiệm vụ Chính của Khai phá dữ liệu (Key Data Mining Tasks)**

Có hai nhóm nhiệm vụ chính trong khai phá dữ liệu:

**A. Mô tả (Descriptive):** Tìm ra các mẫu mô tả dữ liệu.

-   **Phân cụm (Clustering):** Tự động nhóm các đối tượng tương tự nhau vào các cụm mà không cần biết trước nhãn.
    
    -   _Ví dụ thực tế:_ Phân nhóm khách hàng (customer segmentation) thành các nhóm như "khách hàng trung thành", "khách hàng tiềm năng", "khách hàng sắp rời bỏ" để có chiến lược marketing phù hợp.
        
-   **Khai phá luật kết hợp (Association Rule Mining):** Tìm ra các quy tắc thể hiện mối liên hệ giữa các mục.
    
    -   _Ví dụ kinh điển:_ Phân tích giỏ hàng trong siêu thị để tìm ra quy tắc `"Nếu khách hàng mua Bia thì họ cũng có xu hướng mua Tã lót"`. Điều này có thể gợi ý cho siêu thị đặt hai sản phẩm này gần nhau.
        

**B. Dự đoán (Predictive):** Sử dụng dữ liệu hiện có để đưa ra dự đoán cho dữ liệu trong tương lai.

-   **Phân lớp (Classification):** Dự đoán một nhãn/danh mục cho một đối tượng mới dựa trên các thuộc tính của nó.
    
    -   _Ví dụ thực tế:_ Phân loại email là `spam` hay `không phải spam` dựa trên nội dung và người gửi.
        
-   **Hồi quy (Regression):** Dự đoán một giá trị số liên tục.
    
    -   _Ví dụ thực tế:_ Dự đoán giá của một ngôi nhà dựa trên diện tích, số phòng ngủ và vị trí.
        
-   **Phát hiện bất thường (Anomaly Detection):** Xác định các bản ghi/sự kiện hiếm gặp, khác biệt so với phần lớn dữ liệu.
    
    -   _Ví dụ thực tế:_ Phát hiện giao dịch thẻ tín dụng gian lận. Một giao dịch bất thường có thể là một khoản tiền rất lớn được thực hiện ở một quốc gia khác.
        

----------

### **Phần 2: Thực hành Python (Python Practice)**

Để bắt đầu hành trình khai phá dữ liệu, chúng ta cần chuẩn bị "bộ dụng cụ" của mình. Trong Python, đó là các thư viện mạnh mẽ được xây dựng cho khoa học dữ liệu.

#### **1. Thiết lập Môi trường (Setting Up the Environment)**

Các thư viện cốt lõi bạn cần là:

-   **`pandas`:** Dùng để đọc, xử lý và thao tác với dữ liệu dạng bảng (giống như Excel).
    
-   **`scikit-learn`:** "Con dao Thụy Sĩ" của máy học, chứa hầu hết các thuật toán khai phá dữ liệu và các công cụ hỗ trợ.
    
-   **`matplotlib` & `seaborn`:** Dùng để trực quan hóa dữ liệu, giúp chúng ta "nhìn" thấy các mẫu.
    

Trong Google Colab, các thư viện này thường đã được cài sẵn. Nếu bạn làm việc trên máy tính cá nhân, hãy mở Terminal (hoặc Command Prompt) và chạy:


```
pip install pandas scikit-learn matplotlib seaborn
```

#### **2. Tải và Khám phá Dữ liệu Lần đầu**

Chúng ta sẽ bắt đầu với bộ dữ liệu `Iris` kinh điển. Đây là bước đầu tiên trong quy trình KDD - **Lựa chọn dữ liệu**.

```
# Import các thư viện cần thiết
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Tải dữ liệu từ scikit-learn
iris_dataset = load_iris()

# Dữ liệu này là một đối tượng đặc biệt của scikit-learn
# iris_dataset.data chứa các thuộc tính (features)
# iris_dataset.target chứa nhãn (species)
# iris_dataset.feature_names và iris_dataset.target_names là tên của chúng

# 2. Chuyển đổi sang Pandas DataFrame để làm việc dễ dàng hơn
# Đây là bước chuyển đổi quen thuộc trong mọi dự án data science
df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
df['species'] = iris_dataset.target_names[iris_dataset.target]

# 3. Khám phá dữ liệu ban đầu - "Cái nhìn đầu tiên"
print("--- 5 dòng dữ liệu đầu tiên ---")
print(df.head())

print("\n--- Thông tin tổng quan về dữ liệu ---")
# .info() cho biết kiểu dữ liệu của mỗi cột và có giá trị nào bị thiếu (null) không
df.info()

print("\n--- Thống kê mô tả cơ bản ---")
# .describe() cho chúng ta các giá trị thống kê quan trọng: trung bình, độ lệch chuẩn, min, max,...
# Giúp nhanh chóng phát hiện các giá trị bất thường.
print(df.describe())

# 4. Trực quan hóa để hiểu sự phân bổ của các loài hoa
print("\n--- Phân bổ số lượng của mỗi loài hoa ---")
sns.countplot(x='species', data=df)
plt.title('Số lượng mẫu cho mỗi loài hoa Iris')
plt.show()

```

**Phân tích kết quả thực hành:**

-   `df.head()`: Giúp ta hình dung cấu trúc của dữ liệu: 4 cột thuộc tính số (sepal length, sepal width, petal length, petal width) và 1 cột mục tiêu (species).
    
-   `df.info()`: Cung cấp một tin tức tuyệt vời: có 150 mẫu (`entries`) và không có giá trị nào bị thiếu (`non-null` ở tất cả các cột). Điều này có nghĩa là bước **Tiền xử lý (làm sạch)** cho dữ liệu này sẽ khá đơn giản.
    
-   `df.describe()`: Cho thấy thang đo (scale) của các thuộc tính. Ví dụ, `petal length` (chiều dài cánh hoa) có giá trị trung bình là 3.76 cm, trong khi `sepal width` (chiều rộng đài hoa) chỉ là 3.06 cm. Trong các chương sau, chúng ta sẽ học cách **Biến đổi (chuẩn hóa)** các giá trị này.
    
-   Biểu đồ `countplot`: Cho thấy dữ liệu của chúng ta rất **cân bằng**, mỗi loài hoa có chính xác 50 mẫu. Đây là một điều kiện lý tưởng để xây dựng các mô hình dự đoán.

Qua những bước đơn giản trên, chúng ta đã hoàn thành các bước đầu tiên của quy trình KDD và sẵn sàng cho các phân tích sâu hơn ở các chương tiếp theo!
