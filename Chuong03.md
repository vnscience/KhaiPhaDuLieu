Chúng ta cùng khám phá **Chương 3**. Chương này giới thiệu một loại nhiệm vụ khai phá dữ liệu hoàn toàn khác: thay vì dự đoán, chúng ta sẽ đi tìm các **quy luật** hay **mẫu** thú vị ngay bên trong dữ liệu.

-----

### **Chương 3: Khai phá Mẫu (Pattern Mining)**

Hãy tưởng tượng bạn là quản lý một siêu thị. Bạn muốn biết khách hàng thường mua những sản phẩm nào cùng nhau để có thể sắp xếp các kệ hàng hoặc tạo ra các chương trình khuyến mãi hiệu quả. Đây chính là bài toán cốt lõi của Khai phá Mẫu.

> **Định nghĩa:** **Khai phá Mẫu (Pattern Mining)** là quá trình tìm kiếm các mẫu xuất hiện thường xuyên (frequent patterns) trong dữ liệu. Dạng phổ biến nhất của nó là **Khai phá Tập hợp Phổ biến (Frequent Itemset Mining)**, từ đó dẫn đến **Khai phá Luật kết hợp (Association Rule Mining)**.

-----

### **Phần 1: Lý Thuyết (Theory)**

#### **1. Các Khái niệm Cốt lõi**

Để hiểu được các thuật toán, chúng ta cần nắm vững một vài thuật ngữ quan trọng. Hãy xét ví dụ về dữ liệu giao dịch của một siêu thị:

| Mã Giao dịch (TID) | Các sản phẩm (Items)             |
| ------------------ | --------------------------------- |
| 1                  | {Bánh mì, Sữa}                    |
| 2                  | {Bánh mì, Tã, Bia, Trứng}         |
| 3                  | {Sữa, Tã, Bia, Nước ngọt}         |
| 4                  | {Bánh mì, Sữa, Tã, Bia}           |
| 5                  | {Bánh mì, Sữa, Tã, Nước ngọt}    |

  * **Tập hợp các mục (Itemset):** Một tập hợp chứa một hoặc nhiều mục.

      * Ví dụ: `{Bánh mì, Sữa}`, `{Tã, Bia}`.

  * **Độ hỗ trợ (Support):** Thước đo mức độ phổ biến của một itemset. Nó là tỷ lệ phần trăm các giao dịch chứa itemset đó.
    $$Support(X) = \frac{\text{Số giao dịch chứa } X}{\text{Tổng số giao dịch}}$$

      * Ví dụ: Itemset `{Bánh mì, Sữa}` xuất hiện trong 3 giao dịch (TID 1, 4, 5). Tổng có 5 giao dịch.
        Vậy, $Support({\\text{Bánh mì, Sữa}}) = 3/5 = 60%$.

  * **Tập phổ biến (Frequent Itemset):** Một itemset được gọi là "phổ biến" nếu độ hỗ trợ của nó lớn hơn hoặc bằng một ngưỡng tối thiểu do người dùng định nghĩa, gọi là **minimum support (`minsup`)**.

  * **Luật kết hợp (Association Rule):** Một biểu thức có dạng $X \\rightarrow Y$, trong đó X và Y là các itemset không giao nhau.

      * Ví dụ: `{Tã, Bia}` $$\rightarrow$$ `{Sữa}`. Luật này có thể đọc là "Nếu một người mua Tã và Bia, thì họ cũng có khả năng sẽ mua Sữa".

  * **Độ tin cậy (Confidence):** Thước đo mức độ chắc chắn của một luật kết hợp. Nó tính xác suất có điều kiện $P(Y|X)$.
    $$Confidence(X \rightarrow Y) = \frac{Support(X \cup Y)}{Support(X)}$$

      * Ví dụ: Tính Confidence của luật `{Tã, Bia}` $$\rightarrow$$ `{Sữa}`.
          * $Support({\\text{Tã, Bia, Sữa}}) = 2/5 = 40%$ (xuất hiện ở TID 3, 4).
          * $Support({\\text{Tã, Bia}}) = 3/5 = 60%$ (xuất hiện ở TID 2, 3, 4).
          * $Confidence = 40% / 60% \\approx 66.7%$.
      * Điều này có nghĩa là: "Trong số những người đã mua Tã và Bia, có 66.7% người cũng mua Sữa".

#### **2. Thuật toán Apriori**

Đây là thuật toán kinh điển nhất để tìm các tập phổ biến. Nguyên tắc cốt lõi của nó được gọi là **Nguyên tắc Apriori**:

> **Nguyên tắc Apriori:** Nếu một itemset là phổ biến, thì tất cả các tập con của nó cũng phải phổ biến. Ngược lại, nếu một itemset không phổ biến, thì tất cả các tập cha chứa nó cũng sẽ không phổ biến.

Thuật toán này giúp cắt tỉa không gian tìm kiếm một cách hiệu quả, tránh phải đếm support cho mọi itemset có thể có.

-----

### **Phần 2: Áp dụng vào tập dữ liệu Iris**

#### **1. Thách thức: Dữ liệu Số liên tục**

Các thuật toán như Apriori được thiết kế cho dữ liệu **giao dịch** hoặc **phân loại (categorical)**. Tuy nhiên, các thuộc tính của Iris (sepal length, petal width...) là dữ liệu **số liên tục**. Chúng ta không thể tìm một quy luật như `sepal length = 5.1 cm` vì giá trị này quá cụ thể và sẽ hiếm khi lặp lại, dẫn đến support gần như bằng 0.

#### **2. Giải pháp: Rời rạc hóa (Discretization)**

Để giải quyết vấn đề này, chúng ta phải **biến đổi** các thuộc tính số liên tục thành các khoảng giá trị rời rạc (categorical). Đây là một bước **tiền xử lý** quan trọng.

Ví dụ, với `petal length (cm)`, thay vì dùng các giá trị số, chúng ta sẽ chia nó thành 3 nhóm: `short`, `medium`, `long`.

  * Nếu `petal length < 2.5` $\\rightarrow$ `petal_length_short`
  * Nếu `2.5 <= petal length < 5.0` $\\rightarrow$ `petal_length_medium`
  * Nếu `petal length >= 5.0` $\\rightarrow$ `petal_length_long`

Sau khi rời rạc hóa tất cả các thuộc tính, mỗi bông hoa sẽ trở thành một "giao dịch" chứa các "mục" là các đặc điểm đã được rời rạc hóa.

#### **3. Thực hành Python**

Thư viện `scikit-learn` không tích hợp sẵn Apriori, vì vậy chúng ta sẽ sử dụng một thư viện phổ biến khác là `apyori`.

Đầu tiên, hãy cài đặt thư viện này trong Colab:

```bash
pip install apyori
```

Bây giờ, hãy viết code để rời rạc hóa dữ liệu Iris và tìm các luật kết hợp.

```python
import pandas as pd
from sklearn.datasets import load_iris
from apyori import apriori

# Tải và tạo DataFrame
iris_dataset = load_iris()
df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
df['species'] = iris_dataset.target_names[iris_dataset.target]

# 1. Rời rạc hóa dữ liệu
# Sử dụng pd.cut để chia mỗi thuộc tính thành 3 khoảng (bins)
for column in df.columns[:-1]: # Lặp qua các cột thuộc tính, trừ cột 'species'
    # Tạo tên cho các khoảng, ví dụ: 'sepal_length_low', 'sepal_length_medium', ...
    labels = [f"{column.split(' ')[0]}_{label}" for label in ['low', 'medium', 'high']]
    df[column] = pd.cut(df[column], bins=3, labels=labels)

print("--- Dữ liệu sau khi rời rạc hóa ---")
print(df.head())


# 2. Chuẩn bị dữ liệu cho Apriori
# Thuật toán yêu cầu đầu vào là một list các list (mỗi list con là một giao dịch)
records = []
for i in range(len(df)):
    records.append([str(df.values[i,j]) for j in range(len(df.columns))])

# 3. Chạy thuật toán Apriori
# Đặt min_support = 0.3 (một itemset phải xuất hiện trong ít nhất 30% dữ liệu)
# Đặt min_confidence = 0.8 (độ tin cậy của luật phải ít nhất 80%)
association_rules = apriori(records, min_support=0.3, min_confidence=0.8, min_lift=1.0)
association_results = list(association_rules)

# 4. In ra các luật tìm được
print("\n--- Các luật kết hợp tìm được ---")
for item in association_results:
    # Lấy cặp item và các item liên quan
    pair = item[0]
    items = [x for x in pair]
    if len(items) > 1: # Chỉ hiển thị các luật có ý nghĩa
        print("Luật: " + str(item[2][0][0]) + " -> " + str(item[2][0][1]))
        print("Support: " + str(item[1]))
        print("Confidence: " + str(item[2][0][2]))
        print("=====================================")

```

**Phân tích kết quả:**

Kết quả chạy code sẽ cho ra các luật rất thú vị. Một trong những luật mạnh nhất mà bạn có thể tìm thấy là:

```
Luật: frozenset({'petal_low'}) -> frozenset({'setosa'})
Support: 0.3333333333333333
Confidence: 1.0
=====================================
Luật: frozenset({'petal_high'}) -> frozenset({'virginica'})
Support: 0.32666666666666666
Confidence: 0.9423076923076923
=====================================
```

**Diễn giải:**

  * **Luật 1:** `{petal_low}` $$\rightarrow$$ `{setosa}` có **Confidence = 1.0 (100%)**. Điều này có nghĩa là: "Nếu một bông hoa có chiều dài cánh hoa ở mức 'thấp' (low), thì **chắc chắn 100%** nó thuộc loài Setosa".
  * **Luật 2:** Tương tự, nếu chiều rộng cánh hoa ở mức 'cao' (high), nó 94.23% là loài Virginica.

Như vậy, dù không phải là một bộ dữ liệu giao dịch, nhưng thông qua bước **rời rạc hóa**, chúng ta đã có thể áp dụng thành công thuật toán khai phá mẫu để tìm ra những quy luật phân loại rất rõ ràng và dễ hiểu ngay trong dữ liệu Iris.
