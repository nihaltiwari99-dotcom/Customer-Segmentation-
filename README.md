# 🛍️ Customer Segmentation using RFM Analysis & K-Means Clustering

## 📌 Project Overview

This project performs **Customer Segmentation** on the Online Retail dataset using **RFM Analysis** and **K-Means Clustering**.

The objective is to analyze customer purchasing behavior and divide customers into meaningful groups based on:

* **Recency** – How recently a customer purchased
* **Frequency** – How frequently a customer makes purchases
* **Monetary** – How much money a customer spends

After calculating the RFM metrics, the data is standardized and K-Means clustering is applied to identify different customer segments.

These segments can help businesses understand customer behavior and design targeted marketing strategies.

---

## 🎯 Business Objective

Businesses often have thousands of customers, but not every customer behaves in the same way.

For example:

* Some customers purchase frequently and spend a lot.
* Some customers are loyal but spend moderately.
* Some customers have not purchased for a long time.
* Some customers are new or have low purchasing activity.

Instead of treating every customer equally, this project uses machine learning to identify customer groups automatically.

The resulting segments can be used for:

* Customer retention
* Personalized marketing
* Loyalty programs
* Targeted promotions
* Customer reactivation campaigns
* Identifying high-value customers

---

# 📊 Dataset

The project uses the **Online Retail Dataset**, which contains transactional data from a UK-based online retailer.

### Important Features

| Column        | Description                  |
| ------------- | ---------------------------- |
| `InvoiceNo`   | Invoice/transaction number   |
| `StockCode`   | Product code                 |
| `Description` | Product description          |
| `Quantity`    | Number of products purchased |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice`   | Price per unit               |
| `CustomerID`  | Unique customer identifier   |
| `Country`     | Customer's country           |

---

# 🔄 Project Workflow

```text
Online Retail Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Exploratory / Pre-Cluster Analysis
        ↓
RFM Analysis
        ↓
Feature Scaling
        ↓
Elbow Method
        ↓
K-Means Clustering
        ↓
Customer Segmentation
        ↓
Cluster Analysis & Visualization
        ↓
New Customer Prediction
```

---

# 🧹 1. Data Cleaning

The dataset is loaded using Pandas.

```python
df = pd.read_excel(path)
```

Customer IDs with missing values are removed:

```python
df = df.dropna(subset=["CustomerID"])
```

This is important because customer-level RFM analysis requires a valid `CustomerID`.

---

# 🧮 2. Feature Engineering

A new feature called `Total Amount` is created:

```python
df["Total Amount"] = df["Quantity"] * df["UnitPrice"]
```

This represents the total value of each transaction line.

### Formula

```text
Total Amount = Quantity × Unit Price
```

---

# 📈 3. Pre-Cluster Analysis

Before performing customer segmentation, the project analyzes the dataset from a business perspective.

### Analysis performed

#### Total Bills Generated

Counts the number of unique invoices.

#### Cancelled Bills

Identifies invoices beginning with `"C"`.

#### Countries with Highest Sales

Calculates total sales by country.

#### Countries with Highest Number of Orders

Counts unique invoices by country.

#### Most Demanded Products

Identifies products that appear most frequently in transactions.

#### Highest Value Products

Identifies products with the highest average unit price while excluding non-product transactions such as postage, discounts, bank charges, etc.

---

# 📊 4. Data Visualization

Plotly is used to create interactive visualizations.

The project includes visualizations for:

* Top 10 countries by sales
* Top 10 countries by number of orders
* Top 20 most demanded products
* Top 20 highest-value products
* Customer segments
* Segment distribution

Example:

```python
fig = px.bar(
    country_sales,
    x="Country",
    y="Total Amount",
    title="Top 10 Countries by Sales",
    color="Country",
    text_auto=".2s"
)
```

---

# 👥 5. RFM Analysis

RFM analysis is the core of the customer segmentation process.

## R — Recency

Recency measures **how recently a customer made a purchase**.

The last transaction date for every customer is calculated:

```python
rfm = (
    df[df["Quantity"] > 0]
    .groupby("CustomerID")["InvoiceDate"]
    .max()
    .reset_index()
)
```

The analysis date is defined as one day after the last transaction:

```python
analysis_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
```

Recency is calculated as:

```python
rfm["Recency"] = (
    analysis_date - rfm["InvoiceDate"]
).dt.days
```

### Interpretation

```text
Low Recency  → Customer purchased recently
High Recency → Customer has not purchased for a long time
```

---

# 🔁 F — Frequency

Frequency measures **how many unique orders a customer has made**.

```python
frequency = (
    df[df["Quantity"] > 0]
    .groupby("CustomerID")["InvoiceNo"]
    .nunique()
    .reset_index()
)
```

The column is renamed:

```python
frequency.rename(
    columns={"InvoiceNo": "Frequency"},
    inplace=True
)
```

`nunique()` is used because multiple rows can belong to the same invoice.

---

# 💰 M — Monetary

Monetary measures **the total amount spent by a customer**.

```python
monetary = (
    df[df["Quantity"] > 0]
    .groupby("CustomerID")["Total Amount"]
    .sum()
    .reset_index()
)
```

The resulting column is renamed to `Monetary`.

### Formula

```text
Monetary = Sum of Total Amount for each customer
```

---

# 📋 Final RFM Dataset

After combining all three metrics, the dataset looks like:

| CustomerID | Recency | Frequency | Monetary |
| ---------- | ------: | --------: | -------: |
| 12346      |     326 |         1 | 77183.60 |
| 12347      |       2 |         7 |  4310.00 |
| 12348      |      75 |         4 |  1797.24 |

Each row represents one customer.

---

# ⚖️ 6. Feature Scaling

RFM features have very different numerical ranges.

For example:

```text
Recency    → 1–300 days
Frequency  → 1–100 orders
Monetary   → 10–100000+
```

If these features are directly passed to K-Means, the larger numerical values can have a disproportionate influence.

Therefore, `StandardScaler` is used:

```python
from sklearn.preprocessing import StandardScaler

rfm_features = rfm[
    ["Recency", "Frequency", "Monetary"]
]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_features)
```

Standardization transforms the features approximately to:

```text
Mean = 0
Standard Deviation = 1
```

---

# 📉 7. Elbow Method

K-Means requires us to specify the number of clusters (`K`).

The **Elbow Method** is used to identify a suitable value of K.

```python
wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)

    wcss.append(kmeans.inertia_)
```

`kmeans.inertia_` represents the **Within-Cluster Sum of Squares (WCSS)**.

### WCSS

WCSS measures how close the data points are to the centroid of their assigned cluster.

```text
Lower WCSS
     ↓
Points are closer to their centroids
     ↓
More compact clusters
```

The WCSS values are plotted against the number of clusters.

The point where the decrease in WCSS starts becoming significantly smaller is known as the **elbow**.

In this project, **K = 4** is selected.

---

# 🤖 8. K-Means Clustering

The final K-Means model uses four clusters:

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
```

The clusters are generated using:

```python
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
```

Each customer is assigned a cluster number.

For example:

```text
Customer A → Cluster 2
Customer B → Cluster 0
Customer C → Cluster 3
```

---

# 🔍 9. Customer Segment Classification

The clusters are analyzed using their average RFM values.

```python
cluster_summary = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).round(2)
```

Based on the RFM characteristics, the clusters are given meaningful business names.

### 👑 VIP / Champions

Characteristics:

* Very recent purchases
* Very high purchase frequency
* Very high spending

These are the most valuable customers.

### ❤️ Loyal Customers

Characteristics:

* Recent purchases
* Good purchase frequency
* High spending

These customers have demonstrated strong loyalty.

### 🌱 Potential Customers

Characteristics:

* Moderate recency
* Low frequency
* Low spending

These customers could potentially become loyal customers through targeted marketing.

### ⚠️ Lost Customers

Characteristics:

* High recency
* Low frequency
* Low spending

These customers have not purchased recently and may require reactivation campaigns.

---

# 🏷️ Segment Mapping

The clusters are converted into meaningful labels:

```python
cluster_names = {
    0: "Loyal Customers",
    1: "Lost Customers",
    2: "VIP",
    3: "Potential Customers"
}

rfm["Segment"] = rfm["Cluster"].map(cluster_names)
```

This makes the results easier for business users to understand.

---

# 📊 10. Customer Segment Visualization

A scatter plot is created to visualize customer segments:

```python
fig = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color="Segment",
    hover_data=["CustomerID", "Recency"],
    title="Customer Segmentation using RFM + K-Means"
)

fig.show()
```

The visualization helps identify how customers differ in terms of:

* Frequency
* Monetary value
* Recency

---

# 🥧 11. Segment Distribution

A pie chart is used to understand how customers are distributed across the segments.

```text
Customers
    ↓
VIP
Loyal
Potential
Lost
```

This can help businesses understand the size of each customer group.

---

# 🔮 12. New Customer Segment Prediction

The project can also predict the segment of a new customer.

The saved scaler and K-Means model are loaded:

```python
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)
```

The user provides:

```text
Recency
Frequency
Monetary
```

For example:

```text
Recency: 5
Frequency: 15
Monetary: 5000
```

The values are converted into a DataFrame:

```python
new_customer = pd.DataFrame(
    [[R, F, M]],
    columns=["Recency", "Frequency", "Monetary"]
)
```

The new customer's features are then scaled using the **same scaler used during training**:

```python
new_customer_scaled = scaler.transform(new_customer)
```

Finally, K-Means predicts the customer's cluster:

```python
segment = kmeans.predict(new_customer_scaled)
```

The cluster number is converted into a business segment.

---

# 💾 13. Model Persistence

The trained models are saved using Python's `pickle` module.

### Scaler

```python
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

### K-Means Model

```python
with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)
```

This allows the trained preprocessing and clustering model to be reused without training again.

---

# 🛠️ Technologies Used

| Technology             | Purpose                   |
| ---------------------- | ------------------------- |
| Python                 | Programming language      |
| Pandas                 | Data manipulation         |
| NumPy                  | Numerical operations      |
| Scikit-learn           | Scaling and K-Means       |
| Plotly                 | Interactive visualization |
| OpenPyXL               | Excel file processing     |
| Pickle                 | Model persistence         |
| Jupyter / Google Colab | Development environment   |

---

# 📁 Project Structure

```text
Customer-Segmentation/
│
├── Customer Segmentation.ipynb
├── Online Retail.xlsx
├── scaler.pkl
├── kmeans.pkl
├── README.md
└── requirements.txt
```

> **Note:** The dataset and generated model files can be excluded from GitHub if they are large. They can instead be downloaded or generated by running the notebook.

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-segmentation.git
```

### 2. Navigate to the project

```bash
cd customer-segmentation
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn plotly openpyxl
```

Or:

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place:

```text
Online Retail.xlsx
```

in the appropriate project directory.

### 5. Run the notebook

Open:

```text
Customer Segmentation.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

---

# 📌 Key Insights

The project demonstrates how transactional customer data can be transformed into actionable customer segments.

RFM analysis provides three important behavioral dimensions:

```text
Recency
   +
Frequency
   +
Monetary
   ↓
Customer Behavior
   ↓
K-Means Clustering
   ↓
Customer Segments
```

These segments can support business decisions such as:

| Segment             | Possible Business Strategy         |
| ------------------- | ---------------------------------- |
| VIP                 | Premium offers and loyalty rewards |
| Loyal Customers     | Loyalty programs and cross-selling |
| Potential Customers | Personalized promotions            |
| Lost Customers      | Win-back campaigns and discounts   |

---

# 🔮 Future Improvements

Possible improvements include:

* Deploy the segmentation model using **Streamlit**
* Add an interactive customer dashboard
* Use **Silhouette Score** to evaluate clustering quality
* Experiment with different clustering algorithms such as DBSCAN and hierarchical clustering
* Apply log transformation to highly skewed Monetary values
* Add automated customer recommendations
* Build an API using FastAPI
* Create automated marketing recommendations for each segment
* Add customer lifetime value (CLV) analysis

---

# 👨‍💻 Author

**Nihal Tiwari**

This project demonstrates practical skills in:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* RFM Analysis
* Feature Scaling
* Unsupervised Machine Learning
* K-Means Clustering
* Data Visualization
* Model Persistence
* Customer Analytics

---

# ⭐ Conclusion

This project demonstrates an end-to-end **customer analytics and unsupervised machine learning workflow**.

Starting from raw transactional data, the project performs data cleaning and exploratory analysis, calculates RFM metrics, standardizes customer features, determines an appropriate number of clusters using the Elbow Method, and applies K-Means clustering to identify meaningful customer segments.

The trained clustering model can then be reused to classify new customers based on their **Recency, Frequency, and Monetary** behavior.

This approach transforms raw transaction data into actionable customer insights that can support **customer retention, targeted marketing, loyalty programs, and business decision-making**.
