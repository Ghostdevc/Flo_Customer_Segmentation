from customer_segmentation.utils import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

    
df = load_train()

check_df(df)

date_cols = [col for col in df.columns if 'date' in col]

df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x))


df['recency'] = (pd.to_datetime(datetime.now().date()) - df['last_order_date']).dt.days

df['frequency'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

df['monetary'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

df['interested_in_categories_12'].unique()
df['order_channel'].unique()
df['last_order_channel'].unique()

categories = ['AKTIFCOCUK', 'AKTIFSPOR', 'COCUK', 'ERKEK', 'KADIN']

for category in categories:
    df[category] = df['interested_in_categories_12'].apply(lambda x: 1 if category in x else 0)

df['order_channel'] = df['order_channel'].str.replace(' ', '')
df['last_order_channel'] = df['last_order_channel'].str.replace(' ', '')

columns_to_encode = [col for col in df.columns if df[col].dtype == 'O']
columns_to_encode.remove('interested_in_categories_12')
columns_to_encode.remove('master_id')
df = one_hot_encoder(df, columns_to_encode)

df = df.drop(columns=[col for col in df.columns if df[col].dtype == 'O'])
df = df.drop(columns=date_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    num_summary(df, col, True)

for col in cat_cols:
    cat_summary(df, col, True)

df[cat_cols] = df[cat_cols].applymap(lambda x: 1 if x in [True, 1] else 0)

df_cluster = df.copy()

sc = MinMaxScaler((0, 1))
df_cluster = sc.fit_transform(df_cluster)

df_cluster[0:5]

#KMEANS
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_cluster)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_cluster)
elbow.show()

elbow.elbow_value_


kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_cluster)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_

df["cluster_kmeans"] = clusters_kmeans

df.head()

df["cluster_kmeans"].unique()
df["cluster_kmeans"] = df["cluster_kmeans"] + 1

df.groupby("cluster_kmeans").agg(["count","mean","median"])

cat_summary(df, "cluster_kmeans", True)

#HIERARCHICAL CLUSTERING
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

hc_average = linkage(df_cluster, method = "average", metric = "euclidean")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()



plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=40,
           show_contracted=True,
           leaf_font_size=10)
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=1.30, color='y', linestyle='--')
plt.axhline(y=1.28, color='g', linestyle='--')
plt.axhline(y=1.27, color='r', linestyle='--')
plt.axhline(y=1.25, color='b', linestyle='--')
plt.show()
#maybe y = 1.5


from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

cluster_range = range(2, 10)
silhouette_scores = []

for k in cluster_range:
    clusters = fcluster(hc_average, k, criterion="maxclust")
    silhouette_scores.append(silhouette_score(df_cluster, clusters))

plt.plot(cluster_range, silhouette_scores, marker="o")
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette Skoru")
plt.title("En İyi Küme Sayısını Belirleme")
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc_cluster = AgglomerativeClustering(n_clusters=10, linkage='average', metric = 'euclidean')
hc_cluster.fit_predict(df_cluster)

clusters_hc = hc_cluster.labels_
df["cluster_hc"] = clusters_hc
df["cluster_hc"] = df["cluster_hc"] + 1





x_train, x_test, y_train, y_test = train_test_split(df, df["cluster"], test_size=0.30, random_state=42)

base_models_cl(x_train, y_train)

hyperparameter_optimization_cl(x_train, y_train)