from sklearn.cluster import KMeans
from customer_segmentation.utils import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer

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

df["cluster"] = clusters_kmeans

df.head()

df["cluster"].unique()
df["cluster"] = df["cluster"] + 1

df.groupby("cluster").agg(["count","mean","median"])

cat_summary(df, "cluster", True)

x_train, x_test, y_train, y_test = train_test_split(df, df["cluster"], test_size=0.30, random_state=42)

base_models_cl(x_train, y_train)
