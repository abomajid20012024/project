import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# تحميل البيانات
search_df = pd.read_excel("marketplace_data.xlsx", sheet_name="Search History")
browse_df = pd.read_excel("marketplace_data.xlsx", sheet_name="Browse History")
purchase_df = pd.read_excel("marketplace_data.xlsx", sheet_name="Purchase History")
rating_df = pd.read_excel("marketplace_data.xlsx", sheet_name="Product Ratings")
customer_df = pd.read_excel("marketplace_data.xlsx", sheet_name="Customer Data")
products = pd.read_excel("marketplace_data.xlsx", sheet_name="products")

# تنظيف البيانات
products.drop_duplicates(inplace=True)
products['description'] = products['description'].fillna("")
products = products[products['description'] != ""]

customer_df.drop_duplicates(inplace=True)
customer_df.dropna(subset=['name', 'location', 'age_group'], inplace=True)

search_df.drop_duplicates(inplace=True)
search_df.dropna(subset=['user_id', 'keyword'], inplace=True)

browse_df.drop_duplicates(inplace=True)
browse_df.dropna(subset=['user_id', 'product_id'], inplace=True)

purchase_df.drop_duplicates(inplace=True)
purchase_df.dropna(subset=['user_id', 'product_id'], inplace=True)

rating_df.drop_duplicates(inplace=True)
rating_df.dropna(subset=['user_id', 'product_id', 'rating'], inplace=True)

# تحليل النصوص للمنتجات باستخدام TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(products['description'])

product_similarity_model = NearestNeighbors(metric='euclidean')
product_similarity_model.fit(tfidf_matrix)

# إنشاء مصفوفة المستخدم-المنتج
user_product_matrix = purchase_df.pivot_table(
    index='user_id',
    columns='product_id',
    values='purchase_id',
    aggfunc='count'
).fillna(0)
user_product_matrix[user_product_matrix > 0] = 1

user_similarity_model = NearestNeighbors(metric='euclidean')
user_similarity_model.fit(user_product_matrix)

# التوصية بناءً على التعاون
def recommend_collaborative(user_id, user_product_matrix, user_similarity_model, k=5):
    if user_id not in user_product_matrix.index:
        return pd.DataFrame()  # لا يوجد بيانات كافية للمستخدم

    distances, indices = user_similarity_model.kneighbors(
        [user_product_matrix.loc[user_id].values],
        n_neighbors=k + 1
    )
    similar_users = indices.flatten()[1:]
    recommended_products = user_product_matrix.iloc[similar_users].sum().sort_values(ascending=False)

    # استبعاد المنتجات التي اشتراها المستخدم بالفعل
    user_products = user_product_matrix.loc[user_id]
    recommended_products = recommended_products[user_products == 0]

    top_products_ids = recommended_products.head(k).index.tolist()

    # إرجاع جدول يحتوي معلومات المنتجات
    return products[products['product_id'].isin(top_products_ids)][['name', 'price', 'description']]

# التوصية بناءً على المحتوى
def recommend_content_based(product_id, product_similarity_model, tfidf_matrix, products, k=5):
    product_index = products[products['product_id'] == product_id].index[0]
    product_vector = tfidf_matrix[product_index].toarray()
    distances, indices = product_similarity_model.kneighbors(product_vector, n_neighbors=k + 1)
    similar_products = indices.flatten()[1:]
    distances = distances.flatten()[1:]

    recommendations_df = products.iloc[similar_products][['name', 'price', 'description']].copy()
    recommendations_df['score'] = 1 - distances / distances.max()  # عكس المسافة لمقياس تشابه (0 إلى 1)

    return recommendations_df

# دالة لتلوين الصفوف حسب قوة التوصية
def highlight_recommendation(row):
    if row['score'] > 0.5:
        styles = [''] * len(row)
        name_col_idx = row.index.get_loc('name')
        price_col_idx = row.index.get_loc('price')
        score_col_idx = row.index.get_loc('score')
        styles[name_col_idx] = 'background-color: #d0f0c0'  # أخضر فاتح
        styles[price_col_idx] = 'background-color: #d0f0c0'
        styles[score_col_idx] = 'background-color: #d0f0c0'
        return styles
    else:
        return [''] * len(row)

# ========== واجهة المستخدم ==========
st.set_page_config(page_title="نظام التوصية", layout="centered")
st.title("📦 نظام التوصية بالمنتجات")

# اختيار المستخدم
user_options = customer_df[['user_id', 'name']].drop_duplicates()
user_map = {f"{row['name']} (ID: {row['user_id']})": row['user_id'] for _, row in user_options.iterrows()}
selected_user_label = st.selectbox("👤 اختر مستخدمًا:", list(user_map.keys()))
selected_user_id = user_map[selected_user_label]

# اختيار نوع التوصية
rec_type = st.radio("📊 نوع التوصية:", ["بناءً على التعاون بين المستخدمين", "بناءً على المحتوى (منتج مشابه)"])

# اختيار قيمة k
k = st.slider("🔢 اختر عدد التوصيات (k):", min_value=1, max_value=20, value=5)

# عرض التوصيات
if rec_type == "بناءً على التعاون بين المستخدمين":
    st.markdown("""
    🧠 **طريقة التوصية:**
    يتم تحديد المستخدمين المتشابهين بناءً على سجل الشراء، 
    ثم يتم اقتراح المنتجات التي اشتراها هؤلاء المستخدمون ولم يشترها المستخدم الحالي.
    """)
    recommendations_df = recommend_collaborative(selected_user_id, user_product_matrix, user_similarity_model, k=k)
    if recommendations_df.empty:
        st.warning("❗️ لا توجد توصيات كافية لهذا المستخدم.")
    else:
        st.subheader("📌 المنتجات المقترحة:")
        st.dataframe(recommendations_df.rename(columns={
            "name": "اسم المنتج", "price": "السعر", "description": "الوصف"
        }), use_container_width=True)

else:
    user_products = purchase_df[purchase_df['user_id'] == selected_user_id]['product_id'].unique()
    if len(user_products) == 0:
        st.warning("❗️لا يوجد منتجات قام هذا المستخدم بشرائها ليتم اعتمادها للتوصية بالمحتوى.")
    else:
        product_names = {products.loc[products['product_id'] == pid, 'name'].values[0]: pid for pid in user_products}
        selected_product_name = st.selectbox("📘 اختر منتجًا تم شراؤه ليتم التوصية بمنتجات مشابهة:", list(product_names.keys()))
        selected_product_id = product_names[selected_product_name]

        st.markdown("""
        🧠 **طريقة التوصية:**
        يتم تحليل وصف المنتج المحدد باستخدام TF-IDF، ثم اقتراح منتجات مشابهة له في الوصف.
        """)
        recommendations_df = recommend_content_based(selected_product_id, product_similarity_model, tfidf_matrix, products, k=k)
        st.subheader("📌 المنتجات المشابهة:")

        # عرض بدون عمود الوصف (لتحسين التلوين)
        st.dataframe(
            recommendations_df.drop(columns=['description'])
            .style.apply(highlight_recommendation, axis=1),
            use_container_width=True
        )
