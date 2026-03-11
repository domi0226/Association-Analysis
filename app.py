import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="關聯規則分析工具", layout="wide")
st.title("萬用關聯規則分析系統")

# --- 1. 檔案上傳 ---
uploaded_file = st.file_uploader("請上傳 Excel 檔案", type=["xlsx", "xls"])

if uploaded_file:
    # 讀取資料
    df = pd.read_excel(uploaded_file)
    st.write("### 原始資料預覽 (前 5 筆)")
    st.dataframe(df.head())

    # 讓使用者選取包含交易內容的欄位
    all_columns = df.columns.tolist()
    target_col = st.selectbox("請選擇包含『交易項目』的欄位 (例如：game)", all_columns)

    # --- 2. 參數設定 (側邊欄) ---
    st.sidebar.header("演算法參數")
    min_support = st.sidebar.slider("最小支持度 (Support)", 0.01, 0.5, 0.05)
    min_confidence = st.sidebar.slider("最小信心度 (Confidence)", 0.1, 1.0, 0.5)
    min_lift = st.sidebar.slider("最小提升度 (Lift)", 1.0, 5.0, 1.0)

    if st.button("開始執行分析"):
        with st.spinner('計算中...'):
            try:
                # 資料清理：將內容轉為 list 並去除空格
                transactions = df[target_col].apply(lambda x: [i.strip() for i in str(x).split(",")])
                
                # One Hot Encoding
                te = TransactionEncoder()
                te_array = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_array, columns=te.columns_)

                # Apriori 計算
                frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

                if not frequent_itemsets.empty:
                    # 關聯規則
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
                    rules = rules[rules["confidence"] >= min_confidence]

                    if not rules.empty:
                        st.success(f"✅ 分析完成！找到 {len(rules)} 條規則")
                        
                        # 格式化顯示
                        display_df = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
                        display_df["antecedents"] = display_df["antecedents"].apply(lambda x: ', '.join(list(x)))
                        display_df["consequents"] = display_df["consequents"].apply(lambda x: ', '.join(list(x)))
                        
                        st.dataframe(display_df.sort_values("lift", ascending=False), use_container_width=True)
                    else:
                        st.warning("⚠️ 找不到符合過濾條件 (Confidence/Lift) 的規則。")
                else:
                    st.error("❌ 支持度 (Support) 設定太高，找不到任何頻繁項目。")
            
            except Exception as e:
                st.error(f"執行發生錯誤: {e}")
else:
    st.info("💡 請先從上方上傳 Excel 檔案開始分析。")