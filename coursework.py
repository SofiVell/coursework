import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium 

df = pd.read_csv("Electric_Vehicle.csv")

menu = st.sidebar.radio("Виберіть вкладку", [
    "Дані та візуалізації", 
    "Індивідуальний підбір електромобіля", 
    "Приклади підбору електромобіля для різної цільової аудиторії",
    "База даних"
])

if menu == "Дані та візуалізації":
    st.title("Візуалізація характеристик електромобілів")

    # 1.Кількість моделей за країною
    country_counts = df['Country'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=country_counts.index, y=country_counts.values)
    plt.xticks(rotation=45)
    plt.xlabel("Країна")
    plt.ylabel("Кількість моделей")
    plt.title("Кількість моделей за країною")
    st.pyplot(plt)

    # 2. Кількість моделей за роками
    year_min = df["Year"].min()
    bins = range(year_min, 2026)
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Year"], bins=bins, kde=False)
    plt.xlabel("Рік")
    plt.ylabel("Кількість моделей")
    plt.title("Кількість моделей за роками")
    plt.xticks(bins, rotation=45)
    st.pyplot(plt)

    # 3. Середня ціна електромобілів в Німеччині за країнами
    avg_price_per_country = df.groupby('Country')['Price in Germany'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_price_per_country.index, y=avg_price_per_country.values, palette='viridis')
    plt.xticks(rotation=45)
    plt.xlabel("Країна")
    plt.ylabel("Середня ціна в Німеччині (€)")
    plt.title("Середня ціна електромобілів в Німеччині за країнами")
    plt.tight_layout()
    st.pyplot(plt)

    # 4. Класи електромобілів
    segment_counts = df["Segment"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(segment_counts, labels=segment_counts.index, autopct="%1.1f%%")
    plt.title("Класи електромобілів")
    st.pyplot(plt)

    # 5-9. Топ-10
    metrics = [
        ("Топ-10 за ємністю акумулятора", "Battery (kilowatt-hours)", False),
        ("Топ-10 за енергоефективністю", "Efficiency (watt-hours per kilometer)", True),
        ("Топ-10 за швидкістю швидкої зарядки", "Fast_charge", True),
        ("Топ-10 за запасом ходу", "Range", False),
        ("Топ-10 за максимальною швидкістю", "Top_speed", False)
    ]

    for title, column, ascending in metrics:
        st.subheader(title)
        top_df = (
            df[["Model", column]]
            .dropna()
            .sort_values(by=column, ascending=ascending)
            .head(10)
            .reset_index(drop=True)
        )
        top_df.index += 1
        st.write(top_df)
        st.markdown("<hr>", unsafe_allow_html=True) 

    # 10. Розміщення виробників на карті світу
    st.subheader("Розміщення виробників на карті світу")
    country_coords = {
        "USA": [37.0902, -95.7129],
        "Sweden": [60.1282, 18.6435],
        "China": [35.8617, 104.1954],
        "France": [46.6034, 1.8883],
        "Germany": [51.1657, 10.4515],
        "South Korea": [35.9078, 127.7669],
        "Great Britain": [51.5074, -0.1278],
        "Italy": [41.9028, 12.4964],
        "Czech Republic": [49.8175, 15.4730],
        "Spain": [40.4637, -3.7492],
        "Vietnam": [14.0583, 108.2772],
        "Romania": [45.9432, 24.9668],
        "Japan": [36.2048, 138.2529]
    }

    ev_map = folium.Map(location=[20, 0], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(ev_map)

    for country, coords in country_coords.items():
        country_data = df[df['Country'] == country]
        if not country_data.empty:
            unique_producers = sorted(country_data['Producer'].unique())
            producer_list = "<br>".join(unique_producers)
            count = len(unique_producers)

            popup_content = f"""
            <b>{country}</b><br>
            <b>Кількість виробників:</b> {count}<br><br>
            <b>Виробники:</b><br>{producer_list}
            """

            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='blue', icon='car', prefix='fa')
            ).add_to(marker_cluster)

    st_folium(ev_map, width=1200, height=600)

elif menu == "Індивідуальний підбір електромобіля":
    st.title("Індивідуальний підбір електромобіля")

    st.markdown("#### Оберіть критерії, які для вас важливі (від 1 до 3):")

    criteria_options = {
        "Запас ходу": "Range",
        "Швидкість зарядки": "Fast_charge",
        "Ціна": "Price in Germany",
        "Енергоефективність": "Efficiency (watt-hours per kilometer)",
        "Максимальна швидкість": "Top_speed",
        "Ємність батареї": "Battery (kilowatt-hours)"
    }

    selected_labels = st.multiselect("Критерії:", list(criteria_options.keys()), max_selections=3)

    st.markdown("#### Оберіть бажані сегменти автомобіля:")
    segment_options = [
        "Mini", "Compact", "Medium", "Large", "Executive", "Luxury", "Passenger van"
    ]
    selected_segments = st.multiselect(
        "Сегменти:", segment_options, default=segment_options  
    )

    filter_ranges = {}

    if selected_labels:
        st.markdown("#### Налаштуйте діапазони для обраних критеріїв:")
        for label in selected_labels:
            col = criteria_options[label]
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            selected_range = st.slider(
                f"{label} ({col})", min_val, max_val, (min_val, max_val)
            )
            filter_ranges[col] = selected_range

    if not selected_segments:
        st.warning("Будь ласка, оберіть хоча б один сегмент.")
    elif 1 <= len(selected_labels) <= 3:
        selected_cols = [criteria_options[label] for label in selected_labels]

        if len(selected_cols) == 1:
            weights = [1.0]
        elif len(selected_cols) == 2:
            weights = [0.5, 0.5]
        else:
            weights = [0.5, 0.3, 0.2]

        ascending_map = {
            "Price in Germany": True,
            "Efficiency (watt-hours per kilometer)": True,
            "Range": False,
            "Fast_charge": False,
            "Top_speed": False,
            "Battery (kilowatt-hours)": False
        }

        df_filtered = df[df["Segment"].isin(selected_segments)].copy()

        for col, (min_val, max_val) in filter_ranges.items():
            df_filtered = df_filtered[df_filtered[col].between(min_val, max_val)]

        df_filtered = df_filtered.dropna(subset=selected_cols)

        df_filtered["score"] = 0
        for col, weight in zip(selected_cols, weights):
            df_filtered["score"] += df_filtered[col].rank(ascending=ascending_map[col]) * weight

        top10 = df_filtered.sort_values("score").head(10).reset_index(drop=True)

        st.markdown("### Топ-10 рекомендованих моделей:")
        output_cols = ["Model"] + selected_cols + ["Segment"]
        st.dataframe(top10[output_cols])
    else:
        st.info("Будь ласка, оберіть від 1 до 3 критеріїв.")

elif menu == "Приклади підбору електромобіля для різної цільової аудиторії":
    st.title("Приклади підбору електромобіля для різної цільової аудиторії")

    st.markdown("""
    ### Цільові аудиторії:

    - **Далекобійники** — водії, які регулярно долають великі відстані. Вони першочергово звертають увагу на запас ходу та швидкість зарядки. Їхня головна потреба — подорожі без частих зупинок, а також комфорт у тривалих поїздках.
    
    - **Міські жителі** використовують електромобілі переважно в умовах щільного трафіку та обмеженого простору. Для них важлива компактність, легкість у паркуванні й економічність щоденного користування.
    
    - **Еко-ентузіасти** — це споживачі, які надають перевагу енергоефективним моделям із мінімальним впливом на довкілля. Їхній вибір базується на низькому рівні енергоспоживання та екологічній чистоті.
    
    - **Швидкі та потужні** — користувачі, які шукають драйв, емоції та технічну досконалість. Вони цінують високу максимальну швидкість і хорошу динаміку.
    
    - **Бюджетні покупці** зосереджені передусім на вартості автомобіля. Для них головне — доступна ціна.
    """)


    audience = st.selectbox("Оберіть категорію", [
        "Далекобійники",
        "Міські жителі",
        "Еко-ентузіасти",
        "Швидкі та потужні",
        "Бюджетні покупці"
    ])

    df_filtered = df.copy()

    if audience == "Далекобійники":
        st.subheader("Топ-10 електромобілів для далеких поїздок")

        long_trip_df = df_filtered.dropna(subset=["Range", "Fast_charge", "Price in Germany"]).copy()

        long_trip_df["score"] = (
            long_trip_df["Range"].rank(ascending=False) * 0.5 +
            long_trip_df["Fast_charge"].rank(ascending=False) * 0.3 +
            long_trip_df["Price in Germany"].rank(ascending=True) * 0.2
        )

        top_long_trip = long_trip_df.sort_values("score").head(10)[
            ["Model", "Range", "Fast_charge", "Price in Germany"]
        ]

        st.dataframe(top_long_trip.reset_index(drop=True))
    
    elif audience == "Міські жителі":
        st.subheader("Топ-10 компактних моделей")
        city_df = df_filtered[
            df_filtered["Segment"].isin(["Mini", "Compact", "Medium"])
        ].dropna(subset=["Price in Germany", "Range"])

        top_city = city_df.sort_values("Price in Germany").head(10)[["Model", "Price in Germany", "Segment"]]
        st.dataframe(top_city.reset_index(drop=True))

    elif audience == "Еко-ентузіасти":
        st.subheader("Топ-10 за енергоефективністю")
        eco_df = df_filtered.dropna(subset=["Efficiency (watt-hours per kilometer)"])
        top_eco = eco_df.sort_values("Efficiency (watt-hours per kilometer)").head(10)[["Model","Price in Germany", "Efficiency (watt-hours per kilometer)"]]
        st.dataframe(top_eco.reset_index(drop=True))

    elif audience == "Швидкі та потужні":
        st.subheader("Топ-10 за максимальною швидкістю")
        fast_df = df_filtered.dropna(subset=["Top_speed"])
        top_fast = fast_df.sort_values("Top_speed", ascending=False).head(10)[["Model","Price in Germany", "Top_speed"]]
        st.dataframe(top_fast.reset_index(drop=True))

    elif audience == "Бюджетні покупці":
        st.subheader("Топ-10 найдешевших моделей")
        cheap_df = df_filtered.dropna(subset=["Price in Germany"])
        top_cheap = cheap_df.sort_values("Price in Germany").head(10)[["Model", "Price in Germany", "Segment"]]
        st.dataframe(top_cheap.reset_index(drop=True))

elif menu == "База даних":
    st.title("База даних")

    search_query = st.text_input("Пошук за назвою моделі:", "")

    if search_query:
        filtered_df = df[df["Model"].str.contains(search_query, case=False, na=False)]
        if not filtered_df.empty:
            st.success(f"Знайдено {len(filtered_df)} моделей:")
            filtered_df["Year"] = filtered_df["Year"].astype(str)
            st.dataframe(filtered_df.reset_index(drop=True))
        else:
            st.warning("Модель не знайдена.")
    else:
        df["Year"] = df["Year"].astype(str)
        st.dataframe(df)

