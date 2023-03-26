
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)

# Takip Edeceğimiz Adımlar

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)  (faturalar-ürünler matrisi oluşturcaz)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

# 5 adımda projemizi gerçekleştiriyor olacağız

# 1. BASAMAK
# 1. Veri Ön İşleme (Data Preprocessing)

# !pip install mlxtend
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)  # bütün satırları göster
pd.set_option('display.width', 500)  # sütunlar max 500 tane gösterilsin (alttakiyle aynı işlem)
pd.set_option('display.expand_frame_repr', False)  # çıktının tek bir satırda olmasını sağlar.
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II  data seti buradan indirebilirsiniz

df_ = pd.read_excel(r"C:\Users\GunalHincal\Desktop\datasets for github\recommender\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.tail()
# df olarak dataframe i kopyaladık ve df_ olarak yedek tutuyoruz, çünkü veri uzun olduğu için
# sistemin okuması da süre alıyor bu nedenle df_ adında da bir kopya oluşturduk gerektiğinde kullanmak için

# Değişkenler
# InvoiceNo           : Fatura Numarası - Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder
# StockCode           : Ürün Kodu - Her bir ürün için eşsiz numara
# Description         : Ürün İsmi
# Quantity            : Ürün Adedi - Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir
# InvoiceDate         : Fatura Tarihi
# Price               : Fatura Fiyatı (Sterlin)
# Customer ID         : Eşsiz Müşteri Numarası
# Country             : Ülke İsmi

# Dataseti ARL in beklediği formata getirmemiz gerek, ilgili veri yapısını ortaya çıkarmamız gerek

df.describe().T
df.isnull().sum()  # zengin bir data set olduğu için problem çıkaracak veriler varsa bakmalıyız
df.shape  # (541910, 8) gözlem ve sütun sayısı

# eksik değerleri, invoice'da C değerleri olanları yani iade ürünleri datasetten çıkaracağız,
# quantity ve priceda eksik olanları da çıkaracağız ve aykırı olan değerleri baskılayacağız

# fonksiyon yazarak bu işlemleri yapalım. işlemleri 2 ye ayıralım
# 1.si: iade faturaları çıkar, NA leri çıkar, quantity ve price'ı 0 dan büyük yap işlemleri olsun
# 2.si ise quantity ve price fonksiyonları quantity ve price değişkenleri için 0'dan büyük olanları al olsun
# quantity ve price değişkenleri için aykırı değerleri bul onları eşik değerlere baskıla olsun

# eksik değerleri düşür, invoice' da C yazanları çıkart, quantity ve price'da 0' dan büyük olanları al
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)

df.isnull().sum()  # eksik değerler gitti
df.describe().T    # eksili değerler gitti


# aykırı değer kırpma
# eşik değerleri hesaplayacak fonksiyonumuzu getirelim
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# şimdi yapılması gereken işlem, bu hesapladığımız eşik değere göre baskılama yapmak
# fonksiyonumuzu getirelim
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# bu iki ana işlemi de bir arada yapan bir fonksiyon yazıyoruz
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


# dataframe i tekrar getirelim ve son yazdığımız fonksiyonla bütün işlemleri yapalım
df = df_.copy()
df = retail_data_prep(df)
df.isnull().sum()
df.describe().T
df.shape


# 2. BASAMAK
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)

df.head()
# elimizdeki veri şuanda fatura id lerine göre çoklama, ürünlere göre tekil bir formda.
# Bizim istediğimiz daha ölçülebilir üzerinde işlemler yapabileceğimiz matris formda bir veri istiyoruz
# yani, satırlarda invoice ler sütunlarda product lar olsun istiyoruz. Faturada bir ürünün olup olmaması
# 1 ve 0 larla ifade edilsin istiyoruz

# şimdi birliktelik kurallarını fransa ülkesine göre çıkaracağız

df_fr = df[df['Country'] == "France"]

df_fr.shape  # (8342, 8)

# fatura ürün ve hangi üründen kaçar tane alınmış hesaplayalım
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# satırlarda invoice sütunlarda ise ürünleri görmek istediğimiz için unstack ya da pivot table ı kullanıyoruz
# bu satır ve sütunların kesişimlerine de birşeyler yazacağız var yok gibi


# unstack() fonksiyonu ile bunu pivot etmiş oluyoruz yani description isimlendirmelerini değişkne isimlerine çeviriyoruz
# iloc ile de index based seçim yap dedik ve satırlardan ve sütunlardan 5 er tane getir komutunu verdik, gözlemliyoruz
# satırlar fatura id lerine göre tekil çıkacak şimdi
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]


# şimdi kesişimlerde 0 ve 1 ler görmek istediğimiz için unstack işleminden sonra boş olan yani NaN olan yerleri 0 ile
# dolduracağız, ürün olan yerleri 1 ile dolduracağız
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]


# şimdi burada 24 yazan yerde 1 yazmasını istiyoruz, bunun için bir fonksiyon yazcağız applymap fonksiyonu kullandık
df_fr.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


# şimdi bu fonksiyon, istersek ürünleri stock code ile istersek description ile verecek, id=True dersek stock code ile
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)
fr_inv_pro_df.head()

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)
fr_inv_pro_df.head()

# check_id diye bir fonksiyon tanımlıyoruz ki stock koduna bakmak istersek bunun descriptionunu bize getirecek
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 10120)


# 3. BASAMAK
# 3. Birliktelik Kurallarının Çıkarılması (Extracting Association Rules)

# apriori metodu ile olası tüm ürün birlikteliklerinin support değerlerini yani olasılıklarını bulacağız
frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)


# frequent_itemset değişkenindeki itemları support değerine göre büyükten küçüğe doğru sıralayalım
frequent_itemsets.sort_values("support", ascending=False).head(20)


# DataFrame'lerde boolean olmayan tiplerin kullanımının daha kötü bir hesaplama performansına neden olabileceği
# ve gelecekte boolean olmayan DataFrame'lerin desteğinin kesilebileceği uyarısını aldık
# bu nedenle dataframe tipini bool a çeviriyoruz
frequent_itemsets = apriori(fr_inv_pro_df.astype(bool),
                            min_support=0.01,
                            use_colnames=True)


# association_rules metodunu getiriyoruz
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

rules.head(50)

# antecedents: önceki ürün
# consequents: ikinci ürün
# antecedent support: ilk ürünün tek başına gözlenme olasılığı
# consequent support: ikinci ürünün tek başına gözlenme olasılığı
# support: antecedents ve consequents in birlikte gözükme olasılığı
# confidence: antecedents ürünü alındığında consequents ürününün de alınma olasılığı
# lift: antecedents ürünü alındığında consequents ürününün alınma olasılığının kaç kat arttığı, yansız bir metriktir
# leverage: kaldıraç yani lifte benzer bir değerdir, supportu yüksek olan değerlere öncelik verme eğilimindedir
# conviction: y ürünü olmadan x ürününün beklenen frekansıdır ya da x ürünü olmadan y ürününün beklenen frekansıdır

# şimdi belirli kurallara göre ürün seçimi yapalım
rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]
rules.shape

check_id(df_fr, 21086)

# confidence a göre sıralama yapıyoruz ürün önereceğimiz için
rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].sort_values("confidence", ascending=False)


# 4. BASAMAK
# 4. Çalışmanın Scriptini Hazırlama (Preparing the Script of Study)

# bütün işlemleri derli toplu bir hale getirelim ve bir python dosyası yapalım

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe.astype(bool), min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df = df_.copy()


df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].sort_values("confidence", ascending=False)


# 5. BASAMAK
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)  # ['MINI PAINT SET VINTAGE '] şimdi ürün önermesi yapacağız bu ürüne göre

# lifte göre ürünleri sıralıyoruz büyükten küçüğe doğru
sorted_rules = rules.sort_values("lift", ascending=False)

# boş bir öneri listesi oluşturalım
recommendation_list = []

# seçtiğimiz product id ye göre antecedents ürününe karşılık gelen consequents ürününü önerme
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

# recommendation list seçtiğimiz product id ye gör oluştu içinde bir çok ürün olacak bu nedenle ilk 3 ünü getir diyoruz
recommendation_list[0:3]  # [22326, 22556, 22551]

check_id(df, 22326)  # ['ROUND SNACK BOXES SET OF4 WOODLAND ']
check_id(df, 22556)  # ['PLASTERS IN TIN CIRCUS PARADE ']
check_id(df, 22551)  # ['PLASTERS IN TIN SPACEBOY']


# Bu işlemlere de bir fonksiyon yazalım
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


# 22492 stok kodlu ürüne tavsiyeleri getirelim
arl_recommender(rules, 22492, 1)  # [22326]
arl_recommender(rules, 22492, 2)  # [22326, 22556]
arl_recommender(rules, 22492, 3)  # [22326, 22556, 22551]






