# Laporan Proyek Machine Learning Mushroom Classification - Arip Kristiyanto

## Domain Proyek

Jamur merupakan komoditas makanan yang memiliki banyak nutrisi yang baik bagi tubuh. Bahkan jamur dapat menjadi suatu obat bagi penyakit tertentu [1]. Jamur memiliki aroma dan rasa yang unik yang berbeda dari bahan makanan lain. Hal tersebut membuat jamur yang dapat dimakan memiliki tingkat permintaan yang tinggi dalam pasar bahan makanan [2]. 
Di Indonesia yang memiliki iklim tropis dan ada banyak kayu yang sangat cocok untuk pertumbuhan jamur, jamur merupakan salah satu tumbuhan liar yang banyak tersebar di alam bebas. Namun, beberapa jamur ini ada yang beracun dan harus dihindari. Jamur yang tersebar di dunia saat ini diperkirakan sedikitnya ada 45.000 jenis dan 2.000 jenis diantaranya tidak beracun. Ada banyak spesies jamur mematikan yang ada di dunia. Ada sekitar 70 hingga 80 spesies yang paling beracun dan mematikan yang bisa berakibat fatal jika dikonsumsi [3].
terdapat beberapa jenis jamur yang beracun, bahkan dapat membahayakan nyawa [4]. Pada saat ini diperkirakan 140.000 spesies jamur diidentifikasi dan 2.000 diantaranya aman untuk dikonsumsi manusia [5]. Dimana 138.000 species lain masih belum bisa ditentukan dapat dimakan atau tidak dan bahkan mengandung racun yang membahayakan tubuh. Dilaporkan menurut Center for Disease Control and Prevention (CDC) lebih dari 41.000 orang meninggal pada tahun 2008 dikarenakan secara tidak sengaja keracunan, sedangkan World Health Organization (WHO) melaporkan sekitar 0,346 juta kematian sejak tahun 2004.
Berdasarkan permasalahan tersebut agar dapat mengidentifikasi jamur yang dapat dimakan atau beracun.

## Business Understanding
Berdasarkan data  Dilaporkan menurut Center for Disease Control and Prevention (CDC) lebih dari 41.000 orang meninggal pada tahun 2008 dikarenakan secara tidak sengaja keracunan, sedangkan World Health Organization (WHO) melaporkan sekitar 0,346 juta kematian sejak tahun 2004. Artinya cukup tinggi kematian yang diakibatkan jamur beracun. Salah satu manfaat dari adanya model klasifikasi jamur beracun atau tidak  ini adalah model ini dapat digunakan untuk mengedukasi masyarakat terutma masyarakat pedesaan.

### Problem Statements
   1. Berdasarkan eksplorasi terhadap dataset, fitur-fitur apa saja yang dapat menentukan atau memberi pengaruh terhadap klasifikasi beracun atau tidaknya jamur?
   2. Bagaimana memproses dataset agar dapat digunakan untuk pembuatan model machine learning klasifikasi jamur beracun atau tidak?
   3. Bagaimana cara medapatkan model klasifikasi jamur beracun atau tidak dengan performa terbaik?


### Goals
1. Melakukan eksplorasi semua fitur-fitur yang terdapat pada dataset dan melihat fitur-fitur mana saja yang memiliki pengaruh besar atau memiliki korelasi tinggi terhadap label klasifikasi jamur.
2. Melakukan data preparation untuk mempersiapkan model untuk proses training.
3. Melakukan proses training dengan baseline model dari berbagai algoritma dan menggunakan baseline model dengan performa terbaik untuk melakukan hyperparameter tuning. 

### Solution Approach

1. Melakukan eksplorasi fitur dilakukan analisis univariat dan multivariat untuk menemukan hubungan antar fitur baik data numerik maupun data kategorikal. Kemudian, menggunakan barchart, heatmap, dan correlation matrix untuk medapatkan informasi lebih lanjut
2. Mendapatkan data yang bersih untuk diproses ke tahap modelling, dilakukannya proses data preparation yang terdiri dari data cleaning, train test split, dan data transformation. Kebersihan data dapat mempengaruhi performa model yang akan dibuat.
3. Mendapatkan model dengan performa terbaik, digunakan 3 algoritma sebagai baseline model, yaitu KNN, SVM, dan Random Forest. Kemudian, untuk dapat mengetahui baseline model mana yang memiliki performa terbaik dapat dilakukan evaluasi menggunakan Confusion Matrix (Accuracy, Precision, Recall, F1 Score) yang juga divisualisasikan. Selanjutnya, model yang terpilih akan dibantu dengan grid search untuk menemukan hyperparameter yang memiliki performa terbaik. Terakhir, model tersebut dilakukan evaluasi menggunakan Confusion Matrix (Accuracy, Precision, Recall, F1 Score).
   
## Data Understanding

Dataset yang digunakan untuk pembangunan model machine learning ini adalah dataset "Mushroom Dataset (Binary Classification)" yang tersedia di situs web Kaggle. Dataset ersebut adalah dataset kuantitatif yang berisi kolom-kolom yang dapat menentukan sebuah jmaur beracun atau tidak. Dataset ini memiliki 54035 baris dan 9 kolom data.
Dataset ini cocok untuk membangun model supervised learning, khususnya binary classification. Dalam kasus ini adalah untuk mengklasifikasinya sampel sebuah jamur beracun atau tidak 
Dataset tersebut dapat diunduh disini [dataset](https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset/data).
Berikut ini adalah informasi lainnya mengenai variabel-variabel yang terdapat di dataset tersebut:

*  Cap Diameter = Diameter Tutup/topi
*   Cap Shape = Bnetuk tutup
*    Gill Attachment = penempelan ingsang
*    Gill Color = warna ingsang
*   Stem Height = tinggi batang
*    Stem Width = lebar batang
*    Stem Color = warna batang
*   Season = musim
*    Class = target class dapat dimakan atau tidak (0 dapat diamakan, 1 beracun)


### Exploratory data analysis
Exploratory Data Analysis (EDA) adalah pendekatan analisis data yang bertujuan untuk memahami karakteristik utama dari kumpulan data. EDA melibatkan penggunaan teknik statistik dan visualisasi grafis untuk menemukan pola, hubungan, atau anomali untuk membentuk hipotesis. Proses ini sering kali tidak terstruktur dan dianggap sebagai langkah awal penting dalam analisis data yang membantu menentukan arah analisis lebih lanjut.

Berikut ini adalah EDA yang dilakukan:
``` python
    df.shape 
  ``` 
Hasilnya
  ```
  (54035,9)
  ```
 Berdasarkan output diatas, didapatkan informasi:
  * Terdapat 54035 baris data
  * Tedapat 9 kolom
Pada bagian ini, belum dapat diketahui **nama** dari **kolom-kolom** yang ada.
 ``` python
  df.keys()
  ```
Hasilnya
  ``` python
  Index(['cap-diameter', 'cap-shape', 'gill-attachment', 'gill-color',
       'stem-height', 'stem-width', 'stem-color', 'season', 'class'],
      dtype='object')
  ```
Selanjutnya untuk melihat tipedata

 ``` python
# Menampilkan tipe data dari setiap kolom yang ada
df.info()
 ```
Hasilnya
 ```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 54035 entries, 0 to 54034
Data columns (total 9 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   cap-diameter     54035 non-null  int64  
 1   cap-shape        54035 non-null  int64  
 2   gill-attachment  54035 non-null  int64  
 3   gill-color       54035 non-null  int64  
 4   stem-height      54035 non-null  float64
 5   stem-width       54035 non-null  int64  
 6   stem-color       54035 non-null  int64  
 7   season           54035 non-null  float64
 8   class            54035 non-null  int64  
dtypes: float64(2), int64(7)
memory usage: 3.7 MB
```
Selanjutnya 
``` python
# Menampilkan statistika deskriptif unuk setiap kolom
df.describe()
  ```
Hasilnya

|    |cap-diameter	 |cap-shape	  |   gill-attachment	|gill-color	 |   stem-height	|    stem-width	|    stem-color	|    season	    |    class|
|:------|-----------:|-----------:|----------:|--------------:|----------:|----------:|----------:|----------:|----------:|
|count   | 54035.000000	| 54035.000000	| 54035.000000	|    54035.000000	|54035.000000	|54035.000000	|54035.000000	|54035.000000	|54035.000000|
|mean   |  567.257204	|     4.000315	 |    2.142056	|        7.329509	|    0.759110	|    1051.081299	|    8.418062|    0.952163	|    0.549181|
|std	  |   359.883763	|     2.160505	  |  2.228821	  |      3.200266	|   0.650969	|    782.056076	 |   3.262078	|    0.305594	  |  0.497580|
|min	|     0.000000	|   0.000000	|     0.000000	|        0.000000	 |   0.000426	|    0.000000	|    0.000000	|    0.027372|	    0.000000|
|25%	  |   289.000000	|     2.000000	  |   0.000000	  |      5.000000	 |   0.270997	|    421.000000|	    6.000000	|    0.888450	  |  0.000000|
|50%	  |   525.000000	|     5.000000	 |    1.000000	  |      8.000000	|    0.593295	|    923.000000	|    11.000000|	    0.943195	|    1.000000|
|75%	 |    781.000000	  |   6.000000	  |   4.000000	    |    10.000000	|    1.054858	|    1523.000000	|    11.000000|	    0.943195	|    1.000000|
|max	  |   1891.000000	| 6.000000	|     6.000000	 |       11.000000	|    3.835320	 |   3569.000000	  |  12.000000	|    1.804273	|    1.000000|

Berdasarkan _output_ tersebut, didapatkan informasi mengenai statistika deskriptif dari _dataset_ yang digunakan. Berikut ini adalah keterangan untuk setiap bagian:
   * ```count``` : Jumlah data dari sebuah kolom
   * ```mean``` : Rata-rata dari sebuah kolom
   * ```std``` : Standar deviasi dari sebuah kolom
   * ```min``` : Nilai terendah pada sebuah kolom
   * ```25%``` : Nilai kuartil pertama (Q1) dari sebuah kolom
   * ```50%``` : Nilai kuartil kedua (Q2) atau median atau nilai tengah dari sebuah kolom
   * ```75%``` : Nilai kuartil ketiha (Q3) dari sebuah kolom
   * ```max``` : Nilai tertinggi pada sebuah kolom

### Visualisasi data
#### Univariate Analysis
Univariate Analysis adalah jenis analisis data yang memeriksa satu variabel (atau bidang data) pada satu waktu. Tujuannya adalah untuk menggambarkan data dan menemukan pola yang ada dalam distribusi variabel tersebut. Ini termasuk penggunaan statistik deskriptif, histogram, dan box plots untuk menganalisis distribusi dan memahami sifat dari variabel tersebut.
``` python
# Membuat count plot
sns.countplot(x='class', data=df, color='#30D5C8')
sns.despine()
plt.title('Count Plot for Categorical Column')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
```
Hasilnya Univariate Analysis Categorical Column

![image](https://github.com/user-attachments/assets/5354a8bd-1169-4b8f-8827-e5e804cce858)
                             Gambar 1. 

Berdasarkan Gambar diatas, terlihat bahwa class memiliki dua unique value, yaitu '0' yang menyatakan jamur bisa dimakan dan '1' yang menyatakan jamur beracun. Namun, terlihat juga bahwa adanya imbalance data atau ketidakseimbangan data. nilai '0' memiliki baris data hingga nyaris 24000 baris data, sedangkan nilai '1' hanya memiliki sekitar 30000 baris data. Berangkat dari informasi ini, perlu dilakukan penyeimbangan agar tidak terjadi bias pada model machine learning yang akan dibangun.

![image](https://github.com/user-attachments/assets/830d6a82-c1cd-4ae8-bf94-86515b4c0a9a)
            Gambar 2. Univariate Analysis Numeric Column

Berdasarkan Gambar 2., gambar ini menampilkan setiap kolom numerik yang ada pada dataset, cap-diameter, cap-shape, gill-attachment, gill-color,	stem-width , stem-color

#### Multivariate Analysis

Multivariate Analysis adalah prosedur statistik yang digunakan untuk memeriksa hubungan antara beberapa variabel secara bersamaan. Teknik ini mencakup berbagai metode seperti regresi berganda, analisis faktor, dan analisis kluster, yang membantu dalam memahami struktur dan pola yang kompleks dalam data dengan lebih dari satu variabel.
Data Preparation
![Untitled](https://github.com/user-attachments/assets/73496745-4224-4bc0-a499-6a59fa2909c6)
 <div align="center">Gambar 3 - Multivariate Analysis Categorical Column - Every Numeric Column</div>

![Untitled](https://github.com/user-attachments/assets/e89b7cb1-b772-437b-8afc-d7272907244c)
 <div align="center">Gambar 4 - Multivariate Analysis Categorical Column - Numeric Column based on class</div>
 Berdasarkan gambar kedua visualisasi data diatas, dapat terlihat nyaris semua variabel terpisah menunjukkan karakteristik atau pola khusus terhadap variabel label, yaitu 'Class'. 0 dan 1 (ditandai dengan warna oren dan biru)

### Correlation

Uji Korelasi adalah metode statistik yang digunakan untuk menentukan apakah ada hubungan antara dua variabel kuantitatif dan seberapa kuat hubungan tersebut. Uji ini menghasilkan nilai koefisien korelasi, seperti Pearson atau Spearman, yang berkisar antara -1 hingga +1. Nilai mendekati +1 menunjukkan korelasi positif yang kuat, sedangkan nilai mendekati -1 menunjukkan korelasi negatif yang kuat. Nilai mendekati 0 menunjukkan tidak adanya korelasi. Uji korelasi penting dalam menentukan arah dan kekuatan hubungan antar variabel, yang dapat membantu dalam pemodelan prediktif dan analisis penyebab.

   ![Untitled](https://github.com/user-attachments/assets/962667d3-5a75-4f77-8d0d-2b3eaa662eb5)
   <div align="center">Gambar 4 - Multivariate Analysis Categorical Column - Numeric Column based on class</div>
Berdasarkan  visualisasi diatas, terlihat bahwa kolom ```season```, ```gill color```, ```gill attachement```, memiliki skor korelasi yang paling kecil terhadap label. Kolom yang semacam ini baiknya di-drop saja untuk meringankan beban komputasi dan mengurangi dimensi dari dataset yang akan digunakan dalam pelatihan model

### Missing Value

   Missing Values adalah data yang hilang atau tidak tercatat dalam dataset. Hal ini bisa terjadi karena berbagai alasan, seperti kesalahan entri data, kerusakan data, atau tidak tersedianya informasi saat pengumpulan data. Missing values dapat mempengaruhi kualitas model _machine learning_ dan hasil analisis statistik. Oleh karena itu, penting untuk mengidentifikasi, menganalisis, dan mengatasi missing values dengan metode seperti imputasi, di mana nilai yang hilang diganti dengan estimasi, atau dengan menghapus baris atau kolom yang terdampak.
    
Menjumlah total missing value pada dataset
``` python
df.isnull().sum()
```
Hasilnya
``` python
cap-diameter	0
cap-shape	0
gill-attachment	0
gill-color	0
stem-height	0
stem-width	0
stem-color	0
season	0
class	0
dtype: int64
```
Berdasarkan Output diatas tidak ditemukan mising value

### Duplikat data
Berfungsi untuk mengecek apakah terdapat duplikat data dalam dataset
``` python
# Cek baris duplikat dalam dataset
duplicates = df.duplicated()

# Hitung jumlah baris duplikat
duplicate_count = duplicates.sum()

# Cetak jumlah baris duplikat
print(f"Number of duplicate rows: {duplicate_count}")
```
Hasilnya
``` python
Number of duplicate rows: 303
```
Berdasarkan hasil diatas menunjukkan terdapat 303 data duplikat

### Outlier
Outlier adalah nilai yang jauh berbeda dari nilai lainnya dalam kumpulan data. Nilai ini muncul sebagai pengecualian dalam pola data yang ada.
``` python
feature_columns = df.select_dtypes(include=[np.number]).drop('class', axis=1)
feature_columns.plot(kind='box', subplots=True, layout=(1, len(feature_columns.columns)), figsize=(12, 8))
plt.tight_layout()
plt.show()
```
![Untitled](https://github.com/user-attachments/assets/f2190683-a27c-450f-b753-474ddeab557b)
 <div align="center">Gambar 5 - Outlier</div>
Berdasarkan boxplots diatas, semua kolom numerik memiliki outliers-nya masing-masing. Outliers perlu dihapus untuk mendapatkan model dengan performa yang bagus.

## Data Preparation
Data Preparation adalah proses pembersihan, transformasi, dan pengorganisasian data mentah ke dalam format yang dapat dipahami oleh algoritma pembelajaran mesin. Berikut ini adalah **urutan** langkah-langkah Data Preparation yang dilakukan beserta penjelasan dan alasannya:

### Feature selection
#### Dropping Column with Low Correlation
pada bagian ini adalah proses penghapusan fitur-fitur yang memiliki korelasi rendah terhadap variabel target dari dataset. Langkah ini diambil berdasarkan asumsi bahwa fitur dengan korelasi rendah tidak memberikan kontribusi signifikan terhadap prediksi yang dibuat oleh model.

**Alasan**: Tahapan ini perlu dilakukan karena fitur dengan korelasi rendah terhadap variabel target cenderung tidak memberikan informasi yang berguna untuk prediksi dan dapat menambahkan kebisingan yang tidak perlu ke dalam model. Dengan menghilangkan fitur-fitur ini, kita dapat mengurangi kompleksitas model, yang dapat membantu dalam mencegah _overfitting_ dan mempercepat waktu pelatihan. Selain itu, model yang lebih sederhana dengan fitur yang lebih sedikit lebih mudah untuk diinterpretasikan, yang memungkinkan kita untuk lebih memahami bagaimana fitur-fitur tersebut mempengaruhi variabel target.

Berdasarkan visualisasi pada data understading, terlihat bahwa kolom season, gill color, gill attachement, memiliki skor korelasi yang paling kecil terhadap label.
```python
# Mendefinisikan daftar fitur dengan korelasi rendah terhadap variabel target
low_corr = ['season', 'gill-attachment', 'gill-color']

# Menghapus fitur-fitur tersebut dari dataset
# Axis=1 menunjukkan bahwa operasi penghapusan dilakukan pada kolom (fitur)
df = df.drop(low_corr, axis=1)
df
```
| 	|cap-diameter| 	cap-shape| 	stem-height 	|stem-width 	|stem-color 	|class|
  |:------|-----------:|-----------:|----------:|--------------:|----------:|---------------:|
|0 	|1372 	|2 	|3.807467 |	1545 	|11 |	1|
|1 	|1461 	|2 	|3.807467 |	1557 	|11 | 1|
|2 	|1371 	|2 	|3.612496 |1566 	|11 | 1|
|3 	|1261 	|6 	|3.787572 |	1566 |11  |1|
|4 	|1305 	|6 	|3.711971 |	1464 	|11 |1|
|... 	|... 	|... |	...| 	...| 	... |	...|
|54030 	|73 	|5 	|0.887740 	|569 |	12 	|1
|54031 	|82 	|2 	|1.186164 	|490 	|12 	|1
|54032 	|82 	|5 	|0.915593 	|584 	|12 	|1
|54033 	|79 	|2 	|1.034963 	|491 	|12 	|1
|54034 	|72 	|5 	|1.158311 	|492 	|12 	|1

Penghapusan kolom dengan korelasi rendah sudah berhasil dilakukan. Berdasarkan dataframe diatas, tersisa 6 kolom. 1 kolom label dan 5 kolom numerik.

### Data Cleaning
  
Data cleaning adalah adalah langkah penting dalam proses Machine Learning karena melibatkan identifikasi dan penghapusan data yang hilang, duplikat, atau tidak relevan yang terdapat pada dataset. Proses ini memiliki berbagai langkah yang perlu dilakukan supaya dataset siap digunakan untuk pembangunan model Machine Learning.
    
**Alasan**: _Data Cleaning_ diperlukan agar data yang digunakan akurat, konsisten, dan bebas kesalahan, karena data yang salah atau tidak konsisten dapat berdampak negatif terhadap performa model Machine Learning

#### Removal Duplicates
      
Data duplikat adalah baris data yang sama persis untuk setiap variabel yang ada. Dataset yang digunakan perlu diperiksa juga apakah dataset memiliki data yang sama atau data duplikat. Jika ada, maka data tersebut harus ditangani dengan menghapus data duplikat tersebut.

**Alasan**: Data duplikat perlu didektesi dan dihapus karena jika dibiarkan pada dataset dapat membuat model Anda memiliki bias, sehingga menyebabkan _overfitting_. Dengan kata lain, model memiliki performa akurasi yang baik pada data pelatihan, tetapi buruk pada data baru. Menghapus data duplikat dapat membantu memastikan bahwa model Anda dapat menemukan pola yang ada lebih baik lagi.
**Setelah ditemukan 303 duplikat dalam proses data understanding   maka akan kita hapus**
Berikut ini adalah proses pendeteksian dan penghapusan data duplikatnya:
 ```python
 # hapus duplikat data
dfclean=df.drop_duplicates()
      
 # Hitung jumlah baris duplikat
 duplicate_count = duplicates.sum()
      
 # Cetak jumlah baris duplikat
 print(f"Number of duplicate rows: {duplicate_count}")
      ```
 Berikut ini adalah hasilnya:

 ```python
 Number of duplicate rows: 0
 ```
Berdasarkan hasil diatas data duplikat berhasil dihapus
   
#### Handle Missing Value
Berdasarkan data understanding diatas, tidak ada mising value  

#### Outliers Detection and Removal
      
Outliers adalah titik data yang secara signifikan berbeda dari sebagian besar data dalam kumpulan data. Outliers dapat muncul karena variasi dalam pengukuran atau mungkin menunjukkan kesalahan eksperimental; dalam beberapa kasus, outliers bisa juga menunjukkan variabilitas yang sebenarnya dalam data. Penting untuk menganalisis outliers karena mereka dapat memiliki pengaruh besar pada hasil analisis statistik.
 
Proses pembersihan outliers menggunakan metode IQR (Interquartile Range) melibatkan beberapa langkah:
      
   - Menghitung Kuartil: Tentukan kuartil pertama (Q1) dan kuartil ketiga (Q3) dari data. Kuartil ini membagi data menjadi empat bagian yang sama.
   - Menghitung IQR: Hitung IQR dengan mengurangi Q1 dari Q3:
          $$IQR=Q3−Q1$$
   - Menentukan Batas Outliers:
           Batas bawah untuk outliers:
           $$Q1−1.5×IQR$$
            
           Batas atas untuk outliers:
           $$Q3+1.5×IQR$$
      - Identifikasi Outliers: Data yang berada di luar batas bawah dan atas ini dianggap sebagai outliers.
        Pembersihan _Outliers_ yang teridentifikasi kemudian dapat dibersihkan dari dataset, baik dengan menghapusnya atau melakukan transformasi tertentu.
    
**Alasan**:_Outliers_ perlu dideteksi dan dihapus karena jika dibiarkan dapat merusak hasil analisis statistik pada kumpulan data sehingga menghasilkan performa model yang kurang baik. Selain itu, Mendeteksi dan menghapus _outlier_ dapat membantu meningkatkan performa model _Machine Learning_ menjadi lebih baik.

**Berdasarkan boxplots dalam data uderstanding, semua kolom numerik memiliki outliers-nya masing-masing. Outliers perlu dihapus untuk mendapatkan model dengan performa yang bagus.**

  Berikut ini adalah kode untuk menghapus _outliers_ yang ada pada dataframe:
```python
# Assuming 'dfclean' is your DataFrame
Q1 = dfclean.quantile(0.25)
Q3 = dfclean.quantile(0.75)
IQR = Q3 - Q1

# Define bounds for what is considered an outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = dfclean[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]   

```
Outliers yang berada pada setiap kolom sudah dihapus.
```python
df.shape
```
```python
(50667, 6)
```
Berdasarkan output diatas, kini dataframe memiliki:
   * 6 Kolom
   * 50667 baris data
Penghapusan _outliers_ sudah berhasil dilakukan.

#### Imbalance Data
      
Imbalance data adalah kondisi di mana kelas atau kategori dalam dataset tidak diwakili secara merata, dengan satu kelas mendominasi yang lain. Jika hal ini dibiarkan hingga proses pelatihan model dapat mengakibatkan bias pada model. Hal ini bisa diatasi dengan _oversampling_ atau _undersampling_.

**Alasan**: Hal ini dapat menjadi masalah adalah karena _imbalance_ _data_ dapat menyebabkan model bias terhadap kelas mayoritas (lebih banyak) dan menghasilkan performa yang buruk pada kelas minoritas lebih sedikit)
 Berikut ini adalah untuk memeriksa ada berapa baris data untuk masing-masing kelas pada kolom 'Class':
 ```python
 count_0 = df[df['class'] == 0].shape[0]
 count_1 = df[df['class'] == 1].shape[0]
 print("Jumlah baris data yang bernilai '0' ada sebanyak: " + str(count_0))
 print("Jumlah baris data yang bernilai '1' ada sebanyak: " + str(count_1))
 ```
Berikut ini adalah hasilnya:
```python
Jumlah baris data yang bernilai '0' ada sebanyak: 23144
Jumlah baris data yang bernilai '1' ada sebanyak: 27523
```
Berdasarkan output diatas, dataset memiliki ketidakseimbangan jumlah kelas. Hal ini jika dibiarkan data mengakitbatkan bias-nya model.
Untuk mengatasinya, dilakukan undersampling untuk kelas `0` agar menyesuaikanjumlah dengan kelas `1`.

```python
#Melakukan undersampling
df = df.groupby('class').apply(lambda x: x.sample(min(len(x), min(count_0, count_1)))).reset_index(drop=True)
```
Berikut ini adalah untuk memeriksa ada berapa baris data untuk masing-masing kelas pada kolom 'Class' setelah dilakukan undersampling:
```python
count_0 = df[df['class'] == 0].shape[0]
count_1 = df[df['class'] == 1].shape[0]
print("Jumlah baris data yang bernilai '0' ada sebanyak: " + str(count_0))
print("Jumlah baris data yang bernilai '1' ada sebanyak: " + str(count_1))
```
Berikut ini adalah hasilnya:
```python
Jumlah baris data yang bernilai '0' ada sebanyak: 23144
Jumlah baris data yang bernilai '1' ada sebanyak: 23144
```
Berdasarkan output diatas, dataset memiliki sudah jumlah kelasnya sudah seimbang. `0` dan `1` sudah memiliki jumlah baris data yang sama. 
### Train Test Split
  
Train Test Split adalah metode yang digunakan untuk membagi dataset menjadi dua bagian: satu untuk melatih model (_training set_) dan satu lagi untuk menguji model (_testing set_). Biasanya, data dibagi dengan proporsi tertentu, misalnya 80% untuk training dan 20% untuk testing.

**Alasan**: Proses ini dilakukan agar dapat mengevaluasi kinerja model secara objektif. Dengan memisahkan data uji, kita dapat mengukur seberapa baik model memprediksi data baru yang tidak pernah dilihat sebelumnya, yang merupakan indikator penting dari kemampuan generalisasi model.

Berikut adalah bagian untuk membagi dataset menjadi train set dan test set:
```python
X = df.drop(["class"], axis =1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```
Proses train test split berhasil dilakukan. Dataset dibagi menjadi 80% untuk train dan 20% untuk test.
```pyhton
train_count = X_train.shape
test_count = X_test.shape
print("Dataset training memiliki data sebanyak " + str(train_count[0]) + " baris")
print("Dataset test memiliki data sebanyak " + str(test_count[0]) + " baris")
```
Berikut ini adalah hasilnya:
```python
Dataset training memiliki data sebanyak 37030 baris
Dataset test memiliki data sebanyak 9258 baris
```
Berikut ini adalah bagian untuk memeriksa ada berapa baris data untuk train dan test pada dataframe variabel label:
  ```python
  train_count_label = y_train.shape
  test_count_label = y_test.shape
  print("Dataset label training memiliki data sebanyak " + str(train_count_label[0]) + " baris")
  print("Dataset label test memiliki data sebanyak " + str(test_count_label[0]) + " baris")
  ```

  Berikut ini adalah hasilnya:
  ```pyhton
  Dataset label training memiliki data sebanyak 37030 baris
 Dataset label test memiliki data sebanyak 9258 baris
  ```
### Data Transformation
  
Data Transformation adalah proses mengubah data dari satu format atau struktur ke format atau struktur lainnya. Proses ini biasanya dari format sistem sumber menjadi yang dibutuhkan oleh sistem tujuan. _Data Transformation_ dapat dilakukan dengan berbagai cara, seperti mengubah satuan ukuran data, mengubah distribusi data, atau mengubah bentuk data.
    
**Alasan**: Data Tranformasi perlu dilakukan karena dapat meningkatkan efisiensi dan meningkatkan kualitas data yang digunakan dalam pembuatan model _Machine Learning._

#### Standardization

Standardisasi adalah proses mengubah data menjadi format yang lebih seragam dan dapat dibandingkan. Ini biasanya melibatkan pengurangan rata-rata (mean) dan pembagian dengan simpangan baku (standard deviation) untuk setiap fitur, sehingga fitur tersebut akan memiliki rata-rata nol dan varians satu.

**Alasan**: Standardisasi perlu dilakukan karena banyak algoritma _machine learning_ yang berperforma lebih baik jika fitur-fitur berada pada skala yang sama. Standardisasi membantu dalam hal ini dengan memastikan bahwa setiap fitur berkontribusi secara proporsional ke hasil akhir dan menghindari bias terhadap fitur dengan skala yang lebih besar.

Berikut ini adalah penerapan standardisasinya:
```python
X_train[:] = scaler.fit_transform(X_train[:])
```
Hasilnya 
| |cap-diameter| 	cap-shape| 	stem-height| 	stem-width 	|stem-color|
|:------|-----------:|-----------:|----------:|--------------:|----------:|
|count 	|37030.0000 	|37030.0000| 	37030.0000 |	37030.0000| 	37030.0000|
|mean 	|0.0000 	|0.0000 	|-0.0000 	|-0.0000 	|0.0000|
|std 	|1.0000 	|1.0000 	|1.0000 |	1.0000 	|1.0000|
|min 	|-1.6472 	|-1.8810 	|-1.3005 	|-1.3952 	|-2.5834|
|25% 	|-0.7839 	|-0.9488 	|-0.8060 	|-0.8195 	|-0.7422|
|50% 	|-0.1053 	|0.4494 	|-0.2151 	|-0.1578 	|0.7922|
|75% 	|0.6641 	|0.9155 	|0.5990 	|0.6510 	|0.7922|
|max 	|2.9571 	|0.9155 	|3.0086 	|3.0024 	|1.0991|

Setelah dilakukannya standardisasinya, dapat kita cek hasilnya perubahannya dengan melihat mean dan standar deviasinya

## Modelling
Pada bagian ini, data yang yang sudah dibagi menjadi dua bagian menjadi _training dataset_ dan _test dataset_ siap untuk digunakan untuk pembangunan model _Machine Learning_-nya. Untuk kasus ini, digunakan 3 (tiga) _baseline model_ dari 3 algoritma yang berbeda. Berikut ini adalah ketiga algoritma tersebut:
- Random Forest
  - Kelebihan
    - Akurasi tinggi
      
      Dengan menggunakan banyak pohon keputusan, Random Forest dapat mencapai tingkat akurasi yang tinggi dalam klasifikasi dan regresi.
      
    - Dapat menangani data dengan dimensi tinggi
      
      Mampu menangani masalah dengan sejumlah besar fitur dan data tanpa perlu pemilihan fitur.
      
    - _Robust_ terhadap _noise_ dan _outliers
      
      Tidak mudah dipengaruhi oleh noise dan outliers karena menggunakan metode bagging dan ensemble.
      
  - Kekurangan
    - Mahal secara komputasi
      
      Membutuhkan lebih banyak sumber daya komputasi karena melibatkan pembangunan banyak pohon.
      
    - Butuh waktu lebih lama
      
      Proses pembelajaran dan prediksi bisa memakan waktu yang lama karena kompleksitas model.
      
    - Interpretabilitas
      
      Sulit untuk diinterpretasikan dan dipahami karena kompleksitas dari banyak pohon keputusan.
      
- KNN
  - Kelebihan
    - Sederhana dan Mudah Dipahami
      
      Algoritma ini intuitif dan mudah diimplementasikan.
    
    - Non-parametric
      
      Tidak membuat asumsi tentang distribusi data, cocok untuk data yang tidak normal.
      
    - Tidak perlu pelatihan
      
      Tidak ada fase pelatihan yang eksplisit, yang berarti baru pada saat prediksi data diuji
      
  - Kekurangan
    - Sensitif terhadap _outliers_
      
      Kinerja bisa terpengaruh oleh keberadaan outliers.
      
    - Mahal secara komputasi
      
      Memerlukan perhitungan jarak dari setiap titik data ke titik lainnya, yang bisa sangat mahal secara komputasi.
    
    - Memerlukan pilihan K yang baik
      
      Pemilihan jumlah tetangga (K) yang tepat sangat penting untuk kinerja algoritma.
      
- SVM
  - Kelebihan
    - Efektif untuk data dengan dimensi tinggi
      
       Bekerja dengan baik pada data yang memiliki banyak fitur.
      
    - Serbaguna
      
      Kernel yang berbeda dapat digunakan untuk keputusan batas yang berbeda.
      
    - Robust
      
      Tidak terlalu dipengaruhi oleh outliers dan mampu menghasilkan model yang optimal dengan margin yang maksimal.
      
  - Kekurangan
    - Sensitif terhadap pilihan Kernel
      
      Pemilihan kernel yang tepat sangat penting dan bisa mempengaruhi kinerja model.
    - Membutuhkan penyetelan Hyperparameter
    
      Penyetelan hyperparameter seperti C, gamma, dan kernel memerlukan waktu dan usaha.
    
    - _Training Cost_
      
  Biaya pelatihan bisa tinggi, terutama untuk dataset yang besar.
Kemudian, _baseline model_ dari ketiga algoritma tersebut yang memiliki akurasi tertinggi digunakan untuk ke tahap selanjutnya. Selanjutnya, algoritma tersebut digunakan kembali untuk pembangunan model, tetapi dengan memanfaatkan _hyperparameter_ yang ada sehingga mendapatkan hasil terbaik. Untuk menemukan _hyperparamter_ yang memberikan hasil terbaik, ```GridSearch``` digunakan ke model yang terpilih.

Berikut ini adalah hasil dari _baseline model_ untuk ketiga model:

|     |    train	|    test|
|:----|----------:|------------:|
|KNN 	|0.000936 	|0.000909|
|SVM|	0.000638 	|0.000646|
|RF 	|0.001 	|0.000916| 

![Untitled](https://github.com/user-attachments/assets/d91de7fc-0c22-42f0-8db7-736ae7698aff)
<div align="center">Gambar 6 - Baseline Model Train Test Results</div>

![Untitled-1](https://github.com/user-attachments/assets/8c6d82aa-9c6b-4e12-868d-bc18ae556404)
<div align="center">Gambar 7 - Baseline Model All Results</div>

Model Random Forest terpilih sebagai model yang akan digunakan lebih lanjut dengan hyperparamter tuning karena memiliki performa train dan test yang tertinggi dibandingkan dengan 2 model lainnya. Kemudian, hasil Accuracy, Precision, Recall, dan F1 Score dari Random Forest juga menunjukkan hasil yang terbaik.

Berikut ini adalah proses improvement hyperparameter tuning menggunakan GridSearch:
```python
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation score (accuracy): {best_score}")
```

Berikut ini adalah hasil dari _grid search_:
```python
Fitting 3 folds for each of 225 candidates, totalling 675 fits
Best parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}
Best cross-validation score (accuracy): 0.8856332380672297
```

Berdasarkan hasil dari proses `GridSearch`, kombinasi parameter yang terbaik adalah:
 * `max_depth`: 10
 * `min_samples_leaf`: 1
 * `min_samples_split`: 5
 * `n_estimators`: 400

Berikut ini adalah penjelasan dari keempat parameter tersebut:

  - `max_depth`: 10
    Ini menentukan kedalaman maksimum pohon. Dalam kasus ini, pohon tidak akan tumbuh lebih dari 10 tingkat. Kedalaman yang lebih besar bisa meningkatkan keakuratan model tetapi juga meningkatkan risiko overfitting.
    
  - `min_samples_leaf`: 1
    Ini adalah jumlah sampel minimum yang diperlukan untuk menjadi daun pohon. Jadi, setiap daun harus memiliki setidaknya 1 sampel. Parameter ini membantu mengontrol overfitting dengan memastikan bahwa daun tidak terlalu spesifik hanya pada sampel pelatihan.
    
  - `min_samples_split`: 5,
    Ini menunjukkan jumlah sampel minimum yang diperlukan untuk membagi simpul internal. Sebuah simpul akan dibagi jika memiliki 5 atau lebih sampel. Ini juga membantu mencegah overfitting dengan memastikan bahwa pembagian tidak terlalu spesifik.
    
  - `n_estimators`: 400
    Ini menunjukkan jumlah pohon dalam forest. Di sini, model akan menggunakan 400 pohon. Biasanya, semakin banyak pohon, semakin stabil prediksi model, tetapi juga akan membutuhkan lebih banyak waktu komputasi dan memori.
    
Hasil dari `GridSearc` tersebut digunakan sebagai _hyperparameter_ pembangunan model.

# Evaluation
Ketika model sudah dibangun dan sudah melakukan uji dengan data test, perlu dilakukan evaluasi untuk melihat performa dari model tersebut. Untuk melakukan proses evaluasi model klasifikasi biner digunakan metrik ```Accuracy```, ```Precision```, ```Recall```, dan ```F1 Score``` dari _Confusion Matrix_.


![Untitled](https://github.com/user-attachments/assets/e5998e1d-c56f-4c8a-9406-3b461c5a96b9)
<div align="center">Gambar 8 - Confusion Matrix</div>

<br>

_Confusion Matrix_ adalah tabel yang digunakan untuk mengevaluasi performa model klasifikasi. Ini adalah tabel yang menunjukkan jumlah prediksi yang benar dan salah yang dibuat oleh model dengan membaginya ke dalam empat kategori:

- **True Positives (TP):**
  
  Ini adalah kasus-kasus di mana model dengan benar mengidentifikasi kelas positif. Misalnya, dalam konteks klasifikasi air ketika air yang layak minum dan model juga meprediksi hal yang sama.
  
- **True Negatives (TN):**
  
  Ini adalah kasus-kasus di mana model dengan benar mengidentifikasi kelas negatif. Menggunakan contoh yang sama, ini ketika air yang tidak layak minum dan model juga meprediksi hal yang sama.
  
- **False Positives (FP):**
  
  Dikenal juga sebagai ‘Type I error’, ini adalah kasus-kasus di mana model salah mengidentifikasi kelas negatif sebagai positif. Dalam konteks klasifikasi air ketika air yang tidak layak minum, tetapi model memprediksikan bahwa air tersrbut layak minum.

- **False Negatives (FN):**
  
  Dikenal juga sebagai ‘Type II error’, ini adalah kasus-kasus di mana model salah mengidentifikasi kelas positif sebagai negatif. Dalam konteks klasifikasi air ketika air yang layak minum, tetapi model memprediksikan bahwa air tersebut tidak layak minum.

Kemudian, berikut ini terkait ```Accuracy```, ```Precision```, ```Recall```, dan ```F1 Score``` dan cara kerjanya:

- ```Accuracy```

  $$Accuracy = TP + TN / TP + TN + FP + FN$$

  Akurasi adalah ukuran seberapa sering prediksi model benar dan dihitung sebagai jumlah prediksi yang benar dibagi dengan jumlah total prediksi. Akurasi memberikan informasi umum tentang performa model di semua kelas.

- ```Precision```

  $$Precision = TP / TP + FP$$

  _Precision_ mengukur proporsi prediksi positif yang benar-benar positif dan dihitung sebagai jumlah _True Positives_ dibagi dengan jumlah _True Positives_ dan _False Positives_. _Precision_ penting ketika kita ingin meminimalisir _False Positives_

- ```Recall```


  $$Recall = TP / TP + FN$$


  _Recall_ mengukur proporsi positif aktual yang diidentifikasi dengan benar dan dihitung sebagai jumlah _True Positives_ dibagi dengan jumlah _True Positives_ dan _False Negatives_. _Recall_ penting ketika biaya dari _False Negative_ tinggi, seperti di kasus medis atau keamanan.

- ```F1 Score```


  $$F1 Score = Precision  .  Recall / Precision + Recall$$

  
  F1 Score adalah rata-rata harmonik dari presisi dan recall, memberikan keseimbangan antara keduanya, terutama ketika ada distribusi kelas yang tidak seimbang. _F1 Score_ berguna ketika kita membutuhkan keseimbangan antara _presisi_ dan _recall_, dan ketika distribusi kelas tidak seimbang.

Berikut ini adalah hasil evaluasi model menggunakan metrik ```Accuracy```, ```Precision```, ```Recall```, dan ```F1 Score``` dari _Confusion Matrix_:

Berikut ini adalah ```Accuracy``` dengan menggunakan _dataset_ `test`:

```python
best_model = grid_search.best_estimator
y_pred = best_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
```
Hasilnya
```python
Test Accuracy: 0.890149060272197
```
Hasil diatas menujukkan bahwa Accuracy model menggunakan dataset test sebesar 89%. Hasilnya lebih kecil dibandingkan dengan baseline model dari Random Forest tanpa hyperparameter tuning.

Berikut ini adalah Visualisasi dari Confusin Matrix:

![Untitled](https://github.com/user-attachments/assets/f4889913-0372-40e1-bbfa-6a8801a2248d)
<div align="center">Gambar 9 - Visualisasi Confusion Matrix</div>
<br>

Berdasarkan visualisasi data diatas, hasilnya dapat dirincikan sebagai berikut:
   * True Positive (TP): 4399
   * True Negative (TN): 3842
   * False Positive (FP): 764
   * False Negative (FN): 253


Berikut ini adalah laporan lengkap terkait evaluasi model dengan metrik lainnya yang juga digunakan:
```python
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

classification_report = metrics.classification_report(y_test, y_pred)
print(f"Classification Report :\n{classification_report}")
```

Berikut ini adalah hasil dari kode diatas:

```python
Classification Report :
              precision    recall  f1-score   support

           0       0.85      0.95      0.90      4652
           1       0.94      0.83      0.88      4606

    accuracy                           0.89      9258
   macro avg       0.90      0.89      0.89      9258
weighted avg       0.89      0.89      0.89      9258
  
````

Berdasarkan output diatas, berikut ini adalah hasil akhir dari model yang dibangun dengan algoritma `Random Forest` dengan hyperparameter tuning:
- `Accuracy` : 0.89
- `Precision`: 0.89
- `Recall`: 0.89
- `F1-Score`: 0.89

## Referensi

####   [1] Temesgen, Tasisa. 2018. “Application of Mushroom as Food and Medicine.” Advances in Biotechnology & Microbiology 11(4). doi:10.19080/aibm.2018.11.555817.
####   [2] Frutos, P., F. Martínez Peña, P. Ortega Martínez, and S. Esteban. 2009. “Estimating the Social Benefits of Recreational Harvesting of Edible Wild Mushrooms Using Travel Cost Methods.” Forest Systems 18(3):235. doi:10.5424/fs/2009183-01065.
####   [3] Y. Suryani, O. Taupiqurrahman, and Y. Kulsum, Buku Mikologi Dr. Yani Suryani_Lengkap, 2020
#### [4] Wibowo, Agung, Yuri Rahayu, Andi Riyanto, and Taufik Hidayatulloh. 2018.“Classification Algorithm for Edible Mushroom Identification.” 2018 International Conference on Information and Communications Technology,ICOIACT 2018 2018-Janua:250–53. doi:10.1109/ICOIACT.2018.8350746.
#### [5] Özaltun, Betül, and Mustafa Sevindik. 2020. “Evaluation of the Effects on Atherosclerosis and Antioxidant and Antimicrobial Activities of Agaricus Xanthodermus Poisonous Mushroom.” 6(6):539–44. doi: 10.18621/eurj.524149.
