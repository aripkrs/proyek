# Laporan Proyek Machine Learning Mushroom Classification - Arip Kristiyanto

## Domain Proyek

Jamur merupakan komoditas makanan yang memiliki banyak nutrisi yang baik bagi tubuh. Bahkan jamur dapat menjadi suatu obat bagi penyakit tertentu [1]. Jamur memiliki aroma dan rasa yang unik yang berbeda dari bahan makanan lain. Hal tersebut membuat jamur yang dapat dimakan memiliki tingkat permintaan yang tinggi dalam pasar bahan makanan [2]. 
Di Indonesia yang memiliki iklim tropis dan ada banyak kayu yang sangat cocok untuk pertumbuhan jamur, jamur merupakan salah satu tumbuhan liar yang banyak tersebar di alam bebas. Namun, beberapa jamur ini ada yang beracun dan harus dihindari. Jamur yang tersebar di dunia saat ini diperkirakan sedikitnya ada 45.000 jenis dan 2.000 jenis diantaranya tidak beracun. Ada banyak spesies jamur mematikan yang ada di dunia. Ada sekitar 70 hingga 80 spesies yang paling beracun dan mematikan yang bisa berakibat fatal jika dikonsumsi [3].
terdapat beberapa jenis jamur yang beracun, bahkan dapat membahayakan nyawa [4]. Pada saat ini diperkirakan 140.000 spesies jamur diidentifikasi dan 2.000 diantaranya aman untuk dikonsumsi manusia [5]. Dimana 138.000 species lain masih belum bisa ditentukan dapat dimakan atau tidak dan bahkan mengandung racun yang membahayakan tubuh. Dilaporkan menurut Center for Disease Control and Prevention (CDC) lebih dari 41.000 orang meninggal pada tahun 2008 dikarenakan secara tidak sengaja keracunan, sedangkan World Health Organization (WHO) melaporkan sekitar 0,346 juta kematian sejak tahun 2004.
Berdasarkan permasalahan tersebut agar dapat mengidentifikasi jamur yang dapat dimakan atau beracun.

## Business Understanding
Berdasarkan data  Dilaporkan menurut Center for Disease Control and Prevention (CDC) lebih dari 41.000 orang meninggal pada tahun 2008 dikarenakan secara tidak sengaja keracunan, sedangkan World Health Organization (WHO) melaporkan sekitar 0,346 juta kematian sejak tahun 2004. Artinya cukup tinggi kematian yang diakibatkan jamur beracun. Salah satu manfaat dari adanya model klasifikasi jamur beracun atau tidak  ini adalah model ini dapat digunakan untuk mengedukasi masyarakat terutma masyarakat pedesaan.
Pada bagian ini.

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

1. Cap Diameter 
2. Cap Shape
3. Gill Attachment 
4. Gill Color
5. Stem Height
6. Stem Width
7. Stem Color
8. Season
9. Target Class

*** exploratory data analysis ***
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
  - Terdapat 54035 baris data
  - Tedapat 9 kolom
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
```
        cap-diameter	 cap-shape	     gill-attachment	gill-color	    stem-height	    stem-width	    stem-color	    season	        class
count    54035.000000	 54035.000000	 54035.000000	    54035.000000	54035.000000	54035.000000	54035.000000	54035.000000	54035.000000
mean     567.257204	     4.000315	     2.142056	        7.329509	    0.759110	    1051.081299	    8.418062	    0.952163	    0.549181
std	     359.883763	     2.160505	     2.228821	        3.200266	    0.650969	    782.056076	    3.262078	    0.305594	    0.497580
min	     0.000000	     0.000000	     0.000000	        0.000000	    0.000426	    0.000000	    0.000000	    0.027372	    0.000000
25%	     289.000000	     2.000000	     0.000000	        5.000000	    0.270997	    421.000000	    6.000000	    0.888450	    0.000000
50%	     525.000000	     5.000000	     1.000000	        8.000000	    0.593295	    923.000000	    11.000000	    0.943195	    1.000000
75%	     781.000000	     6.000000	     4.000000	        10.000000	    1.054858	    1523.000000	    11.000000	    0.943195	    1.000000
max	     1891.000000	 6.000000	     6.000000	        11.000000	    3.835320	    3569.000000	    12.000000	    1.804273	    1.000000
```
Berdasarkan _output_ tersebut, didapatkan informasi mengenai statistika deskriptif dari _dataset_ yang digunakan. Berikut ini adalah keterangan untuk setiap bagian:
   - ```count``` : Jumlah data dari sebuah kolom
   - ```mean``` : Rata-rata dari sebuah kolom
   - ```std``` : Standar deviasi dari sebuah kolom
   - ```min``` : Nilai terendah pada sebuah kolom
   - ```25%``` : Nilai kuartil pertama (Q1) dari sebuah kolom
   - ```50%``` : Nilai kuartil kedua (Q2) atau median atau nilai tengah dari sebuah kolom
   - ```75%``` : Nilai kuartil ketiha (Q3) dari sebuah kolom
   - ```max``` : Nilai tertinggi pada sebuah kolom

Menjumlah total missing value pada dataset
``` python
df.isnull().sum()
```
Hasinya
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
### Visualisasi data
- _Univariate Analysis
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

 - _Multivariate Analysis_

    _Multivariate Analysis_ adalah prosedur statistik yang digunakan untuk memeriksa hubungan antara beberapa variabel secara bersamaan. Teknik ini mencakup berbagai metode seperti regresi berganda, analisis faktor, dan analisis kluster, yang membantu dalam memahami struktur dan pola yang kompleks dalam data dengan lebih dari satu variabel.
Data Preparation
![Untitled](https://github.com/user-attachments/assets/73496745-4224-4bc0-a499-6a59fa2909c6)
 <div align="center">Gambar 3 - Multivariate Analysis Categorical Column - Every Numeric Column</div>

![Untitled](https://github.com/user-attachments/assets/e89b7cb1-b772-437b-8afc-d7272907244c)
 <div align="center">Gambar 4 - Multivariate Analysis Categorical Column - Numeric Column based on class</div>
 Berdasarkan gambar kedua visualisasi data diatas, dapat terlihat nyaris semua variabel terpisah menunjukkan karakteristik atau pola khusus terhadap variabel label, yaitu 'Class'. 0 dan 1 (ditandai dengan warna oren dan biru)

 - _Correlation_

Uji Korelasi adalah metode statistik yang digunakan untuk menentukan apakah ada hubungan antara dua variabel kuantitatif dan seberapa kuat hubungan tersebut. Uji ini menghasilkan nilai koefisien korelasi, seperti Pearson atau Spearman, yang berkisar antara -1 hingga +1. Nilai mendekati +1 menunjukkan korelasi positif yang kuat, sedangkan nilai mendekati -1 menunjukkan korelasi negatif yang kuat. Nilai mendekati 0 menunjukkan tidak adanya korelasi. Uji korelasi penting dalam menentukan arah dan kekuatan hubungan antar variabel, yang dapat membantu dalam pemodelan prediktif dan analisis penyebab.

   ![Untitled](https://github.com/user-attachments/assets/962667d3-5a75-4f77-8d0d-2b3eaa662eb5)
   <div align="center">Gambar 4 - Multivariate Analysis Categorical Column - Numeric Column based on class</div>
Berdasarkan  visualisasi diatas, terlihat bahwa kolom ```season```, ```gill color```, ```gill attachement```, memiliki skor korelasi yang paling kecil terhadap label. Kolom yang semacam ini baiknya di-drop saja untuk meringankan beban komputasi dan mengurangi dimensi dari dataset yang akan digunakan dalam pelatihan model

- _Missing Value_

    _Missing Values_ adalah data yang hilang atau tidak tercatat dalam dataset. Hal ini bisa terjadi karena berbagai alasan, seperti kesalahan entri data, kerusakan data, atau tidak tersedianya informasi saat pengumpulan data. Missing values dapat mempengaruhi kualitas model _machine learning_ dan hasil analisis statistik. Oleh karena itu, penting untuk mengidentifikasi, menganalisis, dan mengatasi missing values dengan metode seperti imputasi, di mana nilai yang hilang diganti dengan estimasi, atau dengan menghapus baris atau kolom yang terdampak.
   <div align="center">Gambar 5 - Multivariate Analysis Categorical Column - Numeric Column based on class</div>

Berdasarkan gambar diatas tidak ditemukan mising value

# Data Preparation
_Data Preparation_ adalah proses pembersihan, transformasi, dan pengorganisasian data mentah ke dalam format yang dapat dipahami oleh algoritma pembelajaran mesin. Berikut ini adalah **urutan** langkah-langkah Data Preparation yang dilakukan beserta penjelasan dan alasannya:

- _Data Cleaning_
  
  _Data cleaning_ adalah adalah langkah penting dalam proses Machine Learning karena melibatkan identifikasi dan penghapusan data yang hilang, duplikat, atau tidak relevan yang terdapat pada dataset. Proses ini memiliki berbagai langkah yang perlu dilakukan supaya dataset siap digunakan untuk pembangunan model Machine Learning.
    
  **Alasan**: _Data Cleaning_ diperlukan agar data yang digunakan akurat, konsisten, dan bebas kesalahan, karena data yang salah atau tidak konsisten dapat berdampak negatif terhadap performa model Machine Learning
    - _Detection and Removal Duplicates_
      
      Data duplikat adalah baris data yang sama persis untuk setiap variabel yang ada. Dataset yang digunakan perlu diperiksa juga apakah dataset memiliki data yang sama atau data duplikat. Jika ada, maka data tersebut harus ditangani dengan menghapus data duplikat tersebut.

      **Alasan**: Data duplikat perlu didektesi dan dihapus karena jika dibiarkan pada dataset dapat membuat model Anda memiliki bias, sehingga menyebabkan _overfitting_. Dengan kata lain, model memiliki performa akurasi yang baik pada data pelatihan, tetapi buruk pada data baru. Menghapus data duplikat dapat membantu memastikan bahwa model Anda dapat menemukan pola yang ada lebih baik lagi.

      Berikut ini adalah proses pendeteksian dan penghapusan data duplikatnya:
      ```python
      # Cek baris duplikat dalam dataset
      duplicates = df.duplicated()
      
      # Hitung jumlah baris duplikat
      duplicate_count = duplicates.sum()
      
      # Cetak jumlah baris duplikat
      print(f"Number of duplicate rows: {duplicate_count}")

      ```

      Berikut ini adalah hasilnya:

      ```python
        Number of duplicate rows: 303
      ```

Berdasarkan hasil tersebut, ditemukan adanya 303 data duplikat.
selanjutnya 
```python
dfclean=df.drop_duplicates()
```
Berikut ini adalah hasilnya:

      ```python
        Number of duplicate rows: 0
      ```
   
- _Dropping Column with Low Correlation_
      
      Pada bagian ini adalah proses penghapusan fitur-fitur yang memiliki korelasi rendah terhadap variabel target dari dataset. Langkah ini diambil berdasarkan asumsi bahwa fitur dengan korelasi rendah tidak memberikan kontribusi signifikan terhadap prediksi yang dibuat oleh model.
 
      **Alasan**: Tahapan ini perlu dilakukan karena fitur dengan korelasi rendah terhadap variabel target cenderung tidak memberikan informasi yang berguna untuk prediksi dan dapat menambahkan kebisingan yang tidak perlu ke dalam model. Dengan menghilangkan fitur-fitur ini, kita dapat mengurangi kompleksitas model, yang dapat membantu dalam mencegah _overfitting_ dan mempercepat waktu pelatihan. Selain itu, model yang lebih sederhana dengan fitur yang lebih sedikit lebih mudah untuk diinterpretasikan, yang memungkinkan kita untuk lebih memahami bagaimana fitur-fitur tersebut mempengaruhi variabel target. 



Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Rubrik/Kriteria Tambahan (Opsional):

    Menjelaskan proses data preparation yang dilakukan
    Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

Modeling

Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

Rubrik/Kriteria Tambahan (Opsional):

    Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
    Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

Evaluation

Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

Rubrik/Kriteria Tambahan (Opsional):

    Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

---Ini adalah bagian akhir laporan---

Catatan:

####   [1] Temesgen, Tasisa. 2018. “Application of Mushroom as Food and Medicine.” Advances in Biotechnology & Microbiology 11(4). doi:10.19080/aibm.2018.11.555817.
####   [2] Frutos, P., F. Martínez Peña, P. Ortega Martínez, and S. Esteban. 2009. “Estimating the Social Benefits of Recreational Harvesting of Edible Wild Mushrooms Using Travel Cost Methods.” Forest Systems 18(3):235. doi:10.5424/fs/2009183-01065.
####   [3] Y. Suryani, O. Taupiqurrahman, and Y. Kulsum, Buku Mikologi Dr. Yani Suryani_Lengkap, 2020
#### [4] Wibowo, Agung, Yuri Rahayu, Andi Riyanto, and Taufik Hidayatulloh. 2018.“Classification Algorithm for Edible Mushroom Identification.” 2018 International Conference on Information and Communications Technology,ICOIACT 2018 2018-Janua:250–53. doi:10.1109/ICOIACT.2018.8350746.
#### [5] Özaltun, Betül, and Mustafa Sevindik. 2020. “Evaluation of the Effects on Atherosclerosis and Antioxidant and Antimicrobial Activities of Agaricus Xanthodermus Poisonous Mushroom.” 6(6):539–44. doi: 10.18621/eurj.524149.
