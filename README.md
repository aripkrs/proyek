# Laporan Proyek Machine Learning Mushroom Classification - Arip Kristiyanto

## Domain Proyek

Jamur merupakan komoditas makanan yang memiliki banyak nutrisi yang baik bagi tubuh. Bahkan jamur dapat menjadi suatu obat bagi penyakit tertentu [1]. Jamur memiliki aroma dan rasa yang unik yang berbeda dari bahan makanan lain. Hal tersebut membuat jamur yang dapat dimakan memiliki tingkat permintaan yang tinggi dalam pasar bahan makanan [2]. 
Di Indonesia yang memiliki iklim tropis dan ada banyak kayu yang sangat cocok untuk pertumbuhan jamur, jamur merupakan salah satu tumbuhan liar yang banyak tersebar di alam bebas. Namun, beberapa jamur ini ada yang beracun dan harus dihindari. Jamur yang tersebar di dunia saat ini diperkirakan sedikitnya ada 45.000 jenis dan 2.000 jenis diantaranya tidak beracun. Ada banyak spesies jamur mematikan yang ada di dunia. Ada sekitar 70 hingga 80 spesies yang paling beracun dan mematikan yang bisa berakibat fatal jika dikonsumsi [3].
terdapat beberapa jenis jamur yang beracun, bahkan dapat membahayakan nyawa [4]. Pada saat ini diperkirakan 140.000 spesies jamur diidentifikasi dan 2.000 diantaranya aman untuk dikonsumsi manusia [5]. Dimana 138.000 species lain masih belum bisa ditentukan dapat dimakan atau tidak dan bahkan mengandung racun yang membahayakan tubuh. Dilaporkan menurut Center for Disease Control and Prevention (CDC) lebih dari 41.000 orang meninggal pada tahun 2008 dikarenakan secara tidak sengaja keracunan, sedangkan World Health Organization (WHO) melaporkan sekitar 0,346 juta kematian sejak tahun 2004.
Berdasarkan permasalahan tersebut agar dapat mengidentifikasi jamur yang dapat dimakan atau beracun.

## Business Understanding
Berdasarkan data  Dilaporkan menurut Center for Disease Control and Prevention (CDC) lebih dari 41.000 orang meninggal pada tahun 2008 dikarenakan secara tidak sengaja keracunan, sedangkan World Health Organization (WHO) melaporkan sekitar 0,346 juta kematian sejak tahun 2004. Artinya cukup tinggi kematian yang diakibatkan jamur beracun. Salah satu manfaat dari adanya model klasifikasi jamur beracun atau tidak  ini adalah model ini dapat digunakan untuk mengedukasi masyarakat terutma masyarakat pedesaan.
Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.

### Problem Statements


    Berdasarkan eksplorasi terhadap dataset, fitur-fitur apa saja yang dapat menentukan atau memberi pengaruh terhadap klasifikasi layak atau tidaknya air minum?
    Bagaimana memproses dataset agar dapat digunakan untuk pembuatan model machine learning klasifikasi kualitas air minum?
    Bagaimana cara medapatkan model klasifikasi kuaitas air minum dengan performa terbaik?


Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:

    Jawaban pernyataan masalah 1
    Jawaban pernyataan masalah 2
    Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

Rubrik/Kriteria Tambahan (Opsional):

    Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:
    Solution statements

        Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).

Data Understanding

Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: UCI Machine Learning Repository.

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

    accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
    cuisine : merupakan jenis masakan yang disajikan pada restoran.
    dst

Rubrik/Kriteria Tambahan (Opsional):

    Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

Data Preparation

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
