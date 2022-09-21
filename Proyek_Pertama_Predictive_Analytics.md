# LAPORAN PROYEK MACHINE LEARNING - ALFIRA FIBRI NURNINGTYAS
#
#
## Domain Proyek
Saham merupakan suatu gambaran dari kinerja keuangan. Jika harga saham di perusahaan turun hal ini akan menunjukkan suatu informasi bahwa kinerja perusahaan sedang menurun. Sehingga harga saham sangat dipengaruhi oleh kinerja keuangan. Menurut Dermawan Sjahrial, saham adalah surat berharga yang dikeluarkan oleh sebuah perusahaan yang berbentuk perseroan terbatas atau yang disebut emiten. Saham dinyatakan bahwa pemilik saham tersebut juga pemilik sebagian dari sebagian perusahaan itu. Dengan demikian apabila seorang investor membeli saham, maka dia juga menjadi pemilik atau pemegang saham perusahaan.

Netflix yang berbasis di Los Gatos, California mulai beroperasi pada tahun 1997 sebagai suatu layanan DVD-by-mail berlangganan di Amerika Serikat. Namun, layanan inovatif ini tidak berlangsung lama karena tempat penyewaan film Blockbuster akhirnya gulung tikar. Pada tahun 2007, Netflix mulai menawarkan layanan video streaming dengan film dan serial TV berlisensi. Kemudian memasuki bisnis produksi konten dan merilis seri original pertamanya, “House of Cards,” pada Februari 2013. Seiring berjalannya waktu, pertumbuhan pelanggan Netflix mengalami kenaikan yang signifikan. Selama beberapa tahun terakhir, Netflix telah berfokus pada pertumbuhan basis pelanggan globalnya, dengan banyak berinvestasi dalam produksi konten asli berbasis lokal pada negara di seluruh dunia. Akibat kenaikan jumlah penggunanya, hal tersebut juga memengaruhi kinerja saham Netflix. Netflix mengakhiri kuartal pertama dengan 207,64 juta pelanggan di seluruh dunia, di mana AS dan Kanada menyumbang 35,8% dari total basis pelanggannya.

Harga saham Netflix (NASDAQ:NFLX) pada Juli 2022 adalah 189,11 dolar AS. Skor untuk NFLX adalah 50, yaitu 0% di bawah skor rata-rata historisnya yaitu 50, dan menyimpulkan risiko yang lebih tinggi dari biasanya. Dengan harga saham NFLX saat itu, sementara SMA 20-day Netflix adalah 180,58, yang menjadikannya sebagai sinyal beli. Begitupun juga apabila dilihat berdasarkan moving average 50-day Netflix adalah 184,33 sementara harga saham NFLX adalah 189,11, hel tersebut juga dianggap sebagai sinyal beli sebagai secara teknis. Namun, dengan harga saham NFLX hari ini dan menggunakan simple moving average 200-day Netflix adalah 417,38, yang menjadikannya sebagai salah satu sinyal jual.

Berdasarkan pandangan 39 analis yang dikumpulkan oleh MarketBeat, untuk prediksi harga saham Netflix mereka adalah bahwa saham bisa naik hampir 74 persen menjadi 329 dolar AS selama 12 bulan ke depan. Sedangkan berdasarkan prediksi harga jangka panjang terbaru yang dilakukan oleh CoinPriceForecast, harga Netflix akan mencapai 300 dolar AS pada akhir tahun 2022. Kemudian mencapai 500 dolar AS pada akhir tahun 2023. Secara jangka panjang, Netflix akan naik menjadi 700 dolar AS pada tahun 2025, 800 dolar AS pada tahun 2026, dan 1.100 dolar AS pada tahun 2029.

Menurut prediksi saham Netflix berbasis algoritma dari Wallet Investor, saham tersebut dianggap sebagai “investasi jangka panjang yang sangat baik”. Karena diprediksi harga dapat naik sebanyak 22,78 persen menjadi 232,104 dolar AS di tahun mendatang. Perkiraan stok Netflix situs untuk tahun 2025, diprediksikan harga bisa mencapai 372,54 dolar AS.

Namun perlu diingat bahwa bahwa prediksi harga juga bisa salah. Prediksi harga sebaiknya tidak digunakan sebagai informasi dasar dari strategi investasi kamu sendiri. Ingatlah bahwa keputusan kamu untuk trading atau berinvestasi harus bergantung pada toleransi risiko, keahlian di pasar, ukuran portofolio, dan tujuan investasi yang kamu miliki.


## *Business Understanding*
#### 1. *Problem Statements*
Berdasarkan latar belakang di atas, maka dapat dirumuskan permasalahan sebagai berikut:
- Bagaimana cara menganalisis data harga saham Netflix?
- Bagaimana cara mengolah data saham Netflix agar dapat dilatih dengan baik oleh model?
- Bagaimana cara membangun model yang dapat memprediksi time series forecasting dengan tingkat akurasi yang baik?

2. Goals

Penelitian ini dilakukan dengan tujuan sebagai berikut:
- Dapat memprediksi harga saham Netflix dengan menggunakan model machine learning
- Dapat mengolah data dengan optimal agar dapat dilatih dengan baik oleh model machine learning
- Dapat menemukan model yang dapat memprediksi time series forecasting dengan tingkat akurasi yang baik

3. Solution Statements

Dari rumusan masalah dan tujuan di atas, maka solusi yang dapat dilakukan adalah sebagai berikut:
- Melakukan analisis dengan cara menangani missing value pada data, mencari korelasi pada data, menangani outlier pada data, dan melakukan normalisasi pada data. Selain itu juga dapat melakukan eksplorasi dan pemrosesan pada data dengan memvisualisasikannya.
- Membuat model regresi untuk memprediksi harga yang akan datang. Dalam proyek ini akan menggunakan algoritma Support Vector Regression (SVR), K-Nearest Neighbor (KNN), dan Gradient Boosting Regression.
- Melakukan hyperparameter tuning agar model dapat berjalan pada performa yang terbaik. 


## Data Understanding
Dataset yang digunakan pada proyek ini didapatkan dari website kaggle.com. Untuk mengarah pada dataset tersebut dapat mengunjungi link berikut https://www.kaggle.com/datasets/meetnagadia/netflix-stock-price-data-set-20022022. Dataset tersebut memiliki format .csv dan mempunyai total 4196 records dan 7 columns. Kolom-kolom tersebut diantaranya yaitu :
- Date merupakan hari dan tanggal dimana data tersebut didapatkan.
- Open merupakan harga di mana keamanan finansial terbuka di pasar saat perdagangan dimulai.
- High merupakan harga tertinggi di mana suatu saham diperdagangkan selama suatu periode.
- Low merupakan harga terendah di mana suatu saham diperdagangkan selama suatu periode.
- Close merupakan harga penutupan yang mana umumnya mengacu pada harga terakhir di mana perdagangan saham selama sesi perdagangan reguler.
- Adj close merupakan harga penutupan yang disesuaikan harga penutupan saham untuk mencerminkan nilai saham tersebut setelah akuntansi, seperti right issue, stock split, dan stock reverse.
- Volume digunakan untuk mengukur jumlah saham yang diperdagangkan dalam saham atau sebuah kontrak yang diperdagangkan di futures atau opsi.

## Data Preparation
Teknik data preparation yang dilakukan pada proyek kali ini adalah sebagai berikut:
1. Melakukan penghapusan fitur yang tidak diperlukan dengan menggunakan fungsi drop. Alasan melakukan hal tersebut karena pada proyek kali ini tidak memerlukan fitur time dan volume.
2. Melakukan proses splitting pada dataset yaitu dengan membagi dataset menjadi 2, yaitu train dan test data. Train data digunakan sebagai training model, sedangkan test data digunakan sebagai validasi apakah model yang digunakan sudah akurat atau belum. Dalam proyek kali ini dataset dibagi sesua dengan proporsi yang umum digunakan yaitu 80:20, 80% sebagai train data dan 20% sebagai test data.
3. Melakukan data normalization yang bertujuan agar model dapat bekerja dengan lebih optimal. Normalisasi akan melakukan proses transformasi data dalam skala tertentu. Pada proyek ini, data akan dinormalisasi pada skala 0 hingga 1, yaitu X_train dan X_test dengan menggunakan library MinMaxScaler.

## Modelling
Model yang digunakan pada proyek kali ini adalah Support Vector Regression (SVR), K-Nearest Neighbor (KNN), dan Gradient Boosting Regression.
1. Support Vector Regression (SVR)

Support Vector Regression merupakan hasil dari modifikasi dari metode Support Vector Machine yang dipergunakan untuk menyelesaikan masalah prediksi. Pada metode SVM adalah penerapan dari teori machine learning yang menghasilkan nilai bulat, sedangkan pada algoritme Support Vector Regression (SVR) yaitu untuk penerapan kasus regresi yang menghasilkan keluaran berupa bilangan riil. Konsep algoritme SVR dapat menghasilkan nilai prediksi yang bagus karena SVR mempunyai kemampuan menyelesaiakan masalah overfitting. Keuntungan menggunakan SVR yaitu SVR sangat cocok untuk data set berdimensi tinggi dan sangat cocok digunakan untuk kasus non linier dengan menggunakan fungsi Kernel. Selain itu, algoritma SVR menghasilkan trend data yang bergelombang mengikuti jalur data yang terbentuk, sehingga prediksi data yang dihasilkan lebih akurat.
Untuk hyperparameter yang digunakan pada model ini adalah sebagai berikut:
- kernel : hyperparameter ini digunakan untuk menghitung kernel matriks sebelumnya.
- C : hyperparameter ini adalah parameter regularisasi digunakan untuk menukar klasifikasi yang benar dari contoh training terhadap maksimalisasi margin fungsi keputusan.
- gamma : hyperparameter ini digunakan untuk menentukan seberapa jauh pengaruh contoh pelatihan, dimana jika nilainya rendah berarti jauh dan jika nilainya tinggi berarti dekat. Untuk nilai setiap hyperparameter disetiap algoritma adalah sebagai berikut : kernel : ['rbf'] C : [0.001, 0.01, 0.1, 10, 100, 1000] gamma : [0.3, 0.03, 0.003, 0.0003]

2. K-Nearest Neighbors (KNN)

Algoritma K-Nearest Neighbor (KNN) adalah sebuah metode untuk melakukan klasifikasi
terhadap objek yang berdasarkan dari data pembelajaran yang jaraknya paling dekat dengan
objek tersebut. KNN merupakan algoritma supervised learning dimana hasil dari query
instance yang baru diklasifikan berdasarkan mayoritas dari kategori pada algoritma KNN. Algoritma ini dapat digunakan untuk klasifikasi dan regresi.
KNN memiliki kelebihan yaitu tangguh terhadap training data yang noisy dan efektif apabila data latih nya besar. Sedangkan kekurangannya adalah tidak berfungsi dengan baik pada dataset berukuran besar, kurang cocok untuk data dengan dimensi yang tinggi, serta sensitif terhadap noise data, missing values dan outliers.
Untuk hyperparameter yang digunakan pada model ini hanya 1 yaitu n_neighbors, hyperparameter ini adalah jumlah tetangga yang diperlukan untuk menentukan letak data baru. Dimana n_neighbors memiliki nilai sebesar 1 dan 10.

3. Gradient Boosting Regression

Gradient Boosting Regression termasuk supervised  learning berbasis decision  tree yang  dapat digunakan   untuk   klasifikasi. Algoritma ini bekerja   secara   sekuensial menambahkan  prediktor  sebelumnya  yang  kurang  cocok  dengan  prediksi  ke ensemble, memastikan   kesalahan   yang   dibuat   sebelumnya   diperbaiki. Penggambaran   sederhana konsep ensemble adalah    keputusan-keputusan    dari    berbagai    mesin    pembelajaran digabungkan, kemudian untuk kelas yang menerima mayoritas ‘suara’ adalah kelas yang akan    diprediksi    oleh    keseluruhan ensemble. 
Untuk hyperparameter yang digunakan pada model ini adalah sebagai berikut:
- criterion : hyperparameter yang digunakan untuk menemukan fitur dan ambang batas optimal dalam membagi data.
- learning_rate : hyperparameter training yang digunakan untuk menghitung nilai koreksi bobot padad waktu proses training. Umumnya nilai learning rate berkisar antara 0 hingga 1.
- n_estimators : jumlah tahapan boosting yang akan dilakukan. Untuk nilai setiap hyperparameter disetiap algoritma adalah sebagai berikut : criterion : 'squared_error' learning_rate : 0.01 n_estimators : 1000

 ## Evaluation
Evaluasi pada proyek ini adalah dengan menggunakan mse (mean squared error), dimana metrik tersebut digunakan untuk mengukur seberapa dekat garis pas dengan titik data.

Dari hasil perbandingan tiga model yang digunakan, didapatkan bahwa model Support Vector Regression (SVR) menghasilkan performa yang lebih baik jika dibandingkan dengan K-Nearest Neighbors (KNN) dan Gradient Boosting Regression. Sehingga, model tersebut dapat membantu para investor dalam melakukan investasi serta trader yang dapat memprediksi saham netflix. 

![image](https://user-images.githubusercontent.com/100407187/191460636-65bc5a00-e761-47a9-b5ba-8195bfe0a7f4.png)


Berdasarkan tabel hasil mse di atas dapat dilihat bahwa algoritma SVR memiliki nilai mse (mean squared error) pada data train sebesar 2.688962	dan pada data test sebesar	2.823519. Kemudian pada algoritma KNN memiliki nilai mse (mean squared error) pada data train sebesar 0.651909 dan pada data test sebesar 	1.368729. Dan pada algoritma GradientBoosting memiliki nilai mse (mean squared error) pada data train sebesar 0.000865 dan pada data test sebesar 0.210018. Sehingga, dapat disimpulkan bahwa pada proyek kali ini penggunaan model SVR menghasilkan performa yang optimal. Berikut adalah hasil visualisasi mse (mean squared error).

![image](https://user-images.githubusercontent.com/100407187/191460387-d3651411-70c2-4a9f-987b-36a6369afd9a.png)



## References
- PT Exchange Indonesia, Zipmex. (2022). Pahami Saham Netflix Sebelum Kamu Membelinya. https://zipmex.com/id/learn/pahami-saham-netflix-sebelum-anda-membelinya/
- Al Hakim, H. A., & Fudholi, D. H. (2021). Perbandingan Penggunaan Algoritma Machine Learning pada Prediksi Tren Harga Saham Netflix. AUTOMATA, 2(2).
- Cahyono, R. E., Sugiono, J. P., & Tjandra, S. (2019). Analisis Kinerja Metode Support Vector Regression (SVR) dalam Memprediksi Indeks Harga Konsumen. JTIM: Jurnal Teknologi Informasi dan Multimedia, 1(2), 106-116.
- Maulana, N. D. (2018). Implementasi Metode Support Vector Regression (SVR) Dalam Peramalan Penjualan Roti (Studi Kasus: Harum Bakery) (Doctoral dissertation, Universitas Brawijaya).
- Yanosma, D., Johar, A., & Anggriani, K. (2016). Implementasi Metode K-Nearest Neighbor (KNN) dan Simple Addittive Weighting (SAW) dalam Pengambilan Keputusan Seleksi Anggota PASKIBRAKA. Rekursif: Jurnal Informatika, 4(2).
- Suryana, S. E., Warsito, B., & Suparti, S. (2021). Penerapan Gradient Boosting Dengan Hyperopt Untuk Memprediksi Keberhasilan Telemarketing Bank. Jurnal Gaussian, 10(4), 617-623.


