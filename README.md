# Laporan Proyek Machine Learning - Biki Nurul Af'ida

## Domain Proyek

Gaya hidup seperti pola makan, aktivitas fisik, merokok, dan konsumsi alkohol berperan penting dalam menentukan kesehatan dan risiko penyakit kronis (World Health Organization, 2021). Aktivitas fisik yang cukup dapat mengurangi risiko penyakit jantung, diabetes tipe 2, dan kanker (Lee et al., 2012). Selain itu, indeks massa tubuh (BMI), kebiasaan merokok, dan konsumsi alkohol juga berkontribusi signifikan terhadap masalah kesehatan dan kematian dini (Mokdad et al., 2018; Wang et al., 2023). Durasi tidur yang sehat juga penting karena berkaitan dengan risiko obesitas, diabetes, dan penyakit jantung (Buxton & Marcelli, 2010).

Skor kesehatan (Health_Score) adalah nilai yang menggambarkan kondisi kesehatan seseorang secara keseluruhan berdasarkan berbagai faktor gaya hidup tersebut. Namun, hubungan yang kompleks antar faktor-faktor ini membuat analisis dengan cara biasa menjadi kurang efektif. Oleh karena itu, pemanfaatan machine learning untuk memprediksi skor kesehatan berdasarkan data gaya hidup dapat memberikan informasi yang lebih tepat untuk memantau kesehatan dan mencegah penyakit. 

Proyek ini menggunakan dataset Health and Lifestyle Data for Regression yang diambil dari Kaggle (https://www.kaggle.com/datasets/pratikyuvrajchougule/health-and-lifestyle-data-for-regression) dan menerapkan beberapa algoritma regresi untuk memodelkan hubungan antara fitur gaya hidup dan skor kesehatan. Dengan pendekatan ini, diharapkan dihasilkan model prediksi yang akurat dan berguna bagi individu maupun tenaga kesehatan dalam mendorong gaya hidup sehat secara lebih efektif.

## Business Understanding
### Problem Statements

Seiring dengan meningkatnya prevalensi penyakit tidak menular (NCDs) di masyarakat, kebutuhan akan pemantauan kesehatan individu yang efektif dan akurat menjadi semakin penting. Dalam konteks ini, proyek ini berupaya mengembangkan sistem prediksi health score yang dapat memberikan solusi atas beberapa permasalahan berikut:
- Dari sekumpulan fitur gaya hidup dan kondisi kesehatan yang tersedia, fitur mana saja yang memiliki pengaruh paling signifikan terhadap skor kesehatan seseorang? Pemahaman terhadap faktor-faktor kunci ini sangat penting untuk mengidentifikasi risiko utama dan membantu individu maupun tenaga kesehatan dalam memfokuskan upaya pencegahan dan perawatan secara lebih efektif.
- Bagaimana cara memprediksi dengan akurat nilai health score berdasarkan data gaya hidup dan kondisi kesehatan individu? Prediksi yang tepat dapat mendukung pengambilan keputusan dini dan merancang intervensi kesehatan yang lebih terarah guna mengurangi risiko penyakit tidak menular secara signifikan.

Dengan menjawab permasalahan tersebut, diharapkan model prediksi health score yang dikembangkan dapat menjadi alat bantu yang berguna dalam meningkatkan kualitas hidup dan mendorong penerapan gaya hidup sehat di masyarakat.

### Goals

Untuk menjawab permasalahan yang telah diidentifikasi, proyek ini menetapkan tujuan utama sebagai berikut:
- Mengidentifikasi fitur-fitur gaya hidup dan kondisi kesehatan yang memiliki korelasi serta pengaruh paling signifikan terhadap health score. Pemahaman ini penting untuk mengenali faktor risiko utama yang perlu diperhatikan dalam upaya peningkatan kesehatan dan pencegahan penyakit.
- Membangun model machine learning yang mampu memprediksi nilai health score dengan akurasi tinggi dan tingkat kesalahan minimal. Model ini diharapkan memberikan prediksi yang andal sehingga dapat menjadi dasar pengambilan keputusan bagi individu maupun tenaga kesehatan dalam merancang intervensi pencegahan dan perbaikan gaya hidup secara efektif.

### Solution statements
Untuk mencapai tujuan proyek prediksi nilai health score, solusi yang akan diterapkan mencakup beberapa langkah berikut:
- Eksplorasi Data (Exploratory Data Analysis - EDA)
Melakukan analisis mendalam terhadap dataset guna memahami karakteristik fitur, distribusi fitur, serta hubungan antar fitur dengan target health score. EDA juga berfungsi untuk mendeteksi dan menangani data tidak konsisten, outlier, dan missing value agar data siap digunakan dalam proses pemodelan.
- Penggunaan Berbagai Algoritma Machine Learning
Menerapkan beberapa algoritma regresi untuk membangun model prediksi health score. Pendekatan ini memungkinkan perbandingan performa dan pemilihan model terbaik sesuai dengan karakteristik data dan tujuan kesehatan. Algoritma yang digunakan meliputi:

    a. Support Vector Regression (SVR)
        SVR menggunakan prinsip Support Vector Machine yang bertujuan mencari fungsi regresi yang memiliki margin toleransi kesalahan (ε). Dengan kernel RBF, SVR dapat memodelkan hubungan non-linear antara fitur dan target.
    b. Random Forest Regression
        Algoritma ensemble yang kuat dalam menangani hubungan non-linear dan interaksi fitur, sekaligus mengurangi risiko overfitting.
    c. Extreme Gradient Boosting Regression (XGBoost Regression)
        Algoritma boosting yang melakukan iterasi dengan tujuan memperbaiki kesalahan model sebelumnya. Dikenal sangat efisien dan akurat untuk dataset kompleks serta mendukung regularisasi yang efektif.
- Evaluasi Model
Model-model yang dibangun akan dievaluasi menggunakan metrik regresi berikut:
a. Mean Absolute Error (MAE) — rata-rata selisih absolut antara prediksi dan nilai sebenarnya.
b. Mean Squared Error (MSE) — rata-rata kuadrat selisih yang memberi penalti lebih besar untuk kesalahan besar.
c. Root Mean Squared Error (RMSE) — akar kuadrat dari MSE, dalam satuan yang sama dengan target.
d. R-squared (R²) — proporsi variansi target yang dapat dijelaskan oleh model.
- Pemilihan Model Terbaik
Melalui perbandingan hasil evaluasi dari ketiga model, akan dipilih model dengan performa terbaik sebagai solusi utama. Model tersebut akan digunakan sebagai dasar dalam implementasi sistem prediksi health score untuk mendukung monitoring kesehatan dan intervensi pencegahan.
- Penggunaan Library Scikit-learn dan XGBoost
Untuk mempermudah pelatihan dan evaluasi model, library Scikit-learn (untuk Random Forest dan SVR) serta XGBoost akan digunakan. Library ini menyediakan fungsi-fungsi yang mendukung workflow pemodelan secara efisien.

Dengan metode ini, diharapkan solusi yang diterapkan mampu mencapai tujuan proyek dalam menciptakan model prediksi health score yang akurat dan andal, sekaligus memberikan wawasan yang bermanfaat untuk pengambilan keputusan di bidang kesehatan.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Health and Lifestyle Data for Regression, yang diperoleh dari Kaggle (https://www.kaggle.com/datasets/pratikyuvrajchougule/health-and-lifestyle-data-for-regression). Dataset ini terdiri dari data numerik dan kategorikal yang menggambarkan berbagai aspek gaya hidup dan kesehatan individu, dengan total 8 fitur termasuk 1 fitur target yaitu Health_Score. Dataset terdiri dari 1000 baris dan 8 kolom berdasarkan output data.shape. Pada dataset tidak terdapat missing values dan duplikasi data.

### Fitur-fitur pada dataset adalah sebagai berikut:
- Age: Usia individu dalam tahun (fitur kontinu).
- BMI: Body Mass Index individu (fitur kontinu).
- Exercise_Frequency: Jumlah hari dalam seminggu individu berolahraga (fitur kategorikal dengan nilai 0-7).
- Diet_Quality: Indeks kualitas diet, semakin tinggi nilai menunjukkan pola makan yang lebih sehat (fitur kontinu, rentang 0-100).
- Sleep_Hours: Rata-rata jam tidur per malam (fitur kontinu).
- Smoking_Status: Status merokok (fitur biner, 0 = bukan perokok, 1 = perokok).
- Alcohol_Consumption: Rata-rata konsumsi alkohol dalam satuan unit per minggu (fitur kontinu).
- Health_Score: Skor kesehatan yang dihitung sebagai indikator status kesehatan keseluruhan (fitur kontinu, rentang 0-100).

## Exploratory Data Analysis
- Melakukan pengecekan missing value dan data duplikat. Dataset tidak memiliki nilai kosong maupun data duplikat, sehingga tidak diperlukan penanganan khusus untuk hal tersebut.
- Analisis outlier dilakukan menggunakan boxplot dan metode Interquartile Range (IQR) untuk mengidentifikasi nilai ekstrim pada fitur numerik. 
- Visualisasi fitur kategorik untuk melihat proporsi data pada tiap kategori.
- Visualisasi distribusi fitur numerik menggunakan histogram untuk memahami pola sebaran data (normal, skewed, atau lainnya).
- Analisis korelasi antar fitur numerik menggunakan matriks korelasi dan visualisasi heatmap untuk mengidentifikasi hubungan signifikan terhadap Health_Score.

## Visualisasi proses Data Understanding
- Outlier yang ditemukan pada tahap eksplorasi data kemudian ditangani menggunakan metode winsorizing, yaitu mengganti nilai ekstrim dengan batas nilai kuartil berdasarkan metode Interquartile Range (IQR). Metode IQR sebagai berikut:

    a.  IQR merupakan ukuran statistik yang menggambarkan rentang atau sebaran data pada bagian tengah distribusi. IQR dihitung dengan mengurangi nilai kuartil ketiga (Q3) dengan nilai kuartil pertama (Q1). Outlier dianggap sebagai nilai yang berada di luar rentang IQR yang ditentukan.

    b. Cara Kerja Metode IQR:
        - Hitung Q1 (kuartil pertama) dan Q3 (kuartil ketiga) dari data.
        - Hitung IQR dengan mengurangi Q1 dari Q3. Tentukan batas atas dan batas bawah untuk outlier dengan menggunakan rumus: batas atas = Q3 + (1.5 * IQR), batas bawah = Q1 - (1.5 * IQR).
        - Data yang berada di luar batas atas dan batas bawah tersebut dianggap sebagai outlier.

    c. Penanganan outlier dengan winsorizing
    Nilai yang berada di bawah Q1 akan diganti dengan nilai Q1, dan nilai yang berada di atas Q3 akan diganti dengan nilai Q3.

- Visualisasi Boxplot

    a. Boxplot adalah visualisasi yang memudahkan analisis distribusi data dan identifikasi outlier. Boxplot menampilkan Q1, Q2 (median), Q3, serta batas atas dan batas bawah untuk outlier.

    b. Cara kerja Boxplot:
    - Sebuah kotak (box) menunjukkan rentang IQR (dari Q1 sampai Q3).
    - Garis di dalam kotak menunjukkan posisi Q2 (median).
    - Whisker atau garis lurus yang terhubung dengan kotak menunjukkan rentang data yang dianggap tidak sebagai outlier.
    - Titik-titik di luar whisker menandakan data outlier.

Berikut Visualisasi Boxplot Fitur pada Dataset

![boxplot age](https://github.com/user-attachments/assets/38d16d1b-6082-4d23-82c0-24cd4d698a6c)

![boxplot bmi](https://github.com/user-attachments/assets/f5c51902-7110-4363-982a-c3f12faa7fd2)

![boxplot alcohol consumption](https://github.com/user-attachments/assets/5d1417f5-d4ed-4722-ab2f-c3c14a29da60)

![boxplot diet quality](https://github.com/user-attachments/assets/39c2d24f-1403-49fe-844b-d70930a1ee2a)

![boxplot exercise freq](https://github.com/user-attachments/assets/be676652-63ea-41d1-a5d8-7ef411895188)

![boxplot sleep hours](https://github.com/user-attachments/assets/6c6d4f9b-ff0c-4a0a-92dd-eeb42b284533)

![boxplot health score](https://github.com/user-attachments/assets/00930c4c-1ca8-4fe9-b28f-a163d40a8397)

Dari visualisasi boxplot tersebut dapat terlihat bahwa terdapat beberapa fitur yang memiliki outliers, yaitu Age (8 outliers), BMI (8 outliers), Alcohol Consumption (5 outliers), Diet Quality (3 outliers), Sleep Hours (6 outliers), dan Health Score (3 outliers).

- Univariate Analysis
  
    a. Categorical Feature
  
  ![barchart smoking](https://github.com/user-attachments/assets/6d1f4309-9b9d-4503-95c3-03e543e7005b)

    Terlihat bahwa jumlah sampel yang tidak merokok adalah sebanyak 501 orang, sedangkan yang merokok sebanyak 499 orang.

    b. Histogram Numerical Feature
  
  ![Histogram Feature Numeric](https://github.com/user-attachments/assets/94e89888-4379-4d8d-8fa2-b57a6ced151b)

    Histogram fitur numerik ini menggambarkan distribusi berbagai fitur kesehatan dan kebiasaan pada sampel data yang cukup beragam. Usia responden tersebar secara normal dengan sebagian besar berada pada rentang 20 hingga 60 tahun, menunjukkan keragaman kelompok umur. Indeks massa tubuh (BMI) juga mengikuti pola distribusi normal dengan nilai mayoritas di kisaran 18 hingga 35, mencerminkan variasi status berat badan dari yang kurus hingga kelebihan berat badan. Frekuensi olahraga merupakan fitur diskrit dengan distribusi yang relatif merata dari nol hingga enam hari per minggu, menunjukkan perbedaan kebiasaan aktivitas fisik di antara responden. Kualitas diet dan jam tidur memiliki pola distribusi normal dengan puncak pada kualitas diet sedang hingga baik serta jam tidur ideal antara enam sampai delapan jam. Konsumsi alkohol menunjukkan variasi dengan mayoritas pada tingkat konsumsi sedang, meskipun data ini mungkin mengalami transformasi nilai. Sementara itu, skor kesehatan menunjukkan pola distribusi yang tidak normal, dengan banyak responden mengisi nilai maksimal, yang dapat menunjukkan banyaknya individu dengan kondisi kesehatan sangat baik atau adanya batasan skor. Secara keseluruhan, data ini memberikan gambaran populasi yang sehat dengan variasi wajar dalam kebiasaan dan kondisi fisik, namun perlu perhatian pada distribusi skor kesehatan yang terpusat pada nilai maksimum.

- Multivariate Analysis

    a. Categorical Feature
![Average Health Score relative to Smoking Status](https://github.com/user-attachments/assets/ba323ff9-eb59-459d-b142-71fc63c62cec)

    Histogram menunjukkan bahwa rata-rata skor kesehatan pada kelompok non-perokok (Smoking_Status = 0) sebesar 86,96, sedikit lebih tinggi dibandingkan dengan kelompok perokok (Smoking_Status = 1) yang memiliki rata-rata skor kesehatan 83,99. Perbedaan ini mengindikasikan bahwa status merokok berpengaruh negatif terhadap kesehatan, di mana perokok cenderung memiliki kondisi kesehatan yang lebih rendah.

    b. Numerical Feature
  ![pair plot](https://github.com/user-attachments/assets/d1d7aa62-b3a4-4745-bdc1-65d3fef6b920)

    Pairplot ini memperlihatkan hubungan antar fitur numerik dalam data kesehatan. Skor kesehatan (Health Score) memiliki korelasi positif dengan kualitas diet (Diet Quality), frekuensi olahraga (Exercise Frequency), dan jam tidur (Sleep Hours), yang menunjukkan bahwa gaya hidup sehat berkontribusi meningkatkan kesehatan. Sebaliknya, usia (Age) dan indeks massa tubuh (BMI) menunjukkan korelasi negatif dengan skor kesehatan, yang berarti peningkatan usia dan BMI berpotensi menurunkan kondisi kesehatan. Distribusi frekuensi olahraga berupa hitungan jumlah olahraga dalam satu minggu dan tersebar merata. Secara keseluruhan, pola ini menegaskan pentingnya gaya hidup sehat dalam menunjang skor kesehatan responden.

- Matriks Korelasi
  ![matriks korelasi](https://github.com/user-attachments/assets/6400565f-bda5-4cf0-9d5f-c8bbab1b1669)

    Matriks korelasi pada gambar menunjukkan hubungan antar fitur numerik dalam data kesehatan. Hasil analisis memperlihatkan bahwa kualitas diet memiliki korelasi positif paling kuat dengan skor kesehatan (0,68), menandakan bahwa pola makan yang baik sangat berkontribusi terhadap peningkatan kesehatan. Selain itu, jam tidur (0,27) dan frekuensi olahraga (0,25) juga menunjukkan korelasi positif sedang dengan skor kesehatan, yang mengindikasikan pentingnya tidur cukup dan aktivitas fisik rutin dalam menunjang kesehatan. Sebaliknya, indeks massa tubuh (BMI) memiliki korelasi negatif yang cukup signifikan (-0,42) dengan skor kesehatan, serta usia juga menunjukkan korelasi negatif lemah (-0,19), menunjukkan bahwa peningkatan berat badan dan bertambahnya usia berpotensi menurunkan kondisi kesehatan. Konsumsi alkohol menunjukkan korelasi negatif ringan (-0,14), sementara korelasi antar fitur lain relatif kecil, menandakan fitur-fitur tersebut relatif independen satu sama lain dalam kaitannya dengan skor kesehatan.

Berdasarkan analisis korelasi dan visualisasi multivariat, fitur-fitur gaya hidup seperti kualitas diet, frekuensi olahraga, dan jam tidur menunjukkan pengaruh positif terhadap skor kesehatan, sedangkan BMI dan usia memberikan pengaruh negatif yang signifikan.

## Data Preparation
Berikut tahapan data preparation yang dilakukan pada proyek ini:
- Pada dataset, fitur dengan tipe kategorik telah diubah ke tipe numerik sehingga tidak diperlukan tahapan encoding.
- Splitting data menjadi data training dan data testing. Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk menguji seberapa baik model yang telah dilatih dapat melakukan prediksi pada data yang belum pernah dilihat sebelumnya. Pada proyek ini, dataset dibagi menjadi 80:20, yaitu 80% untuk data training dan 20% untuk data testing. Pembagian tersebut merupakan pembagian yang umum digunakan untuk memberikan keseimbangan antara jumlah data yang cukup untuk melatih model dan jumlah data yang cukup untuk menguji performa model. Namun, tidak ada aturan khusus dalam pembagian data ini, bergantung pada jenis data yang digunakan dan juga sesuai kebutuhan analisis.
- Standarisasi fitur numerik. Standarisasi merupakan proses mengubah skala data agar memiliki rata-rata (mean) = 0 dan simpangan baku (standar deviasi) = 1. Tujuannya adalah membantu model bekerja lebih optimal.

## Modeling
Setiap model regresi yang digunakan, dilakukan pencarian hyperparameter terbaik menggunakan metode GridSearchCV. GridSearchCV dipilih untuk mencari kombinasi hyperparameter terbaik guna mengoptimalkan performa model berdasarkan metrik Mean Squared Error (MSE). Metode ini menggunakan 3-fold cross-validation dan memanfaatkan semua core CPU untuk mempercepat proses pencarian parameter.
a. Support Vector Regression
 SVR merupakan perluasan dari algoritma Support Vector Machine (SVM) yang diaplikasikan pada masalah regresi. Tujuan SVR adalah membangun sebuah fungsi regresi, berupa hyperplane, yang sesuai dengan data input dengan memperbolehkan adanya error dalam batas toleransi seminimal mungkin. Fungsi ini dioptimalkan agar dapat memprediksi output secara akurat sambil menjaga margin kesalahan seminimal mungkin. SVR bekerja dengan mencari hyperplane terbaik yang memaksimalkan margin toleransi antara prediksi dan nilai aktual, serta mengizinkan sebagian data berada di luar margin untuk mengakomodasi noise dan outlier. Dengan menggunakan fungsi kernel, SVR dapat memodelkan hubungan non-linear antara fitur dan target dengan memetakan data ke ruang berdimensi lebih tinggi.

Kelebihan:
- Mampu menangani data dengan noise dan pencilan (outlier) secara efektif.
- Memiliki fleksibilitas tinggi melalui pilihan kernel untuk hubungan non-linear.
- Umumnya memberikan generalisasi yang baik pada data baru.

Kekurangan:
- Penyesuaian parameter (seperti C, epsilon, dan kernel) cukup rumit dan penting untuk performa optimal.
- Waktu pelatihan relatif lama untuk dataset besar.
- Interpretasi model kurang intuitif.

b. Random Forest Regression
Random Forest Regression adalah algoritma ensemble learning yang menggunakan banyak pohon keputusan (decision trees) untuk tugas regresi. Model ini membangun sejumlah pohon secara acak dari subset data dan fitur, kemudian menggabungkan hasil prediksi setiap pohon dengan cara averaging untuk menghasilkan prediksi akhir. Random Forest bekerja dengan membuat banyak pohon keputusan pada data training yang berbeda (bootstrap sampling) dan dengan subset fitur yang dipilih secara acak pada setiap split pohon. Pendekatan ini membantu mengurangi overfitting dan meningkatkan akurasi prediksi dengan menggabungkan kekuatan banyak model sederhana.

Kelebihan:
- Sangat efektif dalam menangani dataset dengan fitur yang banyak dan kompleks.
- Dapat menangani data non-linear dan interaksi antar fitur dengan baik.
- Lebih tahan terhadap overfitting dibandingkan pohon keputusan tunggal.
- Tidak memerlukan banyak pra-pemrosesan data, seperti normalisasi.

Kekurangan:
- Model bisa menjadi kurang interpretatif karena banyaknya pohon yang digunakan.
- Relatif lambat saat pelatihan dan prediksi pada dataset yang sangat besar.
- Prediksi cenderung kurang halus dibandingkan model boosting.

c. Extreme Gradient Boosting Regression (XGBoost Regression)
XGBoost Regression adalah algoritma boosting yang menggabungkan banyak model pohon keputusan secara iteratif, dengan tujuan memperbaiki kesalahan prediksi model sebelumnya. Setiap pohon baru berfokus pada memperkecil residual (selisih antara prediksi dan nilai aktual) dari model sebelumnya. XGBoost mengoptimalkan fungsi objektif menggunakan metode gradient boosting dan menambahkan regularisasi untuk menghindari overfitting. Algoritma ini terkenal karena kecepatan, efisiensi komputasi, dan akurasi tinggi, terutama pada dataset besar dan kompleks.

Kelebihan:
- Memiliki kemampuan prediksi yang sangat baik karena mengoreksi kesalahan secara bertahap.
- Mampu menangani data dengan fitur non-linear dan interaksi kompleks.
- Dilengkapi regularisasi untuk mencegah overfitting.
- Implementasi yang dioptimalkan secara efisien sehingga cepat dan hemat memori.

Kekurangan:
- Proses tuning hyperparameter cukup kompleks dan membutuhkan waktu.
- Model relatif sulit diinterpretasikan secara langsung.
- Performa bisa menurun jika data sangat kecil atau terlalu sederhana.

### Tahapan yang dilakukan
1. Support Vector Regression (SVR)
    - Mendefinisikan model dasar SVR dari sklearn.svm.
    - Menyiapkan parameter grid tuning meliputi: 

        a. kernel: tipe kernel yang digunakan, yaitu 'rbf' dan 'linear'.

        b. C: parameter regularisasi yang mengontrol trade-off antara kesalahan training dan margin (nilai diuji [0.1, 1, 10, 100]).

        c. gamma: parameter kernel hanya untuk 'rbf' ('scale' dan 'auto').

        d. epsilon: margin toleransi untuk error dalam regresi ([0.01, 0.1, 0.2]).
    - Melakukan GridSearchCV dengan scoring negatif Mean Squared Error (neg_mean_squared_error), menggunakan 3-fold cross-validation, memanfaatkan semua core CPU (n_jobs=-1).
    - Menjalankan pencarian grid untuk menemukan kombinasi parameter terbaik.
    - Mengambil model terbaik dari hasil GridSearch dan melakukan prediksi pada data train dan test (Best parameters: {'C': 10, 'epsilon': 0.2, 'gamma': 'auto', 'kernel': 'rbf'})
    - Mengevaluasi model dengan metrik MAE, MSE, RMSE, dan R2.
2. Random Forest Regression
    - Mendefinisikan model RandomForestRegressor dengan random_state 55 dan n_jobs=-1.
    - Menyiapkan parameter grid tuning: 

        a. n_estimators: jumlah pohon yang akan dibuat ([50, 100]).

        b. max_depth: kedalaman maksimal pohon ([10, 16]).

        c. min_samples_split: jumlah minimum sampel untuk membagi node ([2, 5]).

        d. min_samples_leaf: jumlah minimum sampel pada daun pohon ([1, 2]).
    - Melakukan GridSearchCV dengan scoring negatif MSE, 3-fold cross-validation, dan pemanfaatan multi-core.
    - Mencari kombinasi parameter optimal.
    - Mengambil model terbaik dan melakukan evaluasi seperti pada SVR (Best parameters Random Forest: {'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}).
3. XGBoost Regression
    - Mendefinisikan model XGBRegressor dengan random_state 55 dan n_jobs=-1.
    - Menyiapkan parameter grid:

        a. n_estimators: jumlah pohon boosting ([50, 100]).

        b. learning_rate: laju pembelajaran ([0.05, 0.1]).

        c. max_depth: kedalaman maksimal pohon ([3, 5]).
    - Melakukan GridSearchCV dengan scoring negatif MSE dan 3-fold cross-validation.
    - Mencari parameter terbaik dan melakukan evaluasi menggunakan metrik yang sama (Best parameters XGBoost: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}).

## Evaluation
1. Mean Absolute Error (MAE)
    MAE adalah rata-rata dari nilai absolut selisih antara nilai aktual (observasi) dan nilai prediksi model. MAE menunjukkan seberapa jauh prediksi rata-rata dari nilai sebenarnya dalam satuan yang sama dengan target. Nilai MAE yang lebih kecil menunjukkan model dengan prediksi yang lebih akurat.

    Rumus:
   
    $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

3. Mean Squared Error (MSE)
    MSE adalah rata-rata dari kuadrat selisih antara nilai aktual dan prediksi. MSE memberikan penalti yang lebih besar terhadap kesalahan prediksi yang besar karena kuadratnya. MSE yang lebih kecil menunjukkan model yang lebih baik.

    Rumus:
   
    $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

5. Root Mean Squared Error (RMSE)
    RMSE adalah akar kuadrat dari MSE, sehingga berada dalam satuan yang sama dengan target (health score). RMSE sering digunakan untuk interpretasi kesalahan prediksi secara langsung. Nilai RMSE yang kecil mengindikasikan performa model yang baik.

    Rumus:

   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
  
7. R-squared (R²)
    R² menunjukkan proporsi variansi dalam fitur target (health score) yang dapat dijelaskan oleh model. Nilai R² berkisar dari 0 sampai 1 (atau bisa negatif jika model sangat buruk). Semakin mendekati 1, semakin baik model dalam menjelaskan variabilitas data.

   Rumus:
   
![Screenshot 2025-06-05 140649](https://github.com/user-attachments/assets/f059ebb3-d7d4-4518-b8e0-86c8e298a5f8)


### Tahapan Evaluasi
| Metric | Dataset | SVR       | RandomForest | XGBoost  |
|--------|---------|-----------|--------------|----------|
| MAE    | train   | 2.718152  | 1.716934     | 2.649905 |
|        | test    | 3.653235  | 4.030895     | 3.990092 |
| MSE    | train   | 15.535832 | 5.293374     | 11.260271|
|        | test    | 26.837114 | 31.596436    | 28.890275|
| RMSE   | train   | 3.941552  | 2.300733     | 3.355633 |
|        | test    | 5.180455  | 5.621071     | 5.374967 |
| R²     | train   | 0.915280  | 0.971134     | 0.938596 |
|        | test    | 0.862347  | 0.837935     | 0.851816 |
 
### Interpretasi
- Nilai MAE lebih kecil menunjukkan prediksi lebih akurat secara rata-rata. Pada data test, SVR menghasilkan MAE 3.65 yang lebih baik dibanding RandomForest (4.03) dan XGBoost (3.99). Ini berarti prediksi SVR lebih dekat secara rata-rata ke nilai sebenarnya.
- MSE memberi penalti lebih berat terhadap kesalahan yang besar. Nilai MSE test SVR (26.83) lebih rendah daripada RandomForest (31.59) dan XGBoost (28.89), menunjukkan model SVR memiliki kesalahan kuadrat yang lebih kecil. Namun, perbedaan tidak terlalu besar.
- RMSE test SVR (5.18) sedikit lebih rendah dibanding RandomForest (5.62) dan XGBoost (5.37), mengonfirmasi bahwa model SVR secara umum memiliki kesalahan prediksi yang lebih kecil.
- Nilai R² test SVR (0.86) tertinggi dibandingkan RandomForest (0.84) dan XGBoost (0.85), menandakan SVR mampu menjelaskan variansi health score paling baik pada data test. Nilai R2 di atas 0.8 menunjukkan model yang cukup baik dalam memprediksi target.

### Kesimpulan
- Model SVR menunjukkan performa terbaik secara keseluruhan pada data test dengan nilai MAE, MSE, RMSE yang paling rendah, serta nilai R2 tertinggi. Ini mengindikasikan SVR memiliki akurasi prediksi terbaik dan mampu menangkap pola data dengan baik tanpa overfitting berlebih.
- Model Random Forest dan XGBoost juga memberikan hasil yang baik, meskipun nilai kesalahan prediksi dan R² mereka sedikit lebih rendah dibandingkan SVR.
- Perbedaan metrik antara train dan test relatif kecil pada ketiga model, menunjukkan model tidak mengalami overfitting yang signifikan dan memiliki generalisasi yang baik.

## Daftar Pustaka
Buxton, O. M., & Marcelli, E. (2010). Short and long sleep durations and health risks among adults in the United States. Social Science & Medicine, 71(5), 1027–1036. https://doi.org/10.1016/j.socscimed.2010.05.041

Lee, I. M., Shiroma, E. J., Lobelo, F., et al. (2012). Effect of physical inactivity on major non-communicable diseases worldwide: An analysis of burden of disease and life expectancy. The Lancet, 380(9838), 219–229. https://doi.org/10.1016/S0140-6736(12)61031-9

Mokdad, A. H., Ballestros, K., Echko, M., et al. (2018). The state of US health, 1990–2016: Burden of diseases, injuries, and risk factors among US states. JAMA, 319(14), 1444–1472. https://doi.org/10.1001/jama.2018.0158

Wang, D., Hu, B., Hu, C., et al. (2023). Adherence to healthy lifestyle prior to infection and risk of post–COVID-19 condition: A prospective cohort study. JAMA Internal Medicine. Advance online publication. https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2800885

World Health Organization. (2021). Noncommunicable diseases: Risk factors. https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases
