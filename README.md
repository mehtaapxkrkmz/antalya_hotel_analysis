# Antalya Hotel Recommendation System
Bu proje, Antalya bölgesindeki otel verilerini ve kullanıcı yorumlarını analiz ederek, Doğal Dil İşleme (NLP) tabanlı bir öneri sistemi sunar. Projenin temel amacı, klasik yıldız puanlarının ötesine geçerek gerçek kullanıcı deneyimlerini matematiksel bir modele dönüştürmektir.

## Proje Hakkında
Geleneksel otel puanlama sistemleri genellikle yüzeysel kalabilmektedir. Bu projede, binlerce kullanıcı yorumu VADER Sentiment Analysis algoritması ile işlenmiş ve her otel için bir Duygu Skoru oluşturulmuştur. Bu skorlar, otelin mevcut reytingiyle harmanlanarak bir Hibrit Puanlama (Hybrid Score) sistemi geliştirilmiştir.

## Teknik Özellikler
Sentiment Analysis: Yorumların pozitif, negatif ve nötr olma durumlarının analizi.

Hybrid Scoring Engine: Duygu analizi ve kullanıcı puanlarının ağırlıklı ortalaması.

Word Cloud: Otel bazlı en çok öne çıkan geri bildirimlerin görselleştirilmesi.

Category Analysis: Yemek, temizlik ve hizmet kalitesi gibi spesifik alanlarda performans ölçümü.

## Kullanılan Teknolojiler
Python (Pandas, NumPy)

NLTK & VADER (Doğal Dil İşleme)

Streamlit (Web Arayüzü)

Matplotlib & WordCloud (Veri Görselleştirme)

## Veri Seti
Veri seti, Hotels.com'dan manuel olarak elde edilen Antalya'daki otellerden toplanan gerçek kullanıcı yorumlarını, otel özelliklerini (havuz, spa, otopark vb.) ve coğrafi koordinat bilgilerini içermektedir.

## Hugging Face Space Linki: [hugging_face](https://huggingface.co/spaces/mehtaapx/antalya-hotel-recommender)

## Kaggle Notebook Linkleri (İngilizce)
- Dataset: [dataset](https://www.kaggle.com/datasets/mehtapkorkmaz/antalya-2026-early-booking-and-sentiment-dataset)
- Notebbok 1: [notebook 1](https://www.kaggle.com/code/mehtapkorkmaz/antalya-tourism-market-analysis-eda)
- Notebook 2: [notebook 2](https://www.kaggle.com/code/mehtapkorkmaz/text-preprocessing)

## Hedef
Bu çalışma, bir veri bilimi projesi olarak; verinin çekilmesi, temizlenmesi, NLP teknikleriyle işlenmesi ve son kullanıcıya etkileşimli bir arayüz ile sunulması süreçlerini kapsamaktadır.

## Geliştirilecek Özellikler
Projeye eklenecek tüm yeni özellikler için: [hugging_face](https://huggingface.co/spaces/mehtaapx/antalya-hotel-recommender)
