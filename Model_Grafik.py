import pandas as pd
import matplotlib.pyplot as plt

# Memuat data dari file CSV
data = pd.read_csv('Rockpaper.csv')

# Mengelompokkan data berdasarkan kolom 'TeamName' dan menghitung jumlah kemunculannya
team_counts = data['TeamName'].value_counts()

# Mengambil 50 tim dengan jumlah kemunculan tertinggi
top_50_teams = team_counts.head(50)

# Konfigurasi font
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})

# Membuat grafik batang
plt.figure(figsize=(10, 6))
top_50_teams.plot(kind='bar', color='skyblue')
plt.title('Top 50 Team Hot News', fontsize=16)
plt.xlabel('Team Name', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Menyimpan grafik sebagai file gambar
plt.savefig('top_50_teams_bar_chart.png')

plt.show()
